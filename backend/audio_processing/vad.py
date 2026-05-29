"""Voice Activity Detection — Silero VAD with energy-based fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    has_speech: bool
    speech_probability: float   # 0.0 – 1.0
    rms_energy: float


# ---------------------------------------------------------------------------
# Energy-based VAD (always available, no ML dependencies)
# ---------------------------------------------------------------------------

class EnergyVAD:
    def __init__(self, threshold: float = 0.008):
        self.threshold = threshold
        self._history: list[bool] = []
        self._history_len = 8

    def process(self, audio: np.ndarray) -> VADResult:
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        has_speech = rms > self.threshold

        self._history.append(has_speech)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        probability = sum(self._history) / len(self._history)
        return VADResult(has_speech=has_speech, speech_probability=probability, rms_energy=rms)


# ---------------------------------------------------------------------------
# Silero VAD (torch-based, higher accuracy)
# ---------------------------------------------------------------------------

class SileroVAD:
    def __init__(self, sample_rate: int = 16000):
        self._model = None
        self._sample_rate = sample_rate
        self._load()

    def _load(self) -> None:
        try:
            import torch
            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            self._model = model
            logger.info("Silero VAD loaded")
        except Exception as exc:
            logger.warning("Silero VAD unavailable (%s) — falling back to energy VAD", exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    def process(self, audio: np.ndarray) -> VADResult:
        import torch

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        tensor = torch.from_numpy(audio.astype(np.float32))
        with torch.no_grad():
            prob = float(self._model(tensor, self._sample_rate).item())

        return VADResult(has_speech=prob > 0.5, speech_probability=prob, rms_energy=rms)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vad(energy_threshold: float = 0.008) -> EnergyVAD | SileroVAD:
    vad = SileroVAD()
    if vad.available:
        return vad
    return EnergyVAD(threshold=energy_threshold)
