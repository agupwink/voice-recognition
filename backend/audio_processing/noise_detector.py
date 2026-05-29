"""SNR-based noise detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseResult:
    snr_db: float
    noise_level: str        # "clean" | "moderate" | "noisy"
    is_acceptable: bool


class NoiseDetector:
    def __init__(self, snr_threshold_db: float = 10.0):
        self.snr_threshold_db = snr_threshold_db

    def analyse(self, audio: np.ndarray) -> NoiseResult:
        """Estimate SNR by comparing loudest vs quietest short frames."""
        audio = audio.astype(np.float64)
        frame_len = max(1, len(audio) // 20)   # ~5% of segment per frame

        frame_powers = [
            np.mean(audio[i: i + frame_len] ** 2)
            for i in range(0, len(audio) - frame_len, frame_len)
        ]
        if not frame_powers:
            return NoiseResult(snr_db=0.0, noise_level="noisy", is_acceptable=False)

        signal_power = np.percentile(frame_powers, 90)
        noise_power  = np.percentile(frame_powers, 10)
        noise_power  = max(noise_power, 1e-12)

        snr_db = 10.0 * np.log10(signal_power / noise_power)

        if snr_db >= self.snr_threshold_db + 8:
            noise_level = "clean"
        elif snr_db >= self.snr_threshold_db:
            noise_level = "moderate"
        else:
            noise_level = "noisy"

        return NoiseResult(
            snr_db=round(snr_db, 1),
            noise_level=noise_level,
            is_acceptable=noise_level != "noisy",
        )
