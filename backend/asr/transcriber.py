"""
ASR layer — Whisper with transparent mock fallback.

If openai-whisper (and torch) is installed, Whisper runs locally.
Otherwise a mock transcriber is activated that returns a visible placeholder,
so the full pipeline can still be exercised and tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_WHISPER_HALLUCINATIONS = frozenset([
    "thank you for watching",
    "please subscribe",
    "www.",
    "[music]",
    "[applause]",
    "subtitles by",
    "amara.org",
])


@dataclass
class TranscriptionResult:
    text: str
    confidence: float       # 0.0 – 1.0
    language: str
    duration_s: float
    is_reliable: bool
    engine: str             # "whisper" | "mock"


class WhisperTranscriber:
    def __init__(self, model_size: str = "base", language: str = "en"):
        self._model = None
        self._model_size = model_size
        self._language = language
        self._try_load()

    def _try_load(self) -> None:
        try:
            import ssl, certifi
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        try:
            import whisper
            logger.info("Loading Whisper model '%s'…", self._model_size)
            self._model = whisper.load_model(self._model_size)
            logger.info("Whisper ready")
        except Exception as exc:
            logger.warning(
                "openai-whisper not available (%s). Mock ASR will be used. "
                "Run: pip install openai-whisper",
                exc,
            )

    @property
    def engine(self) -> str:
        return "whisper" if self._model else "mock"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        duration_s = len(audio) / sample_rate
        if self._model:
            return self._whisper_transcribe(audio, duration_s)
        return self._mock_transcribe(audio, duration_s)

    # ------------------------------------------------------------------

    def _whisper_transcribe(self, audio: np.ndarray, duration_s: float) -> TranscriptionResult:
        try:
            # Whisper expects float32 in [-1, 1]
            audio_f32 = audio.astype(np.float32)
            if audio_f32.max() > 1.0:
                audio_f32 = audio_f32 / 32768.0

            result = self._model.transcribe(
                audio_f32,
                language=self._language,
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                fp16=False,
            )

            text = result["text"].strip()
            segments = result.get("segments", [])

            if segments:
                avg_no_speech = float(
                    np.mean([s.get("no_speech_prob", 0.5) for s in segments])
                )
                confidence = 1.0 - avg_no_speech
            else:
                confidence = 0.5 if text else 0.0

            is_reliable = (
                bool(text)
                and confidence >= 0.45
                and not self._is_hallucination(text)
            )

            return TranscriptionResult(
                text=text,
                confidence=round(confidence, 3),
                language=result.get("language", self._language),
                duration_s=duration_s,
                is_reliable=is_reliable,
                engine="whisper",
            )

        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc, exc_info=True)
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self._language,
                duration_s=duration_s,
                is_reliable=False,
                engine="whisper",
            )

    def _mock_transcribe(self, audio: np.ndarray, duration_s: float) -> TranscriptionResult:
        """
        Deterministic mock.  Real speech energy → plausible placeholder so the
        validation layer can be exercised without a GPU.
        """
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        has_signal = rms > 0.02 and duration_s > 0.8   # only trigger on real, audible speech

        text = "[MOCK — install openai-whisper for real ASR]" if has_signal else ""
        return TranscriptionResult(
            text=text,
            confidence=0.7 if has_signal else 0.0,
            language="en",
            duration_s=duration_s,
            is_reliable=has_signal,
            engine="mock",
        )

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        t = text.lower()
        return any(h in t for h in _WHISPER_HALLUCINATIONS)
