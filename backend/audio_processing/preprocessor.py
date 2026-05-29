"""
Audio front-end pipeline.

Accumulates incoming PCM chunks, runs VAD frame-by-frame, and emits a
complete AudioSegment whenever a speech burst ends (silence detected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .noise_detector import NoiseDetector, NoiseResult
from .vad import VADResult, create_vad

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    audio: np.ndarray           # float32, 16 kHz
    duration_s: float
    vad: VADResult
    noise: NoiseResult
    processable: bool
    rejection_reason: Optional[str] = None  # set when processable=False


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_ms: int = 300,
        max_speech_s: float = 12.0,
        silence_frames: int = 3,
        pre_pad_ms: int = 150,
        snr_threshold_db: float = 3.0,
        energy_threshold: float = 0.002,
    ):
        self._sr = sample_rate
        self._min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._max_speech_samples = int(sample_rate * max_speech_s)
        self._silence_frames_needed = silence_frames
        self._pre_pad_samples = int(sample_rate * pre_pad_ms / 1000)

        self._vad = create_vad(energy_threshold=energy_threshold)
        self._noise = NoiseDetector(snr_threshold_db=snr_threshold_db)

        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._reset_state()

    def push_chunk(self, chunk: np.ndarray) -> Optional[AudioSegment]:
        """
        Feed a PCM chunk (float32, 16 kHz).
        Returns an AudioSegment when a complete speech segment is ready,
        otherwise None.
        """
        vad_result = self._vad.process(chunk)

        if vad_result.has_speech:
            if not self._in_speech:
                self._in_speech = True
                pre_roll = self._rolling_buf[-self._pre_pad_samples:] if len(self._rolling_buf) else np.array([], dtype=np.float32)
                self._speech_buf = np.concatenate([pre_roll, chunk])
                logger.debug("Speech onset  rms=%.5f", vad_result.rms_energy)
            else:
                self._speech_buf = np.concatenate([self._speech_buf, chunk])
            self._silence_count = 0

            # Force-flush when speech buffer hits the max duration
            if len(self._speech_buf) >= self._max_speech_samples:
                logger.info("Max speech duration reached — force-flushing")
                segment = self._finalise(vad_result)
                self._reset_state()
                return segment
        else:
            if self._in_speech:
                self._speech_buf = np.concatenate([self._speech_buf, chunk])
                self._silence_count += 1
                logger.debug("Silence frame %d/%d", self._silence_count, self._silence_frames_needed)

                if self._silence_count >= self._silence_frames_needed:
                    segment = self._finalise(vad_result)
                    self._reset_state()
                    return segment

        # Update rolling pre-roll buffer (last pre_pad_samples)
        self._rolling_buf = np.concatenate([self._rolling_buf, chunk])
        keep = self._pre_pad_samples * 2
        if len(self._rolling_buf) > keep:
            self._rolling_buf = self._rolling_buf[-keep:]

        return None

    def flush(self) -> Optional[AudioSegment]:
        """Force-emit whatever speech is buffered (e.g., on WebSocket close)."""
        if self._in_speech and len(self._speech_buf) >= self._min_speech_samples:
            dummy_vad = self._vad.process(self._speech_buf[-160:])
            seg = self._finalise(dummy_vad)
            self._reset_state()
            return seg
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._in_speech = False
        self._silence_count = 0
        self._speech_buf = np.array([], dtype=np.float32)
        self._rolling_buf = np.array([], dtype=np.float32)

    def _finalise(self, vad_result: VADResult) -> AudioSegment:
        audio = self._speech_buf.copy()
        duration_s = len(audio) / self._sr
        noise_result = self._noise.analyse(audio)

        processable = True
        reason: Optional[str] = None

        if len(audio) < self._min_speech_samples:
            processable = False
            reason = "too_short"
        elif not noise_result.is_acceptable:
            processable = False
            reason = "high_noise"
        elif vad_result.speech_probability < 0.25:
            processable = False
            reason = "low_speech_ratio"

        return AudioSegment(
            audio=audio,
            duration_s=round(duration_s, 3),
            vad=vad_result,
            noise=noise_result,
            processable=processable,
            rejection_reason=reason,
        )
