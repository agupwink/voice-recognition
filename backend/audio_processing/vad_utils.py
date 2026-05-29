"""
VAD utilities — speech segment extraction.

Tries Silero VAD on first import; falls back to energy-based VAD silently.
All public functions are exception-safe and return gracefully on empty audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Silero availability probe ─────────────────────────────────────────────────

SILERO_AVAILABLE: bool = False
_silero_model = None
_silero_utils = None

try:
    import torch  # noqa: F401 — just checking availability

    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    SILERO_AVAILABLE = True
    logger.info("Silero VAD loaded successfully.")
except Exception as _e:
    logger.warning("Silero VAD unavailable (%s); using energy-based VAD.", _e)
    SILERO_AVAILABLE = False


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_s: float


# ── Energy-based VAD (fallback) ───────────────────────────────────────────────

def _energy_vad(
    audio: np.ndarray,
    sr: int,
    threshold: float,
) -> List[SpeechSegment]:
    """30 ms frames, RMS threshold with hysteresis (0.5x for end-of-speech)."""
    if len(audio) == 0:
        return []

    frame_len = int(sr * 0.030)  # 30 ms
    if frame_len == 0:
        return []

    end_threshold = threshold * 0.5  # hysteresis

    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))

    in_speech = False
    seg_start = 0
    segments: List[SpeechSegment] = []

    for i, rms in enumerate(rms_per_frame):
        if not in_speech and rms >= threshold:
            in_speech = True
            seg_start = i * frame_len
        elif in_speech and rms < end_threshold:
            seg_end = (i + 1) * frame_len
            segments.append(
                SpeechSegment(
                    start_sample=seg_start,
                    end_sample=seg_end,
                    start_s=seg_start / sr,
                    end_s=seg_end / sr,
                    duration_s=(seg_end - seg_start) / sr,
                )
            )
            in_speech = False

    # Close any open segment at end of audio
    if in_speech:
        seg_end = len(audio)
        segments.append(
            SpeechSegment(
                start_sample=seg_start,
                end_sample=seg_end,
                start_s=seg_start / sr,
                end_s=seg_end / sr,
                duration_s=(seg_end - seg_start) / sr,
            )
        )

    return segments


# ── Silero-based VAD ──────────────────────────────────────────────────────────

def _silero_vad(
    audio: np.ndarray,
    sr: int,
    threshold: float,
) -> List[SpeechSegment]:
    """Chunk-based Silero VAD using 512-sample windows at 16 kHz."""
    if len(audio) == 0:
        return []

    import torch  # already imported if SILERO_AVAILABLE is True

    window = 512
    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    in_speech = False
    seg_start = 0
    segments: List[SpeechSegment] = []

    for i in range(0, len(audio) - window + 1, window):
        chunk = audio_tensor[i : i + window]
        with torch.no_grad():
            prob = float(_silero_model(chunk, sr).item())

        if not in_speech and prob >= threshold:
            in_speech = True
            seg_start = i
        elif in_speech and prob < threshold * 0.5:
            seg_end = i + window
            segments.append(
                SpeechSegment(
                    start_sample=seg_start,
                    end_sample=seg_end,
                    start_s=seg_start / sr,
                    end_s=seg_end / sr,
                    duration_s=(seg_end - seg_start) / sr,
                )
            )
            in_speech = False

    if in_speech:
        seg_end = len(audio)
        segments.append(
            SpeechSegment(
                start_sample=seg_start,
                end_sample=seg_end,
                start_s=seg_start / sr,
                end_s=seg_end / sr,
                duration_s=(seg_end - seg_start) / sr,
            )
        )

    return segments


# ── Public API ────────────────────────────────────────────────────────────────

def get_speech_segments(
    audio: np.ndarray,
    sr: int = 16_000,
    threshold: float = 0.5,
) -> List[SpeechSegment]:
    """Return a list of SpeechSegment objects detected in *audio*.

    Uses Silero VAD when available, otherwise energy-based VAD.
    Never raises — returns [] on any error or empty input.
    """
    if audio is None or len(audio) == 0:
        return []

    try:
        if SILERO_AVAILABLE:
            return _silero_vad(audio, sr, threshold)
        return _energy_vad(audio, sr, threshold)
    except Exception as exc:
        logger.warning("get_speech_segments failed (%s); returning [].", exc)
        return []


def filter_speech_audio(
    audio: np.ndarray,
    sr: int = 16_000,
    threshold: float = 0.5,
    pad_ms: int = 100,
) -> Tuple[np.ndarray, float]:
    """Return (speech_only_audio, speech_ratio).

    Concatenates all detected speech segments, with *pad_ms* ms of padding
    around each segment.  Falls back to the original audio on any error.
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32), 0.0

    try:
        segments = get_speech_segments(audio, sr, threshold)

        if not segments:
            return audio, 0.0

        pad_samples = int(sr * pad_ms / 1000)
        chunks: List[np.ndarray] = []

        for seg in segments:
            start = max(0, seg.start_sample - pad_samples)
            end   = min(len(audio), seg.end_sample + pad_samples)
            chunks.append(audio[start:end])

        filtered = np.concatenate(chunks)
        speech_ratio = len(filtered) / len(audio)

        return filtered, float(speech_ratio)

    except Exception as exc:
        logger.warning("filter_speech_audio failed (%s); returning original audio.", exc)
        return audio, 1.0
