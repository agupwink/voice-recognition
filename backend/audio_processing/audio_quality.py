"""
Audio quality analysis for voice authentication.

Analyses RMS energy, SNR, clipping and duration before processing.
Designed to fail open — any exception returns QualityReport(is_acceptable=True).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ── Report dataclass ──────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    is_acceptable: bool
    rms_energy: float
    snr_db: float
    clipping_ratio: float
    duration_s: float
    issues: List[str] = field(default_factory=list)
    feedback: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_snr(audio: np.ndarray, sr: int) -> float:
    """SNR estimate: 10th vs 90th percentile of 25 ms frame powers."""
    frame_len = max(1, int(sr * 0.025))
    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return 0.0

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    powers = np.mean(frames ** 2, axis=1)
    powers = powers[powers > 0]  # avoid log(0)

    if len(powers) < 2:
        return 0.0

    noise_power  = float(np.percentile(powers, 10))
    signal_power = float(np.percentile(powers, 90))

    if noise_power <= 0:
        return 60.0  # essentially no noise floor

    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(snr)


def _clipping_ratio(audio: np.ndarray, threshold: float) -> float:
    """Fraction of samples whose absolute value exceeds *threshold*."""
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_audio_quality(
    audio: np.ndarray,
    sr: int = 16_000,
    min_rms: float = 0.002,
    snr_threshold_db: float = 8.0,
    clipping_threshold: float = 0.95,
    min_duration_s: float = 0.5,
) -> QualityReport:
    """Analyse *audio* and return a QualityReport.

    On any exception the function returns QualityReport(is_acceptable=True, ...)
    so that authentication is never blocked by a quality-check crash.
    """
    _fallback = QualityReport(
        is_acceptable=True,
        rms_energy=0.0,
        snr_db=0.0,
        clipping_ratio=0.0,
        duration_s=0.0,
        issues=[],
        feedback="",
    )

    try:
        if audio is None or len(audio) == 0:
            return _fallback

        duration_s     = len(audio) / sr
        rms_energy     = float(np.sqrt(np.mean(audio ** 2)))
        snr_db         = _safe_snr(audio, sr)
        clip_ratio     = _clipping_ratio(audio, clipping_threshold)

        issues: List[str] = []

        if duration_s < min_duration_s:
            issues.append("too_short")
        if rms_energy < min_rms:
            issues.append("too_quiet")
        if snr_db < snr_threshold_db:
            issues.append("noisy")
        if clip_ratio > 0.01:
            issues.append("clipping")

        # Build a single human-readable feedback string (first critical issue wins)
        feedback = ""
        if "too_short" in issues:
            feedback = "Recording too short — please speak your full passphrase."
        elif "too_quiet" in issues:
            feedback = "Audio too quiet — please speak louder or move closer to the microphone."
        elif "clipping" in issues:
            feedback = "Audio is clipping — please lower your microphone volume."
        elif "noisy" in issues:
            feedback = "Background noise detected — try recording in a quieter environment."

        is_acceptable = len(issues) == 0

        return QualityReport(
            is_acceptable=is_acceptable,
            rms_energy=rms_energy,
            snr_db=snr_db,
            clipping_ratio=clip_ratio,
            duration_s=duration_s,
            issues=issues,
            feedback=feedback,
        )

    except Exception as exc:
        logger.warning("analyse_audio_quality raised an exception (%s); failing open.", exc)
        return _fallback
