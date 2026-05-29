"""
Constraint validation engine.

Computes Word Error Rate (WER) and optionally a phonetic similarity score.
Returns a structured ValidationResult with a feedback_code used by the
state machine to generate user-facing messages.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_match: bool
    similarity_score: float     # 0.0 – 1.0  (higher = better)
    wer: float                  # Word Error Rate (lower = better)
    normalised_transcript: str
    normalised_target: str
    feedback_code: Optional[str] = None


class TextValidator:
    def __init__(self, min_match_score: float = 0.75):
        self.min_match_score = min_match_score
        self._jiwer = self._try_jiwer()
        self._jellyfish = self._try_jellyfish()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, transcript: str, target: str) -> ValidationResult:
        nt = self._normalise(transcript)
        nr = self._normalise(target)

        if not nt:
            return ValidationResult(
                is_match=False,
                similarity_score=0.0,
                wer=1.0,
                normalised_transcript=nt,
                normalised_target=nr,
                feedback_code="no_speech",
            )

        wer = self._wer(nt, nr)
        score = max(0.0, 1.0 - wer)

        # For borderline cases blend in phonetic similarity
        margin = 0.12
        if abs(score - self.min_match_score) <= margin:
            phon = self._phonetic_score(nt, nr)
            score = 0.6 * score + 0.4 * phon
            score = max(0.0, min(1.0, score))
            # Recalculate wer from blended score
            wer = 1.0 - score

        feedback_code = None
        if score < self.min_match_score:
            word_ratio = len(nt.split()) / max(len(nr.split()), 1)
            if word_ratio < 0.5:
                feedback_code = "too_short"
            elif word_ratio > 1.6:
                feedback_code = "extra_words"
            else:
                feedback_code = "wrong_sentence"

        return ValidationResult(
            is_match=score >= self.min_match_score,
            similarity_score=round(score, 4),
            wer=round(wer, 4),
            normalised_transcript=nt,
            normalised_target=nr,
            feedback_code=feedback_code,
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        # Keep apostrophes, strip everything else that isn't alphanumeric/space
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # WER
    # ------------------------------------------------------------------

    def _wer(self, hypothesis: str, reference: str) -> float:
        if self._jiwer:
            try:
                import jiwer
                return float(jiwer.wer(reference, hypothesis))
            except Exception:
                pass
        return self._basic_wer(hypothesis, reference)

    @staticmethod
    def _basic_wer(hyp: str, ref: str) -> float:
        ref_w = ref.split()
        hyp_w = hyp.split()
        if not ref_w:
            return 0.0 if not hyp_w else 1.0

        n, m = len(ref_w), len(hyp_w)
        # O(n*m) edit distance, word-level
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, m + 1):
                cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
                dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)

        return dp[m] / n

    # ------------------------------------------------------------------
    # Phonetic similarity
    # ------------------------------------------------------------------

    def _phonetic_score(self, transcript: str, target: str) -> float:
        if self._jellyfish:
            try:
                import jellyfish
                ref_words = target.split()
                hyp_words = transcript.split()
                if not ref_words:
                    return 0.0
                matched = sum(
                    1 for rw in ref_words
                    if any(jellyfish.soundex(rw) == jellyfish.soundex(hw) for hw in hyp_words)
                )
                return matched / len(ref_words)
            except Exception:
                pass
        # Fallback: character-level SequenceMatcher
        return SequenceMatcher(None, transcript, target).ratio()

    # ------------------------------------------------------------------
    # Optional-dependency probes
    # ------------------------------------------------------------------

    @staticmethod
    def _try_jiwer() -> bool:
        try:
            import jiwer  # noqa: F401
            return True
        except ImportError:
            logger.info("jiwer not installed — using built-in WER")
            return False

    @staticmethod
    def _try_jellyfish() -> bool:
        try:
            import jellyfish  # noqa: F401
            return True
        except ImportError:
            logger.info("jellyfish not installed — phonetic fallback active")
            return False
