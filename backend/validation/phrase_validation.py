"""
Passphrase validation — compares ASR transcript against expected passphrase.

Design goals
────────────
• Lenient: only rejects obvious mismatches so legitimate users aren't locked out.
• Unicode-safe normalisation before comparison.
• Two complementary similarity signals averaged with fixed weights.
"""

from __future__ import annotations

import logging
import unicodedata
from difflib import SequenceMatcher
from string import punctuation
from typing import Tuple

logger = logging.getLogger(__name__)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation, normalize unicode, collapse whitespace."""
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


# ── Similarity ────────────────────────────────────────────────────────────────

def compare_phrase_similarity(hypothesis: str, reference: str) -> float:
    """Return a similarity score in [0.0, 1.0].

    Score = 0.6 * word_overlap + 0.4 * sequence_similarity
    """
    hyp = normalize_phrase(hypothesis)
    ref = normalize_phrase(reference)

    if not ref:
        # No reference to compare against — treat as passing
        return 1.0

    hyp_words = hyp.split()
    ref_words = ref.split()

    # Word overlap: fraction of reference words present in hypothesis
    if not ref_words:
        word_score = 1.0
    else:
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        matching = len(ref_set & hyp_set)
        word_score = matching / len(ref_set)

    # Sequence similarity
    seq_score = SequenceMatcher(None, hyp, ref).ratio()

    combined = 0.6 * word_score + 0.4 * seq_score
    return float(max(0.0, min(1.0, combined)))


# ── Public API ────────────────────────────────────────────────────────────────

def validate_phrase(
    transcript: str,
    expected: str,
    threshold: float = 0.50,
) -> Tuple[bool, float, str]:
    """Compare *transcript* against *expected* passphrase.

    Returns
    ───────
    (passed, score, feedback)
        passed   – True when score >= threshold
        score    – float in [0.0, 1.0]
        feedback – empty string on pass; human-readable message on fail
    """
    try:
        score = compare_phrase_similarity(transcript, expected)

        if score >= threshold:
            return True, score, ""

        # Build a helpful message
        norm_expected = normalize_phrase(expected)
        feedback = (
            f"Passphrase not recognised (score {score:.0%}). "
            f"Please say: \"{norm_expected}\"."
        )
        logger.debug("Phrase validation failed: score=%.3f transcript=%r expected=%r",
                     score, transcript, expected)
        return False, score, feedback

    except Exception as exc:
        logger.warning("validate_phrase raised an exception (%s); failing open.", exc)
        return True, 0.0, ""
