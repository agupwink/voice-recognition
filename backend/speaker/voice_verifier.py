"""
Speaker verification using Resemblyzer (GE2E pre-trained model).

Resemblyzer maps any audio clip to a 256-dim d-vector embedding.
Cosine similarity between two embeddings from the same speaker is
typically > 0.75; different speakers fall below that.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75   # tune upward for stricter security


class VoiceVerifier:
    def __init__(self):
        self._encoder = None
        self._load()

    def _load(self):
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            logger.info("Resemblyzer VoiceEncoder loaded (256-dim GE2E embeddings)")
        except Exception as exc:
            logger.error("Could not load Resemblyzer: %s", exc)

    @property
    def available(self) -> bool:
        return self._encoder is not None

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Return a 256-dim voice embedding, or None on failure."""
        if not self.available:
            return None
        try:
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(audio.astype(np.float32), source_sr=sample_rate)
            if len(wav) < 1600:   # need at least 100 ms after preprocessing
                return None
            return self._encoder.embed_utterance(wav)
        except Exception as exc:
            logger.error("embed() failed: %s", exc)
            return None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def verify(
        self,
        stored_embedding: np.ndarray,
        probe_audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[bool, float]:
        """
        Returns (passed, similarity_score).
        passed=True when the probe voice matches the stored enrollment.
        """
        probe_emb = self.embed(probe_audio, sample_rate)
        if probe_emb is None:
            return False, 0.0
        score = self.cosine_similarity(stored_embedding, probe_emb)
        return score >= SIMILARITY_THRESHOLD, round(score, 4)


# Module-level singleton
_verifier: Optional[VoiceVerifier] = None


def get_verifier() -> VoiceVerifier:
    global _verifier
    if _verifier is None:
        _verifier = VoiceVerifier()
    return _verifier
