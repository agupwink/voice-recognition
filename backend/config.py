from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # Audio pipeline
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION_MS: int = 100                  # frontend sends 100ms chunks
    MIN_SPEECH_DURATION_MS: int = 800             # ignore anything shorter than 0.8 s
    MAX_SPEECH_DURATION_S: int = 15
    PRE_SPEECH_PAD_MS: int = 200                  # include this much audio before speech

    # VAD
    VAD_ENERGY_THRESHOLD: float = 0.004           # RMS threshold for energy-based VAD
    VAD_SILENCE_FRAMES: int = 6                   # 600 ms of silence to close a segment

    # Noise
    SNR_THRESHOLD_DB: float = 3.0

    # ASR
    WHISPER_MODEL: str = "base"
    WHISPER_LANGUAGE: str = "en"
    ASR_CONFIDENCE_THRESHOLD: float = 0.45

    # Validation
    MIN_MATCH_SCORE: float = 0.75
    MAX_RETRIES: int = 3

    # Session
    SESSION_TIMEOUT_S: float = 120.0

    # Challenge sentences
    CHALLENGE_SENTENCES: List[str] = field(default_factory=lambda: [
        "My voice is my password verify me",
        "The quick brown fox jumps over the lazy dog",
        "Security systems protect important information carefully",
        "Please verify my identity using this spoken phrase",
        "Authentication requires both accuracy and clarity",
        "I have a secure voice authentication system",
        "The blue bird sings a happy song in the morning",
        "She sells seashells by the seashore every day",
        "Artificial intelligence can recognize human speech",
        "Voice biometrics provide an extra layer of security",
    ])


settings = Settings()
