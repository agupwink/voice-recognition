"""In-memory user store. Replace with a database in production."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VoiceUser:
    username: str
    display_name: str
    passphrase_text: str        # what the user typed as their chosen phrase
    voice_transcript: str       # what Whisper heard during enrollment
    voice_embedding: np.ndarray # 256-dim Resemblyzer d-vector
    engine: str                 # "whisper" | "mock"
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None


_store: dict[str, VoiceUser] = {}


def exists(username: str) -> bool:
    return username.lower() in _store


def create(user: VoiceUser) -> None:
    _store[user.username.lower()] = user


def get(username: str) -> Optional[VoiceUser]:
    return _store.get(username.lower())


def touch_login(username: str) -> None:
    u = _store.get(username.lower())
    if u:
        u.last_login = time.time()


def all_usernames() -> list[str]:
    return [u.display_name for u in _store.values()]
