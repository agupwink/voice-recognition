from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SessionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    VALIDATING = "VALIDATING"
    PAUSED = "PAUSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


# State → allowed next states
_TRANSITIONS: dict[SessionState, list[SessionState]] = {
    SessionState.IDLE:       [SessionState.LISTENING],
    SessionState.LISTENING:  [SessionState.VALIDATING, SessionState.PAUSED],
    SessionState.VALIDATING: [SessionState.ACCEPTED, SessionState.REJECTED, SessionState.PAUSED, SessionState.LISTENING],
    SessionState.PAUSED:     [SessionState.LISTENING, SessionState.REJECTED],
    SessionState.ACCEPTED:   [],
    SessionState.REJECTED:   [],
}


@dataclass
class Session:
    session_id: str
    target_sentence: str
    state: SessionState = SessionState.IDLE
    attempt_count: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    pause_reason: Optional[str] = None
    transcript_history: list = field(default_factory=list)

    def transition(self, new_state: SessionState, reason: Optional[str] = None) -> bool:
        if new_state not in _TRANSITIONS.get(self.state, []):
            return False
        self.state = new_state
        self.pause_reason = reason if new_state == SessionState.PAUSED else None
        self.last_activity = time.time()
        return True

    def is_terminal(self) -> bool:
        return self.state in (SessionState.ACCEPTED, SessionState.REJECTED)

    def can_resume(self) -> bool:
        return self.state == SessionState.PAUSED

    def exhausted(self) -> bool:
        return self.attempt_count >= self.max_attempts
