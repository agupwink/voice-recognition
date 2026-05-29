"""REST endpoints: session lifecycle."""

from __future__ import annotations

import random
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from state_machine.session import Session, SessionState

router = APIRouter()

# In-memory store — replace with Redis/DB in production
sessions: dict[str, Session] = {}


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class SessionStartResponse(BaseModel):
    session_id: str
    sentence: str
    max_attempts: int
    websocket_url: str


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    sentence: str
    attempt_count: int
    max_attempts: int
    pause_reason: Optional[str]
    elapsed_s: float


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/session/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid.uuid4())
    sentence = random.choice(settings.CHALLENGE_SENTENCES)
    sessions[session_id] = Session(
        session_id=session_id,
        target_sentence=sentence,
        max_attempts=settings.MAX_RETRIES,
    )
    return SessionStartResponse(
        session_id=session_id,
        sentence=sentence,
        max_attempts=settings.MAX_RETRIES,
        websocket_url=f"/ws/voice/{session_id}",
    )


@router.get("/session/status/{session_id}", response_model=SessionStatusResponse)
async def session_status(session_id: str):
    s = _get_or_404(session_id)
    return SessionStatusResponse(
        session_id=session_id,
        state=s.state.value,
        sentence=s.target_sentence,
        attempt_count=s.attempt_count,
        max_attempts=s.max_attempts,
        pause_reason=s.pause_reason,
        elapsed_s=round(time.time() - s.created_at, 1),
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "deleted"}


# ------------------------------------------------------------------
# Helpers (used by WebSocket handler)
# ------------------------------------------------------------------

def _get_or_404(session_id: str) -> Session:
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s
