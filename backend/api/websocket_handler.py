"""
WebSocket handler — the core runtime loop.

Protocol
--------
Client  → Server  : binary  — raw int16-LE PCM at 16 kHz, 100 ms chunks
Client  → Server  : text/JSON  — control messages  { "type": "resume" | "ping" | "flush" }
Server  → Client  : text/JSON  — structured messages (see _msg_* helpers)

Flow per session
----------------
    IDLE → LISTENING  (on WS connect)
    LISTENING → VALIDATING  (complete speech segment detected)
    VALIDATING → ACCEPTED | PAUSED  (based on ASR + validation)
    PAUSED → LISTENING  (client sends { "type": "resume" })
    PAUSED / VALIDATING → REJECTED  (attempts exhausted)
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from api.routes import sessions
from audio_processing.preprocessor import AudioPreprocessor, AudioSegment
from asr.transcriber import WhisperTranscriber
from config import settings
from state_machine.session import Session, SessionState
from validation.text_validator import TextValidator

logger = logging.getLogger(__name__)

# CPU-bound work (VAD, Whisper) runs in this pool so asyncio isn't blocked
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="asr")

# Module-level singletons — loaded once on first use
_transcriber: Optional[WhisperTranscriber] = None
_validator: Optional[TextValidator] = None


def _get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber(
            model_size=settings.WHISPER_MODEL,
            language=settings.WHISPER_LANGUAGE,
        )
    return _transcriber


def _get_validator() -> TextValidator:
    global _validator
    if _validator is None:
        _validator = TextValidator(min_match_score=settings.MIN_MATCH_SCORE)
    return _validator


# ------------------------------------------------------------------
# Feedback catalogue
# ------------------------------------------------------------------

_FEEDBACK: dict[str, str] = {
    "high_noise":       "Background noise detected. Please move to a quieter place.",
    "too_short":        "Speech too short. Please speak the complete sentence.",
    "low_speech_ratio": "Speech unclear. Please speak directly into the microphone.",
    "no_speech":        "No clear speech detected. Try again.",
    "low_confidence":   "Speech unclear. Please speak slowly and clearly.",
    "wrong_sentence":   "Please speak ONLY the sentence shown on screen.",
    "extra_words":      "Too many words detected. Speak only the shown sentence.",
    "too_many_attempts":"Too many failed attempts. Authentication rejected.",
    "timeout":          "Session timed out. Please start a new session.",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pcm_to_float32(raw: bytes) -> Optional[np.ndarray]:
    n = len(raw) // 2
    if n == 0:
        return None
    samples = struct.unpack(f"<{n}h", raw)
    return np.array(samples, dtype=np.float32) / 32768.0


async def _run_in_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def _send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _send_status(ws: WebSocket, session: Session, message: str = "") -> None:
    await _send(ws, {
        "type": "status",
        "state": session.state.value,
        "attempt_count": session.attempt_count,
        "max_attempts": session.max_attempts,
        "pause_reason": session.pause_reason,
        "message": message,
    })


async def _send_feedback(ws: WebSocket, code: str) -> None:
    await _send(ws, {
        "type": "feedback",
        "code": code,
        "message": _FEEDBACK.get(code, "Please try again."),
    })


# ------------------------------------------------------------------
# Main WebSocket endpoint
# ------------------------------------------------------------------

async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    session: Optional[Session] = sessions.get(session_id)
    if not session:
        await _send(websocket, {"type": "error", "message": "Session not found"})
        await websocket.close(code=4404)
        return

    preprocessor = AudioPreprocessor(
        sample_rate=settings.SAMPLE_RATE,
        min_speech_ms=settings.MIN_SPEECH_DURATION_MS,
        silence_frames=settings.VAD_SILENCE_FRAMES,
        pre_pad_ms=settings.PRE_SPEECH_PAD_MS,
        snr_threshold_db=settings.SNR_THRESHOLD_DB,
        energy_threshold=settings.VAD_ENERGY_THRESHOLD,
    )
    transcriber = _get_transcriber()
    validator = _get_validator()

    session.transition(SessionState.LISTENING)
    await _send_status(websocket, session, "Listening… please speak the sentence shown.")

    try:
        while not session.is_terminal():
            try:
                raw = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.SESSION_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await _send_feedback(websocket, "timeout")
                session.state = SessionState.REJECTED
                break

            if raw.get("type") == "websocket.disconnect":
                break

            # ── Control messages (JSON text) ──────────────────────────
            if "text" in raw:
                await _handle_control(raw["text"], websocket, session, preprocessor)
                continue

            # ── Audio frames (binary PCM) ─────────────────────────────
            if "bytes" not in raw:
                continue

            # Skip audio if not in LISTENING state
            if session.state != SessionState.LISTENING:
                continue

            audio_chunk = _pcm_to_float32(raw["bytes"])
            if audio_chunk is None:
                continue

            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            logger.debug("chunk rms=%.5f  bytes=%d", rms, len(raw["bytes"]))

            # VAD + noise — offload so asyncio stays unblocked
            segment: Optional[AudioSegment] = await _run_in_executor(
                preprocessor.push_chunk, audio_chunk
            )

            if segment is None:
                # Send live mic level so the UI can show an activity indicator
                await _send(websocket, {"type": "audio_level", "rms": round(rms, 5)})
                continue  # still accumulating

            # ── Got a complete speech segment ─────────────────────────
            session.transition(SessionState.VALIDATING)
            await _send_status(websocket, session, "Validating speech…")

            segment_rms = float(np.sqrt(np.mean(segment.audio ** 2)))

            if not segment.processable:
                await _handle_bad_segment(websocket, session, preprocessor, segment.rejection_reason)
                continue

            # ── ASR ───────────────────────────────────────────────────
            txn = await _run_in_executor(
                transcriber.transcribe, segment.audio, settings.SAMPLE_RATE
            )

            if not txn.is_reliable:
                await _handle_unreliable_asr(websocket, session, preprocessor, segment_rms)
                continue

            # ── Validation ────────────────────────────────────────────
            result = validator.validate(txn.text, session.target_sentence)
            session.transcript_history.append({
                "attempt": session.attempt_count + 1,
                "transcript": txn.text,
                "score": result.similarity_score,
                "engine": txn.engine,
            })

            if result.is_match:
                session.state = SessionState.ACCEPTED
                await _send(websocket, {
                    "type": "result",
                    "accepted": True,
                    "transcript": txn.text,
                    "score": result.similarity_score,
                    "message": "Voice authentication successful!",
                })
                await _send_status(websocket, session)
            else:
                session.attempt_count += 1
                code = result.feedback_code or "wrong_sentence"
                await _send(websocket, {
                    "type": "result",
                    "accepted": False,
                    "transcript": txn.text,
                    "score": result.similarity_score,
                    "message": _FEEDBACK.get(code, _FEEDBACK["wrong_sentence"]),
                })

                if session.exhausted():
                    session.state = SessionState.REJECTED
                    await _send_feedback(websocket, "too_many_attempts")
                    await _send_status(websocket, session)
                else:
                    session.transition(SessionState.PAUSED, reason=code)
                    await _send_status(websocket, session)

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", session_id, exc, exc_info=True)
        await _send(websocket, {"type": "error", "message": "Internal error. Please retry."})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Sub-handlers
# ------------------------------------------------------------------

async def _handle_control(
    text: str,
    ws: WebSocket,
    session: Session,
    preprocessor: AudioPreprocessor,
) -> None:
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return

    match msg.get("type"):
        case "resume":
            if session.can_resume():
                preprocessor.reset()
                if session.transition(SessionState.LISTENING):
                    await _send_status(ws, session, "Resumed. Please speak the sentence again.")
            else:
                await _send(ws, {
                    "type": "error",
                    "message": f"Cannot resume from state {session.state}",
                })
        case "ping":
            await _send(ws, {"type": "pong"})
        case "flush":
            segment = preprocessor.flush()
            await _send(ws, {"type": "flush_ack", "has_segment": segment is not None})


async def _handle_bad_segment(
    ws: WebSocket,
    session: Session,
    preprocessor: AudioPreprocessor,
    reason: Optional[str],
) -> None:
    # Background noise / clipped fragment — do NOT count as an attempt.
    # Just reset the audio buffer and stay in LISTENING so the user can speak.
    if reason in ("too_short", "low_speech_ratio"):
        logger.debug("Discarding noise segment (%s) — staying LISTENING", reason)
        preprocessor.reset()
        session.transition(SessionState.LISTENING)
        await _send_status(ws, session)
        return

    # Real audio that was too noisy — give feedback but still stay LISTENING.
    await _send_feedback(ws, reason or "high_noise")
    preprocessor.reset()
    session.transition(SessionState.LISTENING)
    await _send_status(ws, session, "Please try again in a quieter environment.")


async def _handle_unreliable_asr(
    ws: WebSocket,
    session: Session,
    preprocessor: AudioPreprocessor,
    segment_rms: float,
) -> None:
    # If the audio was very quiet it was probably background noise — discard silently.
    if segment_rms < 0.015:
        logger.debug("Unreliable ASR on quiet audio (rms=%.4f) — discarding", segment_rms)
        preprocessor.reset()
        session.transition(SessionState.LISTENING)
        await _send_status(ws, session)
        return

    # Audible speech that ASR couldn't decode — this is a real attempt.
    await _send_feedback(ws, "low_confidence")
    session.attempt_count += 1

    if session.exhausted():
        session.state = SessionState.REJECTED
        await _send_feedback(ws, "too_many_attempts")
        await _send_status(ws, session)
    else:
        session.transition(SessionState.PAUSED, reason="low_confidence")
        await _send_status(ws, session)
