"""
Voice authentication — signup and login.

Verification logic
──────────────────
PRIMARY  : Speaker verification (Resemblyzer cosine similarity ≥ 0.75).
           This alone decides PASS / FAIL.

SECONDARY: Passphrase transcription (Whisper).
           Shown in the response for transparency / logging.
           A low passphrase score does NOT block login.
           It only raises a soft warning when well below threshold.

Audio format
────────────
16-bit PCM WAV at 16 kHz mono, encoded in the browser via OfflineAudioContext.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from asr.transcriber import WhisperTranscriber
from audio_processing.audio_quality import analyse_audio_quality
from audio_processing.vad_utils import filter_speech_audio
from config import settings
from speaker.voice_verifier import VoiceVerifier, get_verifier
from storage import database as db
from validation.phrase_validation import validate_phrase
from validation.text_validator import TextValidator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="auth")

_transcriber: Optional[WhisperTranscriber] = None
_validator:   Optional[TextValidator]      = None


def _get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def _get_validator() -> TextValidator:
    global _validator
    if _validator is None:
        _validator = TextValidator(min_match_score=0.55)   # lenient — informational only
    return _validator


def _run(fn, *args):
    return asyncio.get_event_loop().run_in_executor(_executor, fn, *args)


# ── Audio decoding ────────────────────────────────────────────────────────────

def _wav_to_float32(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples


async def _decode_audio(raw: bytes) -> np.ndarray:
    try:
        audio = _wav_to_float32(raw)
    except Exception as exc:
        logger.error("WAV decode failed: %s", exc)
        raise HTTPException(400, "Could not read audio file. Please try recording again.")

    duration_s = len(audio) / 16_000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    logger.info("Audio decoded | duration=%.2fs  rms=%.4f  bytes=%d", duration_s, rms, len(raw))

    if duration_s < 0.5:
        raise HTTPException(400, f"Recording too short ({duration_s:.1f}s). Please speak your full passphrase.")
    if rms < 0.002:
        raise HTTPException(400, "Audio too quiet — check microphone volume and try again.")

    return audio


# ── Signup ────────────────────────────────────────────────────────────────────

@router.post("/signup")
async def signup(
    username:     str        = Form(...),
    display_name: str        = Form(...),
    passphrase:   str        = Form(...),
    audio:        UploadFile = File(...),
):
    username     = username.strip().lower()
    display_name = display_name.strip()
    passphrase   = passphrase.strip()

    if not username or not passphrase:
        raise HTTPException(400, "Username and passphrase are required.")
    if db.exists(username):
        raise HTTPException(409, "Username already taken. Choose another.")

    raw   = await audio.read()
    audio_arr = await _decode_audio(raw)

    # Quality check (non-blocking — only warn, don't reject on quality failure)
    if settings.ENABLE_NOISE_DETECTION:
        quality = analyse_audio_quality(audio_arr, 16_000,
            min_rms=settings.MIN_RMS_ENERGY,
            snr_threshold_db=settings.SNR_THRESHOLD_DB)
        logger.info("Signup quality: rms=%.4f snr=%.1fdB issues=%s",
                    quality.rms_energy, quality.snr_db, quality.issues)
        if "too_quiet" in quality.issues:
            raise HTTPException(422, quality.feedback or "Audio too quiet — please speak louder.")
        if "too_short" in quality.issues:
            raise HTTPException(422, quality.feedback or "Recording too short.")

    # VAD: extract speech-only audio for better enrollment
    if settings.ENABLE_VAD:
        try:
            filtered, speech_ratio = filter_speech_audio(audio_arr, 16_000,
                threshold=settings.VAD_THRESHOLD)
            if len(filtered) >= int(16_000 * settings.MIN_SPEECH_DURATION_MS / 1000):
                audio_arr = filtered
                logger.info("VAD filtered: speech_ratio=%.2f", speech_ratio)
        except Exception as vad_err:
            logger.warning("VAD failed, using original audio: %s", vad_err)

    verifier    = get_verifier()
    transcriber = _get_transcriber()

    emb_task = _run(verifier.embed, audio_arr, 16_000)
    asr_task = _run(transcriber.transcribe, audio_arr, 16_000)
    embedding, asr_result = await asyncio.gather(emb_task, asr_task)

    if embedding is None and verifier.available:
        raise HTTPException(422, "Voice too short to enroll — please speak longer.")

    embedding = embedding if embedding is not None else np.zeros(256, dtype=np.float32)
    voice_transcript = (
        asr_result.text
        if (asr_result.is_reliable and transcriber.engine == "whisper")
        else passphrase
    )

    db.create_user(
        username=username,
        display_name=display_name,
        passphrase_text=passphrase,
        voice_transcript=voice_transcript,
        voice_embedding=embedding,
        engine=transcriber.engine,
    )

    logger.info("Enrolled %s (%s) | engine=%s transcript='%s'",
                username, display_name, transcriber.engine, voice_transcript[:60])

    return {
        "success": True,
        "username": username,
        "display_name": display_name,
        "engine": transcriber.engine,
        "transcript": asr_result.text,
        "voice_enrolled": verifier.available,
        "duration_s": round(len(audio_arr) / 16_000, 1),
    }


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(
    username: str        = Form(...),
    audio:    UploadFile = File(...),
):
    username = username.strip().lower()
    user = db.get_user(username)
    if not user:
        raise HTTPException(404, "No account found with that username.")

    raw       = await audio.read()
    audio_arr = await _decode_audio(raw)

    # Quality check (non-blocking — only warn, don't reject on quality failure)
    if settings.ENABLE_NOISE_DETECTION:
        quality = analyse_audio_quality(audio_arr, 16_000,
            min_rms=settings.MIN_RMS_ENERGY,
            snr_threshold_db=settings.SNR_THRESHOLD_DB)
        logger.info("Login quality: rms=%.4f snr=%.1fdB issues=%s",
                    quality.rms_energy, quality.snr_db, quality.issues)
        if "too_quiet" in quality.issues:
            raise HTTPException(422, quality.feedback or "Audio too quiet — please speak louder.")
        if "too_short" in quality.issues:
            raise HTTPException(422, quality.feedback or "Recording too short.")

    # VAD: extract speech-only audio for better enrollment
    if settings.ENABLE_VAD:
        try:
            filtered, speech_ratio = filter_speech_audio(audio_arr, 16_000,
                threshold=settings.VAD_THRESHOLD)
            if len(filtered) >= int(16_000 * settings.MIN_SPEECH_DURATION_MS / 1000):
                audio_arr = filtered
                logger.info("VAD filtered: speech_ratio=%.2f", speech_ratio)
        except Exception as vad_err:
            logger.warning("VAD failed, using original audio: %s", vad_err)

    verifier    = get_verifier()
    transcriber = _get_transcriber()
    validator   = _get_validator()

    # Run speaker verification and ASR in parallel
    voice_task = _run(verifier.verify, user["voice_embedding"], audio_arr, 16_000)
    asr_task   = _run(transcriber.transcribe, audio_arr, 16_000)
    (voice_ok, voice_score), asr_result = await asyncio.gather(voice_task, asr_task)

    # Passphrase check — informational only, does NOT affect pass/fail
    if asr_result.is_reliable and transcriber.engine == "whisper":
        phrase_val   = validator.validate(asr_result.text, user["voice_transcript"])
        phrase_score = phrase_val.similarity_score
    else:
        phrase_score = None

    # Phrase validation (optional layer before embedding comparison)
    if (settings.ENABLE_PHRASE_VALIDATION
            and asr_result.is_reliable
            and transcriber.engine == "whisper"):
        phrase_passed, phrase_score, phrase_msg = validate_phrase(
            asr_result.text,
            user["passphrase_text"],
            threshold=settings.PHRASE_SIMILARITY_THRESHOLD,
        )
        logger.info("Phrase validation: score=%.3f passed=%s", phrase_score, phrase_passed)
        if not phrase_passed:
            return {
                "success": False,
                "message": phrase_msg,
                "voice_score": 0.0,
                "phrase_score": round(phrase_score, 3),
                "transcript": asr_result.text,
            }

    transcript = asr_result.text if asr_result.is_reliable else "(unclear)"

    logger.info(
        "Login %s | voice_ok=%s score=%.3f phrase_score=%s transcript='%s'",
        username, voice_ok, voice_score,
        f"{phrase_score:.2f}" if phrase_score is not None else "n/a",
        transcript[:60],
    )

    # ── Decision: voice score is the ONLY factor ──────────────────────────────
    if voice_ok:
        db.touch_login(username)
        return {
            "success":      True,
            "display_name": user["display_name"],
            "message":      f"Welcome back, {user['display_name']}!",
            "voice_score":  round(voice_score, 3),
            "phrase_score": round(phrase_score, 3) if phrase_score is not None else None,
            "transcript":   transcript,
        }

    # Failed — give specific reason
    if voice_score < 0.60:
        msg = "Voice not recognised. Are you the account owner?"
    else:
        msg = "Voice similarity too low. Try speaking more clearly."

    return {
        "success":      False,
        "message":      msg,
        "voice_score":  round(voice_score, 3),
        "phrase_score": round(phrase_score, 3) if phrase_score is not None else None,
        "transcript":   transcript,
    }


# ── Management endpoints ──────────────────────────────────────────────────────

@router.get("/users")
async def list_users():
    return {"users": db.list_users()}


@router.get("/passphrase/{username}")
async def get_passphrase(username: str):
    user = db.get_user(username.strip().lower())
    if not user:
        raise HTTPException(404, "User not found.")
    return {"passphrase": user["passphrase_text"]}


@router.delete("/users/{username}")
async def delete_user(username: str):
    deleted = db.delete_user(username.strip().lower())
    if not deleted:
        raise HTTPException(404, "User not found.")
    return {"success": True, "message": f"User '{username}' deleted."}


@router.get("/status")
async def system_status():
    t = _get_transcriber()
    v = get_verifier()
    return {
        "asr_engine":           t.engine,
        "speaker_verification": v.available,
        "whisper_model":        "base",
        "primary_factor":       "voice biometric (Resemblyzer)",
        "secondary_factor":     "passphrase transcription (informational)",
        "registered_users":     db.get_user_count(),
        "database":             str(db._DB_PATH),
    }
