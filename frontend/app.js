"use strict";

const API = window.location.origin;

// ── WAV encoder ───────────────────────────────────────────────────────────────

async function blobToWav(blob) {
    const arrayBuf  = await blob.arrayBuffer();
    const decodeCtx = new AudioContext();
    let audioBuf;
    try   { audioBuf = await decodeCtx.decodeAudioData(arrayBuf); }
    finally { decodeCtx.close(); }

    const TARGET  = 16_000;
    const frames  = Math.ceil(audioBuf.duration * TARGET);
    const offCtx  = new OfflineAudioContext(1, frames, TARGET);
    const src     = offCtx.createBufferSource();
    src.buffer    = audioBuf;
    src.connect(offCtx.destination);
    src.start(0);
    const rendered = await offCtx.startRendering();
    const floats   = rendered.getChannelData(0);

    const wav = new ArrayBuffer(44 + floats.length * 2);
    const dv  = new DataView(wav);
    const ws  = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    ws(0,"RIFF"); dv.setUint32(4, 36 + floats.length * 2, true);
    ws(8,"WAVE"); ws(12,"fmt ");
    dv.setUint32(16,16,true); dv.setUint16(20,1,true); dv.setUint16(22,1,true);
    dv.setUint32(24,TARGET,true); dv.setUint32(28,TARGET*2,true);
    dv.setUint16(32,2,true); dv.setUint16(34,16,true);
    ws(36,"data"); dv.setUint32(40, floats.length * 2, true);
    const i16 = new Int16Array(wav, 44);
    for (let i = 0; i < floats.length; i++)
        i16[i] = Math.round(Math.max(-1, Math.min(1, floats[i])) * 32767);
    return wav;
}

// ── DOM & tabs ────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.add("hidden"));
    t.classList.add("active");
    $(`panel-${t.dataset.tab}`).classList.remove("hidden");
}));

function showResult(prefix, data) {
    const el = $(`${prefix}-result`);
    if (!el) return;
    const vBar  = data.voice_score  != null ? sbar("Voice match", data.voice_score,  0.75)      : "";
    const pBar  = data.phrase_score != null ? sbar("Passphrase",  data.phrase_score, 0.55, true) : "";
    const heard = data.transcript   ? `<div class="result-heard">Heard: <em>"${data.transcript}"</em></div>` : "";
    el.className = `result-card ${data.success ? "success" : "failure"}`;
    el.innerHTML = `<div class="result-title">${data.success ? "✓" : "✗"}  ${data.message}</div>${vBar}${pBar}${heard}`;
    el.hidden = false;
}

function sbar(label, score, thr, dim = false) {
    const pct = Math.round(score * 100);
    const ok  = score >= thr;
    const col = ok ? (dim ? "#4ade80" : "#22c55e") : (dim ? "#94a3b8" : "#ef4444");
    return `<div class="score-row"${dim ? ' style="opacity:.55"' : ""}>
      <span class="score-label-txt">${ok ? "✓" : "✗"} ${label}</span>
      <div class="score-track-inline"><div class="score-fill-inline" style="width:${pct}%;background:${col}"></div></div>
      <span class="score-pct">${pct}%</span></div>`;
}

// ── Smart Recorder ────────────────────────────────────────────────────────────
//
// Progress bar advances ONLY during confirmed speech.
// Silence / noise / phrase mismatch FREEZE progress and show why.
// MediaRecorder captures everything (backend VAD filters it).
// Falls back to time-based progress if AnalyserNode is unavailable.

const REC = {
    TARGET_SPEECH_MS:   4500,   // valid speech needed to complete recording
    MAX_SESSION_MS:    14_000,   // hard cap — stops even without enough speech
    SPEECH_RMS:         0.012,  // RMS above this = active voice
    NOISE_RMS:          0.004,  // RMS above this (below SPEECH) = background noise
    ONSET_FRAMES:           4,  // consecutive speech frames before entering SPEECH state
    PAUSE_SILENCE_MS:     700,  // silence duration before showing "no speech" message
    PAUSE_NOISE_MS:       900,  // noise duration before showing noise warning
    PHRASE_MIN_OVERLAP:   0.25, // fraction of phrase words needed for a match
};

// State → { message, css class }
const REC_STATES = {
    listening:        { msg: "Listening… speak your passphrase",                cls: "status-idle"       },
    speech:           { msg: "Speech detected — keep speaking",                 cls: "status-good"       },
    paused_silence:   { msg: "No speech detected — please start speaking",      cls: "status-warn"       },
    paused_noise:     { msg: "Too much background noise — move somewhere quieter", cls: "status-warn"    },
    paused_phrase:    { msg: "Please say the displayed passphrase",             cls: "status-warn"       },
    processing:       { msg: "Processing voice authentication…",                cls: "status-processing" },
};

// Lenient phrase check used for the live-feedback warning only.
// Returns true if transcript overlaps the phrase enough to count as "correct".
function _phraseMatch(transcript, phrase) {
    if (!transcript || !phrase) return true;   // no data → assume OK
    const clean = s => s.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/).filter(Boolean);
    const tw = clean(transcript), pw = clean(phrase);
    if (!tw.length) return true;
    const ps   = new Set(pw);
    const hits = tw.filter(w => ps.has(w)).length;
    return hits / pw.length >= REC.PHRASE_MIN_OVERLAP;
}

function makeRecorder({ prefix, getPhrase = null, onDone }) {
    const btn       = $(`${prefix}-record-btn`);
    const label     = $(`${prefix}-btn-label`);
    const hint      = $(`${prefix}-rec-hint`);
    const statusEl  = $(`${prefix}-rec-status`);
    const levelFill = $(`${prefix}-level-fill`);
    const progWrap  = $(`${prefix}-prog-wrap`);
    const progFill  = $(`${prefix}-prog-fill`);
    const progLabel = $(`${prefix}-prog-label`);

    let mediaRec = null, stream = null, chunks = [];
    let timer = null, rafId = null, active = false;
    let audioCtx = null, analyser = null;

    // State machine
    let recState    = "listening";
    let validMs     = 0;        // accumulated valid-speech milliseconds
    let silenceMs   = 0;        // consecutive silence duration
    let noiseMs     = 0;        // consecutive noise duration
    let onsetCount  = 0;        // consecutive speech frames (onset guard)
    let smoothRMS   = 0;        // exponentially-smoothed RMS
    let lastTs      = 0;        // previous RAF timestamp

    // SpeechRecognition (phrase validation — informational only)
    let srText      = "";
    let srMatch     = true;     // optimistic: assume match until proven otherwise
    let recog       = null;

    const rmsArr    = new Float32Array(256);

    // ── UI helpers ────────────────────────────────────────────────────────────

    function setStatus(stateKey) {
        const s = REC_STATES[stateKey] || REC_STATES.listening;
        if (statusEl) { statusEl.textContent = s.msg; statusEl.className = `rec-status ${s.cls}`; }
    }

    function resetUI() {
        if (levelFill) levelFill.style.width = "0%";
        if (progFill)  progFill.style.width  = "0%";
        if (progLabel) progLabel.textContent = `0 / ${(REC.TARGET_SPEECH_MS / 1000).toFixed(1)}s`;
        if (progWrap)  progWrap.hidden = true;
        if (statusEl)  { statusEl.textContent = ""; statusEl.className = "rec-status"; }
    }

    function getRMS() {
        if (!analyser) return -1;   // -1 signals "analyser unavailable"
        analyser.getFloatTimeDomainData(rmsArr);
        let s = 0;
        for (let i = 0; i < rmsArr.length; i++) s += rmsArr[i] * rmsArr[i];
        return Math.sqrt(s / rmsArr.length);
    }

    // ── RAF loop ──────────────────────────────────────────────────────────────

    function tick(ts) {
        if (!active) return;

        const dt  = lastTs ? Math.min(ts - lastTs, 80) : 0;   // cap delta to 80 ms
        lastTs = ts;

        const rawRMS = getRMS();

        // ── Fallback: if analyser unavailable, use time-based progress ────────
        if (rawRMS < 0) {
            const elapsed = Date.now() - (ts - dt);
            const pct = Math.min(100, (elapsed / REC.MAX_SESSION_MS) * 100);
            if (progFill)  progFill.style.width  = `${pct}%`;
            if (progLabel) progLabel.textContent = `${(elapsed / 1000).toFixed(1)}s`;
            setStatus("listening");
            rafId = requestAnimationFrame(tick);
            return;
        }

        // Smooth the RMS to avoid jitter
        smoothRMS = 0.25 * rawRMS + 0.75 * smoothRMS;

        // ── Level meter ───────────────────────────────────────────────────────
        if (levelFill)
            levelFill.style.width = `${Math.min(100, (smoothRMS / 0.07) * 100)}%`;

        // ── State machine ─────────────────────────────────────────────────────
        const phrase = getPhrase ? getPhrase() : "";

        if (smoothRMS >= REC.SPEECH_RMS) {
            silenceMs = 0; noiseMs = 0;
            onsetCount = Math.min(onsetCount + 1, REC.ONSET_FRAMES + 1);

            if (onsetCount >= REC.ONSET_FRAMES) {
                // Phrase mismatch check — only warn if SR has enough data AND clearly wrong
                if (srText.length > 4 && !srMatch) {
                    recState = "paused_phrase";
                    // Don't advance validMs while phrase is wrong
                } else {
                    recState = "speech";
                    validMs += dt;                  // ← ONLY valid speech advances progress
                }
            } else {
                recState = "listening";             // onset guard: ignore brief pops
            }

        } else if (smoothRMS >= REC.NOISE_RMS) {
            onsetCount = 0;
            noiseMs   += dt;
            silenceMs  = 0;

            if (recState === "speech" || recState === "paused_phrase") {
                // Brief dip — keep previous state briefly before switching
                if (noiseMs > REC.PAUSE_NOISE_MS) recState = "paused_noise";
            } else if (recState !== "paused_silence") {
                if (noiseMs > REC.PAUSE_NOISE_MS) recState = "paused_noise";
            }

        } else {
            // Silence
            onsetCount = 0; noiseMs = 0;
            silenceMs += dt;

            if (recState === "speech" || recState === "paused_noise" || recState === "paused_phrase") {
                if (silenceMs > REC.PAUSE_SILENCE_MS) recState = "paused_silence";
            } else if (recState !== "paused_silence") {
                if (silenceMs > REC.PAUSE_SILENCE_MS) recState = "paused_silence";
                else recState = "listening";
            }
        }

        // ── Update status message ─────────────────────────────────────────────
        setStatus(recState);

        // ── Progress bar (valid speech only) ─────────────────────────────────
        const pct = Math.min(100, (validMs / REC.TARGET_SPEECH_MS) * 100);
        if (progFill)  progFill.style.width  = `${pct}%`;
        if (progLabel) progLabel.textContent =
            `${(validMs / 1000).toFixed(1)} / ${(REC.TARGET_SPEECH_MS / 1000).toFixed(1)}s`;

        // ── Auto-complete: enough valid speech collected ───────────────────────
        if (validMs >= REC.TARGET_SPEECH_MS) { stop(); return; }

        rafId = requestAnimationFrame(tick);
    }

    // ── Start ─────────────────────────────────────────────────────────────────

    async function start() {
        if (active) return;
        chunks = []; validMs = 0; silenceMs = 0; noiseMs = 0;
        onsetCount = 0; smoothRMS = 0; lastTs = 0;
        recState = "listening"; srText = ""; srMatch = true;
        $(`${prefix}-result`).hidden = true;
        resetUI();

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, echoCancellation: true,
                         noiseSuppression: true, autoGainControl: true },
                video: false,
            });
        } catch {
            alert("Microphone access denied — please allow it in browser settings.");
            return;
        }

        // AnalyserNode — visualization only, never affects capture
        try {
            audioCtx = new AudioContext();
            if (audioCtx.state === "suspended") await audioCtx.resume();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.2;
            audioCtx.createMediaStreamSource(stream).connect(analyser);
        } catch (_) { analyser = null; }   // graceful fallback to time-based progress

        // SpeechRecognition — phrase matching feedback only, never blocks capture
        const SRC = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SRC && getPhrase) {
            try {
                recog = new SRC();
                recog.continuous = true; recog.interimResults = true; recog.lang = "en-US";
                recog.onresult = ev => {
                    let tx = "";
                    for (let i = ev.resultIndex; i < ev.results.length; i++)
                        tx += ev.results[i][0].transcript + " ";
                    srText  = tx.trim();
                    srMatch = _phraseMatch(srText, getPhrase());
                };
                recog.onend   = () => { if (active) try { recog.start(); } catch(_){} };
                recog.onerror = () => {};
                recog.start();
            } catch(_) { recog = null; }
        }

        // MediaRecorder — captures everything for server-side processing
        const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus" : "audio/webm";
        mediaRec = new MediaRecorder(stream, { mimeType: mime });
        mediaRec.ondataavailable = e => { if (e.data?.size > 0) chunks.push(e.data); };
        mediaRec.onstop = async () => {
            cancelAnimationFrame(rafId);
            try { recog?.stop(); } catch(_) {}
            audioCtx?.close().catch(() => {});
            audioCtx = null; analyser = null; recog = null;

            btn.classList.remove("recording");
            if (label) label.textContent = "Start";
            resetUI();
            if (hint) hint.textContent = "Verifying…";
            setStatus("processing");

            try {
                const wav = await blobToWav(new Blob(chunks, { type: mime }));
                onDone(wav);
            } catch (err) {
                console.error(err);
                if (statusEl) { statusEl.textContent = ""; statusEl.className = "rec-status"; }
                if (hint) hint.textContent = "Audio error — please try again";
            }
        };
        mediaRec.start(100);

        active = true;
        btn.classList.add("recording");
        if (label) label.textContent = "Recording…";
        if (progWrap) progWrap.hidden = false;
        setStatus("listening");

        // Hard-cap fallback: stop after MAX_SESSION_MS regardless
        timer = setTimeout(stop, REC.MAX_SESSION_MS);

        rafId = requestAnimationFrame(tick);
    }

    // ── Stop ──────────────────────────────────────────────────────────────────

    function stop() {
        if (!active) return;
        active = false;
        clearTimeout(timer);
        cancelAnimationFrame(rafId);
        stream?.getTracks().forEach(t => t.stop());
        if (mediaRec?.state !== "inactive") mediaRec.stop();
    }

    btn.addEventListener("click", () => active ? stop() : start());
}

// ── Login ─────────────────────────────────────────────────────────────────────

async function loadPassphraseHint(username) {
    if (!username) return;
    try {
        const res = await fetch(`${API}/auth/passphrase/${encodeURIComponent(username)}`);
        if (!res.ok) return;
        const { passphrase } = await res.json();
        $("login-hint-text").textContent = passphrase;
        $("login-hint-box").hidden       = false;
        $("login-record-btn").disabled   = false;
        const hint = $("login-rec-hint");
        if (hint) hint.textContent = "Press the button and speak your passphrase";
    } catch (_) {}
}

(function setupLogin() {
    const inp = $("login-username");
    let debounce = null;
    inp.addEventListener("input", () => {
        clearTimeout(debounce);
        const v = inp.value.trim();
        if (!v) return;
        debounce = setTimeout(() => loadPassphraseHint(v), 400);
    });
    inp.addEventListener("keydown", e => {
        if (e.key === "Enter") { clearTimeout(debounce); loadPassphraseHint(inp.value.trim()); }
    });

    makeRecorder({
        prefix:    "login",
        getPhrase: () => $("login-hint-text")?.textContent?.trim() || "",
        async onDone(wav) {
            const form = new FormData();
            form.append("username", $("login-username").value.trim());
            form.append("audio", new Blob([wav], { type: "audio/wav" }), "voice.wav");
            try {
                const res  = await fetch(`${API}/auth/login`, { method: "POST", body: form });
                const data = await res.json();
                const hint = $("login-rec-hint");
                if (res.ok) {
                    if (hint) hint.textContent = data.success ? "✓ Verified!" : "✗ Not verified";
                    showResult("login", data);
                } else {
                    if (hint) hint.textContent = data.detail || "Login failed";
                    showResult("login", { success: false, message: data.detail || "Login failed" });
                }
            } catch (e) {
                const hint = $("login-rec-hint");
                if (hint) hint.textContent = "Network error";
            }
        },
    });
})();

// ── Signup ────────────────────────────────────────────────────────────────────

(function setupSignup() {
    const nameInp    = $("signup-name");
    const userInp    = $("signup-username");
    const phraseInp  = $("signup-passphrase");
    const phraseDisp = $("signup-phrase-display");
    const phraseText = $("signup-phrase-text");

    function updateReady() {
        const ok = nameInp.value.trim() && userInp.value.trim() && phraseInp.value.trim();
        $("signup-record-btn").disabled = !ok;
        const hint = $("signup-rec-hint");
        if (hint && !ok) hint.textContent = "Fill in the fields above";
        if (hint && ok)  hint.textContent = "Press the button and speak your passphrase";
    }

    phraseInp.addEventListener("input", () => {
        const p = phraseInp.value.trim();
        phraseText.textContent = p;
        phraseDisp.hidden      = !p;
        updateReady();
    });
    [nameInp, userInp].forEach(el => el.addEventListener("input", updateReady));

    makeRecorder({
        prefix:    "signup",
        getPhrase: () => phraseInp.value.trim(),
        async onDone(wav) {
            const form = new FormData();
            form.append("username",     userInp.value.trim());
            form.append("display_name", nameInp.value.trim());
            form.append("passphrase",   phraseInp.value.trim());
            form.append("audio", new Blob([wav], { type: "audio/wav" }), "voice.wav");
            try {
                const res  = await fetch(`${API}/auth/signup`, { method: "POST", body: form });
                const data = await res.json();
                const hint = $("signup-rec-hint");
                if (res.ok && data.success) {
                    if (hint) hint.textContent = `✓ Enrolled — welcome, ${data.display_name}!`;
                    showResult("signup", { success: true, message: `Account created for ${data.display_name}`, transcript: data.transcript });
                    setTimeout(() => {
                        $("login-username").value = userInp.value.trim();
                        document.querySelector('[data-tab="login"]').click();
                        loadPassphraseHint(userInp.value.trim());
                    }, 2000);
                } else {
                    if (hint) hint.textContent = data.detail || data.message || "Sign-up failed";
                    showResult("signup", { success: false, message: data.detail || data.message || "Sign-up failed" });
                }
            } catch (e) {
                const hint = $("signup-rec-hint");
                if (hint) hint.textContent = "Network error";
            }
        },
    });
})();
