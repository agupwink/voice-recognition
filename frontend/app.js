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

// ── Recorder ──────────────────────────────────────────────────────────────────
// Click once → records for up to 8 seconds → auto-stops → submits

function makeRecorder({ prefix, onDone }) {
    const btn   = $(`${prefix}-record-btn`);
    const label = $(`${prefix}-btn-label`);
    const hint  = $(`${prefix}-rec-hint`);

    let mediaRec = null, stream = null, chunks = [], timer = null, active = false;

    async function start() {
        if (active) return;
        chunks = [];
        $(`${prefix}-result`).hidden = true;

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

        const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus" : "audio/webm";
        mediaRec = new MediaRecorder(stream, { mimeType: mime });
        mediaRec.ondataavailable = e => { if (e.data?.size > 0) chunks.push(e.data); };
        mediaRec.onstop = async () => {
            btn.classList.remove("recording");
            if (label) label.textContent = "Start";
            if (hint)  hint.textContent  = "Verifying…";
            try {
                const wav = await blobToWav(new Blob(chunks, { type: mime }));
                onDone(wav);
            } catch (err) {
                console.error(err);
                if (hint) hint.textContent = "Audio error — please try again";
            }
        };
        mediaRec.start(100);

        active = true;
        btn.classList.add("recording");
        if (label) label.textContent = "Recording…";

        // Live timer display
        const startTime = Date.now();
        const timerInterval = setInterval(() => {
            const secs = ((Date.now() - startTime) / 1000).toFixed(1);
            if (hint) hint.textContent = `Recording… ${secs}s captured`;
        }, 100);

        // Auto-stop after 8 seconds
        timer = setTimeout(() => { clearInterval(timerInterval); stop(); }, 8000);
    }

    function stop() {
        if (!active) return;
        active = false;
        clearTimeout(timer);
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
        prefix: "login",
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
        prefix: "signup",
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
