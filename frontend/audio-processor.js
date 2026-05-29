/**
 * AudioWorkletProcessor — runs on the dedicated audio thread.
 *
 * Responsibilities
 * ────────────────
 *  1. Receive 128-sample frames from the Web Audio graph (native sample rate).
 *  2. Downsample to 16 000 Hz using linear interpolation (anti-aliased box filter).
 *  3. Accumulate into 100 ms chunks (1 600 samples at 16 kHz).
 *  4. Transfer each chunk to the main thread as a transferable Int16Array buffer.
 *
 * Messages FROM main thread
 * ─────────────────────────
 *  { type: "start" }   — begin capturing
 *  { type: "stop"  }   — stop capturing (buffer flushed)
 */

const TARGET_SR   = 16_000;
const CHUNK_SIZE  = 1_600; // 100 ms at 16 kHz

class VoiceAuthProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);

        this._sourceSR = options.processorOptions.sourceSampleRate || sampleRate;
        this._ratio    = this._sourceSR / TARGET_SR;   // e.g. 44100/16000 = 2.75625

        this._resampled = [];   // output samples (int16 range)
        this._srcPhase  = 0;    // fractional position in input frame
        this._active    = false;

        this.port.onmessage = ({ data }) => {
            if (data.type === "start") this._active = true;
            if (data.type === "stop")  this._flush();
        };
    }

    process(inputs) {
        if (!this._active) return true;

        const channel = inputs[0]?.[0];
        if (!channel || channel.length === 0) return true;

        // Walk forward by ratio steps across the input frame using linear interpolation
        while (this._srcPhase < channel.length - 1) {
            const i0   = Math.floor(this._srcPhase);
            const frac = this._srcPhase - i0;
            const s    = channel[i0] * (1 - frac) + channel[i0 + 1] * frac;

            this._resampled.push(Math.round(Math.max(-1, Math.min(1, s)) * 32_767));
            this._srcPhase += this._ratio;
        }

        this._srcPhase -= channel.length;  // carry fractional remainder
        this._emitChunks();

        return true; // keep processor alive
    }

    _emitChunks() {
        while (this._resampled.length >= CHUNK_SIZE) {
            const chunk = this._resampled.splice(0, CHUNK_SIZE);
            const buf   = new Int16Array(chunk).buffer;
            this.port.postMessage({ type: "audio", buffer: buf }, [buf]);
        }
    }

    _flush() {
        // Emit whatever is left (may be < CHUNK_SIZE)
        if (this._resampled.length > 0) {
            const buf = new Int16Array(this._resampled).buffer;
            this.port.postMessage({ type: "audio", buffer: buf }, [buf]);
            this._resampled = [];
        }
        this._active = false;
    }
}

registerProcessor("voice-auth-processor", VoiceAuthProcessor);
