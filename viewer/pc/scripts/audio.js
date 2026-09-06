/* Synthesised combat audio — no assets, no extra engine systems. */

export class CombatAudio {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.noise = null;
    this.enabled = true;
  }

  /** Browsers block audio until a gesture; call from the first click. */
  unlock() {
    if (this.ctx) {
      if (this.ctx.state === "suspended") this.ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { this.enabled = false; return; }
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(this.ctx.destination);
    const len = Math.floor(this.ctx.sampleRate * 0.5);
    this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }

  get ready() {
    return this.enabled && this.ctx && this.ctx.state === "running";
  }

  _burst({ dur, freq, q = 1, gain = 0.3, type = "bandpass", decay = 6, curve = "exp" }) {
    if (!this.ready) return;
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.8 + Math.random() * 0.4;
    const filt = this.ctx.createBiquadFilter();
    filt.type = type;
    filt.frequency.value = freq;
    filt.Q.value = q;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    if (curve === "exp") g.gain.exponentialRampToValueAtTime(0.0005, t + dur);
    else g.gain.linearRampToValueAtTime(0, t + dur);
    src.connect(filt).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + dur + 0.02);
  }

  _tone({ freq, to, dur, gain = 0.2, type = "sine" }) {
    if (!this.ready) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    if (to) osc.frequency.exponentialRampToValueAtTime(Math.max(20, to), t + dur);
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.0005, t + dur);
    osc.connect(g).connect(this.master);
    osc.start(t);
    osc.stop(t + dur + 0.02);
  }

  /** `near` is 0 (in your face) .. 1 (far away) and drives loudness/darkness. */
  playerShot(near = 0) {
    const k = 1 - Math.min(1, near);
    this._burst({ dur: 0.09, freq: 1500 + 900 * k, q: 0.7, gain: 0.16 + 0.2 * k });
    this._tone({ freq: 150, to: 48, dur: 0.1, gain: 0.14 * k + 0.04, type: "triangle" });
  }

  enemyShot(near = 0) {
    const k = 1 - Math.min(1, near);
    this._burst({ dur: 0.11, freq: 900 + 700 * k, q: 0.9, gain: 0.05 + 0.16 * k });
    this._tone({ freq: 110, to: 40, dur: 0.12, gain: 0.09 * k, type: "triangle" });
  }

  hit() {
    this._tone({ freq: 1500, to: 900, dur: 0.055, gain: 0.16, type: "square" });
  }

  headshot() {
    this._tone({ freq: 2300, to: 1200, dur: 0.08, gain: 0.18, type: "square" });
  }

  kill() {
    this._tone({ freq: 620, to: 180, dur: 0.24, gain: 0.16, type: "sawtooth" });
  }

  impact() {
    this._burst({ dur: 0.05, freq: 380, q: 1.6, gain: 0.1 });
  }

  hurt() {
    this._burst({ dur: 0.16, freq: 220, q: 0.8, gain: 0.22, type: "lowpass" });
    this._tone({ freq: 90, to: 44, dur: 0.2, gain: 0.16, type: "sine" });
  }

  reloadStart() {
    this._burst({ dur: 0.04, freq: 2200, q: 3, gain: 0.1 });
    setTimeout(() => this.ready && this._burst({ dur: 0.05, freq: 1100, q: 3, gain: 0.11 }), 150);
  }

  reloadEnd() {
    this._burst({ dur: 0.05, freq: 700, q: 4, gain: 0.14 });
  }

  empty() {
    this._burst({ dur: 0.03, freq: 3000, q: 6, gain: 0.07 });
  }
}
