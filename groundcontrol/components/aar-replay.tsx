"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { fmtClock, type Basemap, type Run } from "@/lib/mission";
import { Icon } from "./studio-icons";

/** Frame of ``frames`` (rows start with t) at time ``t``: the last one not after it. */
function frameAt(frames: number[][], t: number) {
  if (!frames.length) return null;
  let lo = 0, hi = frames.length - 1;
  if (t <= frames[0][0]) return frames[0];
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  return frames[lo];
}

const EVENT_STYLE: Record<string, { color: string; label: string }> = {
  waypoint: { color: "#3fb950", label: "Waypoint" }, complete: { color: "#3fb950", label: "Objective" },
  spotted: { color: "#f0883e", label: "Spotted" }, hit_taken: { color: "#f85149", label: "Hit" },
  casualty: { color: "#f85149", label: "Casualty" }, bot_down: { color: "#a371f7", label: "Enemy down" },
  phase_line: { color: "#8b949e", label: "Phase line" }, off_route: { color: "#d29922", label: "Off route" },
};

/**
 * After-action review: a top-down replay of a recorded rehearsal over the scanned surface
 * (hillshade), with the planned route, the player's track, every enemy with its line of
 * sight when it could see the player, and a timeline with the exposure strip and events.
 */
export function AarReplay({ run, basemap, onClose, onPdf }: { run: Run; basemap: Basemap | null; onClose: () => void; onPdf?: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const strip = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const t0 = run.player[0][0], t1 = run.player.at(-1)![0];
  const [t, setT] = useState(t0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);

  useEffect(() => {
    if (!basemap) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = `data:image/png;base64,${basemap.png_base64}`;
  }, [basemap]);

  // Frame the action: everything the player, the enemies and the route touched, plus margin.
  const view = useMemo(() => {
    const xs: number[] = [], zs: number[] = [];
    for (const p of run.player) { xs.push(p[1]); zs.push(p[3]); }
    for (const b of run.bots) for (const f of b.frames) { xs.push(f[1]); zs.push(f[3]); }
    for (const [x, z] of run.route) { xs.push(x); zs.push(z); }
    const pad = 12;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, z0 = Math.min(...zs) - pad, z1 = Math.max(...zs) + pad;
    const size = Math.max(x1 - x0, z1 - z0);
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    return { x0: cx - size / 2, z0: cz - size / 2, size };
  }, [run]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now(), raf = 0;
    const tick = (now: number) => {
      setT((prev) => {
        const next = prev + ((now - last) / 1000) * speed;
        if (next >= t1) { setPlaying(false); return t1; }
        return next;
      });
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, t1]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    const px = (x: number) => ((x - view.x0) / view.size) * W;
    const pz = (z: number) => ((z - view.z0) / view.size) * H;
    ctx.fillStyle = "#1b2027";
    ctx.fillRect(0, 0, W, H);
    if (image && basemap) {
      const b = basemap.bounds;
      const sx = ((view.x0 - b.x0) / (b.x1 - b.x0)) * image.width, sy = ((view.z0 - b.z0) / (b.z1 - b.z0)) * image.height;
      const sw = (view.size / (b.x1 - b.x0)) * image.width, sh = (view.size / (b.z1 - b.z0)) * image.height;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, sx, sy, sw, sh, 0, 0, W, H);
    }
    // Scale bar.
    const metres = [5, 10, 20, 50, 100, 200].find((m) => (m / view.size) * W > 70) ?? 200;
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(10, H - 28, (metres / view.size) * W + 16, 20);
    ctx.fillStyle = "#fff"; ctx.fillRect(18, H - 16, (metres / view.size) * W, 3);
    ctx.font = "10px system-ui"; ctx.fillText(`${metres} m`, 18, H - 19);
    // Planned route.
    if (run.route.length > 1) {
      ctx.setLineDash([7, 5]); ctx.strokeStyle = "rgba(255,160,40,.95)"; ctx.lineWidth = 2.5;
      ctx.beginPath(); run.route.forEach(([x, z], i) => (i ? ctx.lineTo(px(x), pz(z)) : ctx.moveTo(px(x), pz(z)))); ctx.stroke();
      ctx.setLineDash([]);
      run.route.forEach(([x, z]) => { ctx.fillStyle = "#ffa028"; ctx.beginPath(); ctx.arc(px(x), pz(z), 4, 0, Math.PI * 2); ctx.fill(); });
    }
    // Player track so far, coloured where the player was seen.
    const seenAt = (time: number) => run.bots.some((b) => { const f = frameAt(b.frames, time); return !!f && f[5] > 0.5 && f[6] > 0.5; });
    ctx.lineWidth = 3;
    for (let i = 1; i < run.player.length && run.player[i][0] <= t; i++) {
      const a = run.player[i - 1], b = run.player[i];
      if (Math.hypot(b[1] - a[1], b[3] - a[3]) > 8) continue;      // a respawn, not a walk
      ctx.strokeStyle = seenAt(b[0]) ? "#ff5a4f" : "#4fd1ff";
      ctx.beginPath(); ctx.moveTo(px(a[1]), pz(a[3])); ctx.lineTo(px(b[1]), pz(b[3])); ctx.stroke();
    }
    const me = frameAt(run.player, t)!;
    // Enemies: position, facing, line of sight to the player while they could see.
    for (const bot of run.bots) {
      const f = frameAt(bot.frames, t);
      if (!f) continue;
      const x = px(f[1]), z = pz(f[3]);
      const alive = f[6] > 0.5;
      if (alive && f[5] > 0.5) {
        ctx.strokeStyle = "rgba(255,70,60,.9)"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, z); ctx.lineTo(px(me[1]), pz(me[3])); ctx.stroke();
      }
      const yaw = f[4];
      const dx = -Math.sin(yaw), dz = -Math.cos(yaw);
      ctx.save(); ctx.translate(x, z);
      ctx.fillStyle = alive ? "#e5484d" : "#6e7681";
      ctx.beginPath(); ctx.moveTo(dx * 10, dz * 10); ctx.lineTo(-dz * 6 - dx * 5, dx * 6 - dz * 5); ctx.lineTo(dz * 6 - dx * 5, -dx * 6 - dz * 5); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.fillText(bot.label || `#${bot.id}`, x + 8, z - 6);
    }
    // Events so far.
    for (const e of run.events) {
      if (!e.at || e.t > t) continue;
      const st = EVENT_STYLE[e.type];
      if (!st) continue;
      ctx.fillStyle = st.color; ctx.beginPath(); ctx.arc(px(e.at[0]), pz(e.at[2]), 4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.save(); ctx.translate(px(me[1]), pz(me[3]));
    const dx = -Math.sin(me[4]), dz = -Math.cos(me[4]);
    ctx.fillStyle = "#4fd1ff"; ctx.strokeStyle = "#062a36"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(0, 0, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(dx * 16, dz * 16); ctx.strokeStyle = "#4fd1ff"; ctx.stroke();
    ctx.restore();
  }, [run, view, image, basemap, t]);

  // Exposure strip: how many enemies could see the player, over the whole run.
  useEffect(() => {
    const c = strip.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const n = Math.max(1, run.bots.length);
    for (let i = 0; i < W; i++) {
      const time = t0 + ((t1 - t0) * i) / W;
      const k = run.bots.reduce((s, b) => { const f = frameAt(b.frames, time); return s + (f && f[5] > 0.5 && f[6] > 0.5 ? 1 : 0); }, 0);
      if (!k) continue;
      ctx.fillStyle = k > 1 ? "#f85149" : "#f0883e";
      const h = Math.max(4, (k / n) * H);
      ctx.fillRect(i, H - h, 1, h);
    }
    for (const e of run.events) {
      const st = EVENT_STYLE[e.type];
      if (!st) continue;
      ctx.fillStyle = st.color;
      ctx.fillRect(((e.t - t0) / Math.max(t1 - t0, 1e-6)) * W - 1, 0, 2, 6);
    }
  }, [run, t0, t1]);

  const s = run.summary;
  return <div className="aar" role="dialog" aria-label="After-action review">
    <div className="aar-head">
      <div><strong>After-action review</strong><small>{new Date(run.created_at).toLocaleString()} · {run.conditions.light ?? "day"}{run.conditions.filter === "nvg" ? " + NVG" : ""}{run.conditions.fog_m ? ` · visibility ${run.conditions.fog_m} m` : ""}</small></div>
      <span className="aar-actions">{onPdf && <button className="button secondary small" onClick={onPdf}><Icon name="download" size={13} />AAR PDF</button>}
        <button className="icon-button" aria-label="Close review" onClick={onClose}><Icon name="close" size={15} /></button></span>
    </div>
    <div className="aar-body">
      <canvas ref={canvas} width={640} height={640} className="aar-map" aria-label="Top-down replay" />
      <div className="aar-side">
        <div className="aar-stats">
          <div><b>{fmtClock(s.duration_s)}</b><small>duration</small></div>
          <div><b>{s.distance_m.toFixed(0)} m</b><small>walked</small></div>
          <div className={s.exposure_s ? "bad" : "good"}><b>{s.exposure_s.toFixed(1)} s</b><small>seen ({s.exposed_pct}%)</small></div>
          <div><b>{s.waypoints_reached}</b><small>waypoints{s.completed ? " · objective" : ""}</small></div>
          <div className={s.hits_taken ? "bad" : ""}><b>{s.hits_taken}</b><small>hits taken{s.casualty ? ` · ${s.casualty} down` : ""}</small></div>
          <div><b>{s.neutralised}/{s.per_bot.length}</b><small>enemies neutralised</small></div>
        </div>
        <table className="plan-table aar-bots"><thead><tr><th>Enemy</th><th>Saw you</th><th>First</th></tr></thead>
          <tbody>{s.per_bot.map((b) => <tr key={b.id}><td>{b.label || `#${b.id}`}{b.neutralised ? " ✕" : ""}</td><td>{b.exposure_s.toFixed(1)} s</td><td>{b.first_seen_s === null ? "—" : fmtClock(b.first_seen_s)}</td></tr>)}</tbody></table>
        <ol className="aar-events">{run.events.filter((e) => EVENT_STYLE[e.type]).map((e, i) => <li key={i}><button onClick={() => { setPlaying(false); setT(e.t); }}>
          <i style={{ background: EVENT_STYLE[e.type].color }} /><span>{fmtClock(e.t - t0)}</span>{e.text}</button></li>)}</ol>
      </div>
    </div>
    <div className="aar-timeline">
      <button className="icon-button" aria-label={playing ? "Pause" : "Play"} onClick={() => { if (t >= t1) setT(t0); setPlaying(!playing); }}><Icon name={playing ? "stop" : "play"} size={15} /></button>
      <div className="aar-track">
        <canvas ref={strip} width={900} height={22} aria-hidden />
        <input type="range" min={t0} max={t1} step={0.1} value={t} aria-label="Replay time" onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
      </div>
      <span className="aar-clock">{fmtClock(t - t0)} / {fmtClock(t1 - t0)}</span>
      <select aria-label="Replay speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>{[1, 4, 16].map((v) => <option key={v} value={v}>{v}×</option>)}</select>
    </div>
    <p className="aar-note">{run.note} Red track: seen by at least one enemy. Red lines: who could see you at that moment.</p>
  </div>;
}
