"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { fmtClock, type Basemap, type MissionAnalysis } from "@/lib/mission";
import type { PhaseLineParams, PlanState, RouteParams, SymbolParams } from "@/lib/plan";
import { Icon } from "./studio-icons";

const FRAME: Record<string, { fill: string; stroke: string }> = {
  hostile: { fill: "#ff8080", stroke: "#7a0000" }, friendly: { fill: "#80c0ff", stroke: "#003a7a" },
  unknown: { fill: "#ffff80", stroke: "#6b6b00" }, neutral: { fill: "#aaffaa", stroke: "#006b00" },
};
const ROLE_GLYPH: Record<string, string> = {
  infantry: "INF", sniper: "SNP", machine_gun: "MG", vehicle: "VEH", observation_post: "OP", objective: "OBJ",
  obstacle: "OBS", checkpoint: "CP", rally_point: "RP", hlz: "HLZ", support_by_fire: "SBF",
};

type XZ = [number, number];
/** Position at ``station`` metres along a polyline. */
function along(line: XZ[], station: number): XZ {
  let left = station;
  for (let i = 1; i < line.length; i++) {
    const [ax, az] = line[i - 1], [bx, bz] = line[i];
    const seg = Math.hypot(bx - ax, bz - az);
    if (left <= seg || i === line.length - 1) { const f = seg ? Math.min(1, left / seg) : 0; return [ax + (bx - ax) * f, az + (bz - az) * f]; }
    left -= seg;
  }
  return line[line.length - 1];
}

/**
 * Sand table (RH-8): the objective as a briefing miniature. Symbols with affiliation frames
 * and sectors, routes, phase lines, and the approach played back phase by phase on its
 * Tobler timings. Fullscreen for a projector; nothing here is an observation.
 */
export function SandTable({ state, analysis, basemap, onClose }: { state: PlanState; analysis: MissionAnalysis | null; basemap: Basemap | null; onClose: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const shell = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const features = state.proposal.features.filter((f) => !f.hidden);
  const symbols = features.filter((f) => f.type === "symbol");
  const routes = features.filter((f) => f.type === "route");
  const phaseLines = features.filter((f) => f.type === "phase_line");
  const lead = routes.find((r) => (r.params as RouteParams).kind === "approach") ?? routes[0];
  const report = analysis?.routes.find((r) => r.id === lead?.id)?.report ?? null;
  const duration = report?.eta_s ?? 0;

  useEffect(() => {
    if (!basemap) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = `data:image/png;base64,${basemap.png_base64}`;
  }, [basemap]);

  const view = useMemo(() => {
    const xs: number[] = [], zs: number[] = [];
    for (const f of features) {
      if (f.type === "symbol") { const p = (f.params as SymbolParams).position; xs.push(p[0]); zs.push(p[1]); }
      if (f.type === "route") for (const [x, z] of (f.params as RouteParams).waypoints) { xs.push(x); zs.push(z); }
      if (f.type === "phase_line") for (const [x, z] of (f.params as PhaseLineParams).line) { xs.push(x); zs.push(z); }
    }
    if (!xs.length && basemap) { xs.push(basemap.bounds.x0, basemap.bounds.x1); zs.push(basemap.bounds.z0, basemap.bounds.z1); }
    if (!xs.length) { xs.push(-50, 50); zs.push(-50, 50); }
    const pad = 25;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, z0 = Math.min(...zs) - pad, z1 = Math.max(...zs) + pad;
    const size = Math.max(x1 - x0, z1 - z0);
    return { x0: (x0 + x1) / 2 - size / 2, z0: (z0 + z1) / 2 - size / 2, size };
  }, [features, basemap]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now(), raf = 0;
    const tick = (now: number) => {
      setT((prev) => { const next = prev + ((now - last) / 1000) * speed; if (next >= duration) { setPlaying(false); return duration; } return next; });
      last = now; raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration]);

  // Station along the lead route at time t, from the report's waypoint times.
  const station = useMemo(() => {
    if (!report || !report.waypoints.length) return 0;
    const w = report.waypoints;
    if (t <= w[0].time_s) return w[0].station_m;
    for (let i = 1; i < w.length; i++) {
      if (t <= w[i].time_s) { const f = (t - w[i - 1].time_s) / Math.max(1e-6, w[i].time_s - w[i - 1].time_s); return w[i - 1].station_m + f * (w[i].station_m - w[i - 1].station_m); }
    }
    return w[w.length - 1].station_m;
  }, [report, t]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    const px = (x: number) => ((x - view.x0) / view.size) * W;
    const pz = (z: number) => ((z - view.z0) / view.size) * H;
    const m = W / view.size;
    ctx.fillStyle = "#2a2f26"; ctx.fillRect(0, 0, W, H);
    if (image && basemap) {
      const b = basemap.bounds;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, ((view.x0 - b.x0) / (b.x1 - b.x0)) * image.width, ((view.z0 - b.z0) / (b.z1 - b.z0)) * image.height,
        (view.size / (b.x1 - b.x0)) * image.width, (view.size / (b.z1 - b.z0)) * image.height, 0, 0, W, H);
    }
    // Grid every 50 m for talk-on ("two squares east of the OP").
    ctx.strokeStyle = "rgba(255,255,255,.12)"; ctx.lineWidth = 1; ctx.font = "11px system-ui"; ctx.fillStyle = "rgba(255,255,255,.55)";
    const step = view.size > 600 ? 100 : 50;
    for (let x = Math.ceil(view.x0 / step) * step; x < view.x0 + view.size; x += step) { ctx.beginPath(); ctx.moveTo(px(x), 0); ctx.lineTo(px(x), H); ctx.stroke(); ctx.fillText(`${x}`, px(x) + 3, 12); }
    for (let z = Math.ceil(view.z0 / step) * step; z < view.z0 + view.size; z += step) { ctx.beginPath(); ctx.moveTo(0, pz(z)); ctx.lineTo(W, pz(z)); ctx.stroke(); ctx.fillText(`${z}`, 3, pz(z) - 3); }
    // Phase lines, labelled at both ends.
    for (const f of phaseLines) {
      const p = f.params as PhaseLineParams;
      const crossed = report?.phase_lines.find((l) => l.label === (p.label || f.name));
      const passed = !!crossed && crossed.time_s <= t;
      ctx.strokeStyle = passed ? "#9be89b" : "#e8e8e8"; ctx.lineWidth = 2.5; ctx.setLineDash([12, 6]);
      ctx.beginPath(); p.line.forEach(([x, z], i) => (i ? ctx.lineTo(px(x), pz(z)) : ctx.moveTo(px(x), pz(z)))); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = passed ? "#9be89b" : "#fff"; ctx.font = "bold 13px system-ui";
      for (const [x, z] of [p.line[0], p.line[p.line.length - 1]]) ctx.fillText(`PL ${p.label || f.name}`, px(x) + 6, pz(z) - 6);
    }
    // Sectors of fire under the symbols.
    for (const f of symbols) {
      const p = f.params as SymbolParams;
      if (p.affiliation !== "hostile" || !p.sector_deg || ["objective", "obstacle", "checkpoint", "rally_point", "hlz"].includes(p.role)) continue;
      const facing = state.evaluation.features[f.id]?.facing_xz ?? [0, -1];
      const heading = Math.atan2(facing[1], facing[0]);
      const half = (Math.min(p.sector_deg, 360) / 2) * (Math.PI / 180);
      const r = p.range_m * m;
      ctx.fillStyle = "rgba(255,70,60,.14)"; ctx.strokeStyle = "rgba(255,70,60,.6)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px(p.position[0]), pz(p.position[1]));
      ctx.arc(px(p.position[0]), pz(p.position[1]), r, heading - half, heading + half); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    // Routes with direction arrows and waypoint names.
    for (const f of routes) {
      const p = f.params as RouteParams;
      const patrol = p.kind === "patrol";
      ctx.strokeStyle = patrol ? "#ff6b6b" : f.id === lead?.id ? "#4fb3ff" : "#9fd0ff"; ctx.lineWidth = 3.5;
      ctx.beginPath(); p.waypoints.forEach(([x, z], i) => (i ? ctx.lineTo(px(x), pz(z)) : ctx.moveTo(px(x), pz(z)))); ctx.stroke();
      for (let i = 1; i < p.waypoints.length; i++) {
        const [ax, az] = p.waypoints[i - 1], [bx, bz] = p.waypoints[i];
        const mx = px((ax + bx) / 2), mz = pz((az + bz) / 2), ang = Math.atan2(pz(bz) - pz(az), px(bx) - px(ax));
        ctx.save(); ctx.translate(mx, mz); ctx.rotate(ang); ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-6, -6); ctx.lineTo(-6, 6); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      ctx.font = "bold 12px system-ui"; ctx.fillStyle = "#fff";
      p.waypoints.forEach(([x, z], i) => { ctx.beginPath(); ctx.arc(px(x), pz(z), 4, 0, Math.PI * 2); ctx.fill(); ctx.fillText(p.names[i] || `WP${i + 1}`, px(x) + 7, pz(z) + 14); });
    }
    // Symbols: APP-6 style frames (hostile diamond, friendly rectangle, unknown quatrefoil-ish circle, neutral square).
    for (const f of symbols) {
      const p = f.params as SymbolParams;
      const fr = FRAME[p.affiliation] ?? FRAME.unknown;
      const x = px(p.position[0]), z = pz(p.position[1]), s = 15;
      ctx.fillStyle = fr.fill; ctx.strokeStyle = fr.stroke; ctx.lineWidth = 2.5;
      ctx.beginPath();
      if (p.affiliation === "hostile") { ctx.moveTo(x, z - s); ctx.lineTo(x + s, z); ctx.lineTo(x, z + s); ctx.lineTo(x - s, z); ctx.closePath(); }
      else if (p.affiliation === "friendly") ctx.rect(x - s * 1.3, z - s * 0.85, s * 2.6, s * 1.7);
      else if (p.affiliation === "neutral") ctx.rect(x - s, z - s, s * 2, s * 2);
      else ctx.arc(x, z, s, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#111"; ctx.font = "bold 9px system-ui"; ctx.textAlign = "center";
      ctx.fillText(ROLE_GLYPH[p.role] ?? p.role.slice(0, 3).toUpperCase(), x, z + 3);
      ctx.fillStyle = "#fff"; ctx.font = "11px system-ui"; ctx.fillText(f.name, x, z + s + 14);
      if (p.count > 1) ctx.fillText(`×${p.count}`, x + s + 8, z - s);
      ctx.textAlign = "start";
    }
    // The element moving along the lead route.
    if (lead && report) {
      const [x, z] = along((lead.params as RouteParams).waypoints as XZ[], station);
      ctx.fillStyle = "#4fb3ff"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(px(x), pz(z), 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    // Scale bar and north arrow (scene north unless georeferenced).
    const metres = [10, 20, 50, 100, 200, 500, 1000].find((v) => v * m > 90) ?? 1000;
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(12, H - 34, metres * m + 20, 24);
    ctx.fillStyle = "#fff"; ctx.fillRect(22, H - 18, metres * m, 4); ctx.font = "11px system-ui"; ctx.fillText(`${metres} m`, 22, H - 22);
    ctx.save(); ctx.translate(W - 30, 40); ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.moveTo(0, -18); ctx.lineTo(8, 8); ctx.lineTo(0, 3); ctx.lineTo(-8, 8); ctx.closePath(); ctx.fill();
    ctx.font = "bold 12px system-ui"; ctx.textAlign = "center"; ctx.fillText("N", 0, 24); ctx.restore(); ctx.textAlign = "start";
  }, [view, image, basemap, symbols, routes, phaseLines, state, lead, report, station, t]);

  const waypointTimes = report?.waypoints ?? [];
  const phase = [...waypointTimes].reverse().find((w) => w.time_s <= t);
  const fullscreen = () => { const el = shell.current; if (!el) return; if (document.fullscreenElement) void document.exitFullscreen(); else void el.requestFullscreen(); };

  return <div className="aar sand-table" ref={shell} role="dialog" aria-label="Sand table briefing">
    <div className="aar-head">
      <div><strong>Sand table · {state.proposal.name}</strong><small>Planned, not observed · {state.frame.status === "georeferenced" ? "north is true north" : "north is scene north (-Z)"}{report ? "" : " · analyse routes to play the approach"}</small></div>
      <span className="aar-actions"><button className="button secondary small" onClick={fullscreen}><Icon name="expand" size={13} />Fullscreen</button>
        <button className="icon-button" aria-label="Close sand table" onClick={onClose}><Icon name="close" size={15} /></button></span>
    </div>
    <div className="sand-body"><canvas ref={canvas} width={1100} height={1100} className="sand-map" aria-label="Top-down briefing map" /></div>
    {report && <div className="aar-controls">
      <button className="icon-button" aria-label={playing ? "Pause" : "Play"} onClick={() => { if (t >= duration) setT(0); setPlaying(!playing); }}><Icon name={playing ? "stop" : "play"} size={15} /></button>
      <input type="range" min={0} max={duration} step={0.5} value={t} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} aria-label="Time" />
      <span className="aar-clock">{fmtClock(t)} / {fmtClock(duration)}{phase ? ` · past ${phase.name}` : ""}</span>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">{[4, 8, 16, 32].map((s) => <option key={s} value={s}>{s}×</option>)}</select>
      {waypointTimes.map((w) => <button key={w.name + w.time_s} className="text-button" onClick={() => { setPlaying(false); setT(w.time_s); }}>{w.name}</button>)}
    </div>}
  </div>;
}
