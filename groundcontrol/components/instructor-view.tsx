"use client";

import { useEffect, useRef, useState } from "react";
import { missionApi, type Basemap, type SessionSnapshot } from "@/lib/mission";
import { Icon } from "./studio-icons";

/**
 * Instructor view (RH-9): who is in the exercise and where, where this session's enemy posts
 * are (as the first player's client simulates them), and three controls - move a post, send a
 * message, end the exercise. Join links carry the session token; LAN links use the local HTTPS
 * port so a headset browser may use WebXR.
 */
export function InstructorView({ scene, missionId, basemap, onClose }: { scene: string; missionId: string; basemap: Basemap | null; onClose: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<{ session: string; token: string; links: { path: string; lan: string[]; note: string } } | null>(null);
  const [snap, setSnap] = useState<SessionSnapshot | null>(null);
  const [bot, setBot] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!basemap) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = `data:image/png;base64,${basemap.png_base64}`;
  }, [basemap]);

  useEffect(() => {
    if (!session) return;
    let live = true;
    const poll = async () => {
      try { const s = await missionApi.sessionSnapshot(scene, missionId, session.session); if (live) setSnap(s); } catch (e) { if (live) setError(e instanceof Error ? e.message : String(e)); }
      if (live) timer = setTimeout(poll, 700);
    };
    let timer = setTimeout(poll, 0);
    return () => { live = false; clearTimeout(timer); };
  }, [session, scene, missionId]);

  const b = basemap?.bounds;
  const toCanvas = (x: number, z: number, W: number, H: number) => b ? [((x - b.x0) / (b.x1 - b.x0)) * W, ((z - b.z0) / (b.z1 - b.z0)) * H] : [W / 2, H / 2];
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.fillStyle = "#1b2027"; ctx.fillRect(0, 0, W, H);
    if (image) { ctx.imageSmoothingEnabled = false; ctx.drawImage(image, 0, 0, W, H); }
    (snap?.bots ?? []).forEach(([x, , z], k) => {
      const [px, pz] = toCanvas(x, z, W, H);
      ctx.fillStyle = k === bot ? "#ffd33d" : "#e5484d";
      ctx.beginPath(); ctx.moveTo(px, pz - 8); ctx.lineTo(px + 8, pz); ctx.lineTo(px, pz + 8); ctx.lineTo(px - 8, pz); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.fillText(`E${k}`, px + 9, pz - 6);
    });
    for (const p of snap?.players ?? []) {
      if (!p.pose) continue;
      const [px, pz] = toCanvas(p.pose[0], p.pose[2], W, H);
      ctx.fillStyle = p.lost ? "#8b949e" : p.alive ? "#4fb3ff" : "#6e7681";
      ctx.beginPath(); ctx.arc(px, pz, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, pz); ctx.lineTo(px - Math.sin(p.pose[3]) * 14, pz - Math.cos(p.pose[3]) * 14); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "11px system-ui"; ctx.fillText(p.name, px + 9, pz + 4);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap, image, bot]);

  const open = async () => {
    setError("");
    try { setSession(await missionApi.openSession(scene, missionId, "Exercise")); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const send = async (command: Record<string, unknown>) => {
    if (!session) return;
    try { await missionApi.sessionCommand(scene, missionId, session.session, command); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const clickMap = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (bot === null || !b || !snap) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = b.x0 + ((event.clientX - rect.left) / rect.width) * (b.x1 - b.x0);
    const z = b.z0 + ((event.clientY - rect.top) / rect.height) * (b.z1 - b.z0);
    const y = snap.bots[bot]?.[1] ?? 0;
    void send({ type: "move_bot", bot, to: [x, y, z] });
    setBot(null);
  };
  const localLink = session ? session.links.path.replace("/viewer/pc.html", "/runtime/viewer/pc.html").replace("asset=/work/", "asset=/runtime/work/") : "";

  return <div className="aar sand-table" role="dialog" aria-label="Instructor view">
    <div className="aar-head"><div><strong>Instructor · multi-user rehearsal</strong><small>Poses are shared; each player&apos;s client simulates the enemy posts, so hits are not replayed across clients.</small></div>
      <span className="aar-actions">{session && <button className="button danger small" onClick={() => void send({ type: "end" })}><Icon name="stop" size={13} />End exercise</button>}
        <button className="icon-button" aria-label="Close instructor view" onClick={onClose}><Icon name="close" size={15} /></button></span></div>
    {error && <p className="plan-message" role="alert">{error}</p>}
    {!session ? <div className="instructor-start"><p className="aar-note">Open a session, then give each player a join link. Players on this machine use the local link; headsets and laptops on the same network use a LAN link.</p>
      <button className="button primary small" onClick={() => void open()}><Icon name="play" size={13} />Open session</button></div>
      : <div className="aar-body">
        <canvas ref={canvas} width={640} height={640} className="aar-map" onClick={clickMap} style={{ cursor: bot !== null ? "crosshair" : "default" }} aria-label="Live exercise map" />
        <div className="instructor-side">
          <h4>Join links</h4>
          <a className="text-button" href={localLink} target="_blank" rel="noreferrer">Join from this machine</a>
          {session.links.lan.map((l) => <code key={l} className="instructor-link">{l}</code>)}
          {!session.links.lan.length && <p className="aar-note">No LAN address found; only this machine can join.</p>}
          <p className="aar-note">{session.links.note}</p>
          <h4>Players ({snap?.players.length ?? 0})</h4>
          <ul className="aar-events">{(snap?.players ?? []).map((p) => <li key={p.id}><span>{p.lost ? "lost" : p.alive ? "live" : "down"}</span>{p.name}{p.health !== null ? ` · ${Math.round(p.health)} hp` : ""}</li>)}</ul>
          <h4>Enemy posts</h4>
          {(snap?.bots ?? []).length ? <div className="instructor-bots">{snap!.bots.map((_, k) => <button key={k} className={`text-button${bot === k ? " active" : ""}`} onClick={() => setBot(bot === k ? null : k)}>E{k}</button>)}</div>
            : <p className="aar-note">Posts appear once a player is in.</p>}
          {bot !== null && <p className="aar-note">Click the map where E{bot} should move.</p>}
          <h4>Message</h4>
          <form onSubmit={(e) => { e.preventDefault(); if (text.trim()) { void send({ type: "message", text }); setText(""); } }}>
            <input value={text} maxLength={140} onChange={(e) => setText(e.target.value)} placeholder="Hold at PL BLUE" className="instructor-input" />
          </form>
        </div>
      </div>}
  </div>;
}
