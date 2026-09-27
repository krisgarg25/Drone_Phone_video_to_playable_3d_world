"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import Image from "next/image";
import { duration, type Mode, type ProjectDetail } from "@/lib/workspace";
import { Icon } from "./studio-icons";

/** Swipe / side-by-side: a second viewer on the Existing view, cameras synced by the host. */
export type CompareView = {
  mode: "swipe" | "side"; iframeRef: RefObject<HTMLIFrameElement | null>; position: number;
  onPosition: (percent: number) => void; onLoad: () => void;
};

function CompareLayer({ mode, frameRef, position, onPosition, onLoad, src, revision }: {
  mode: CompareView["mode"]; frameRef: CompareView["iframeRef"]; position: number; onPosition: (percent: number) => void;
  onLoad: () => void; src: string; revision: string;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const at = (clientX: number) => {
    const rect = viewport.current?.parentElement?.getBoundingClientRect();
    if (rect && rect.width > 0) onPosition(Math.min(95, Math.max(5, ((clientX - rect.left) / rect.width) * 100)));
  };
  const swipe = mode === "swipe";
  return <div ref={viewport} className={`compare-layer compare-${mode}`}>
    <iframe ref={frameRef} key={`compare-${revision}`} src={src} title="Existing scan (compare)" className="compare-frame"
      style={swipe ? { clipPath: `inset(0 ${100 - position}% 0 0)` } : undefined} onLoad={onLoad} />
    {swipe && <div className={`compare-divider${dragging ? " dragging" : ""}`} style={{ left: `${position}%` }} role="slider" tabIndex={0}
      aria-label="Swipe between existing and proposed" aria-valuemin={5} aria-valuemax={95} aria-valuenow={Math.round(position)}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDragging(true); at(e.clientX); }}
      onPointerMove={(e) => { if (dragging) at(e.clientX); }}
      onPointerUp={(e) => { e.currentTarget.releasePointerCapture(e.pointerId); setDragging(false); }}
      onKeyDown={(e) => { if (e.key === "ArrowLeft") onPosition(Math.max(5, position - 5)); if (e.key === "ArrowRight") onPosition(Math.min(95, position + 5)); }}>
      <span aria-hidden>⟨ ⟩</span></div>}
    {dragging && <div className="compare-shield" onPointerMove={(e) => at(e.clientX)} onPointerUp={() => setDragging(false)} />}
    <div className="compare-tags" aria-hidden><span>EXISTING</span><span>PROPOSED</span></div>
  </div>;
}

export function ViewerPanel({ gameQuery, compare, project, iframeRef, ready, mode, selectedFrame, message, framesOpen, showFrames = true, showMessage = true, arena = false, game = false, bots = 3, onCommand, onFrame, onFramesToggle, onRun, onExitGame }: {
  gameQuery?: string; compare?: CompareView; project: ProjectDetail; iframeRef: RefObject<HTMLIFrameElement | null>; ready: boolean;
  mode: Mode; selectedFrame: number; message: string; framesOpen: boolean; showFrames?: boolean; showMessage?: boolean;
  arena?: boolean; game?: boolean; bots?: number;
  onCommand: (command: string, options?: Record<string, unknown>) => void;
  onFrame: (index: number) => void; onFramesToggle: () => void; onRun: () => void; onExitGame?: () => void;
}) {
  // The arena screen reports failure, nothing else: a load line over the play area
  // would be HUD the game already owns. A message left over from a session that has
  // already ended says nothing about this one, so it is not shown rather than being
  // reset from an effect (which is a cascading render React warns about).
  const [gameError, setGameError] = useState("");
  const src = game ? `/runtime/viewer/pc.html?asset=/runtime/work/${encodeURIComponent(project.id)}/viewer_assets&combat=1&bots=${bots}${gameQuery ? `&${gameQuery}` : ""}` : project.viewer_url;
  useEffect(() => {
    if (!game) return;
    const timer = setInterval(() => {
      const view = iframeRef.current?.contentWindow as (Window & { __combat?: unknown; __combatError?: string; __loadError?: string }) | null;
      if (view?.__combatError || view?.__loadError) { setGameError(`Game could not start: ${view.__combatError || view.__loadError}`); clearInterval(timer); }
      else if (view?.__combat) { setGameError(""); clearInterval(timer); }
    }, 800);
    return () => clearInterval(timer);
  }, [game, iframeRef, src]);
  return <section className="viewer-section" aria-label="Reconstruction viewer">
    <div className={`viewport${compare && project.viewer_url && !game ? ` comparing comparing-${compare.mode}` : ""}`}>
      {compare && project.viewer_url && !game && <CompareLayer mode={compare.mode} frameRef={compare.iframeRef} position={compare.position} onPosition={compare.onPosition} onLoad={compare.onLoad} src={project.viewer_url} revision={project.model_revision} />}
      {project.viewer_url ? <iframe ref={iframeRef} key={`${project.model_revision}-${game}`} src={src!} title={game ? "Local game session" : "3D reconstruction"} allow="fullscreen" onLoad={() => { if (!game) onCommand("get-state"); }} /> : <div className="viewport-placeholder">{project.thumbnail_url && <Image unoptimized width={640} height={400} src={project.thumbnail_url} alt="" />}<Icon name="cube" size={48} /><h2>Your scene starts here.</h2><p>{project.video_count ? "Your capture is ready. Analyze the footage, then reconstruct it to open the 3D workspace." : "The model or required viewer assets are not available yet."}</p><button className="button primary" onClick={onRun} disabled={!project.video_count}><Icon name="play" size={15} />Set up reconstruction</button></div>}
      {project.viewer_url && !game && <><div className="viewer-toolbar" aria-label="Navigation mode">{(["orbit", "fly", "walk"] as Mode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} disabled={!ready} onClick={() => onCommand("mode", { value: item })}>{item[0].toUpperCase() + item.slice(1)}</button>)}</div><div className="viewer-toolbar viewer-tools"><button className="icon-button" title="Fit model to view" aria-label="Fit model to view" disabled={!ready} onClick={() => onCommand("fit")}><Icon name="expand" size={16} /></button><button className="icon-button" title="Save viewport snapshot" aria-label="Save viewport snapshot" disabled={!ready} onClick={() => onCommand("snapshot")}><Icon name="camera" size={16} /></button><button className="icon-button" title="Fullscreen viewer" aria-label="Fullscreen viewer" onClick={() => void iframeRef.current?.requestFullscreen()}><Icon name="eye" size={16} /></button></div>{!ready && !message && <div className="viewport-loading">Opening reconstruction…</div>}<div className="viewport-caption"><span>{mode === "orbit" ? "DRAG TO ORBIT · SCROLL TO ZOOM · RIGHT DRAG TO PAN" : "WASD TO MOVE · DRAG TO LOOK"}</span></div></>}
      {game && arena && <div className="arena-exit"><button className="button secondary small" onClick={onExitGame}><Icon name="close" size={14} />Exit</button></div>}
      {arena ? game && gameError && <div className="viewport-message" role="status">{gameError}</div>
        : (showMessage ? message : "") && <div className="viewport-message" role="status">{message}</div>}
    </div>
    {showFrames && <div className="frame-strip"><div className="frame-strip-heading"><button onClick={onFramesToggle}><Icon name="image" size={15} />Source frames<Icon name="chevron" size={12} style={{ transform: framesOpen ? "rotate(90deg)" : "rotate(-90deg)" }} /></button><span>{project.frames.length} source frames</span></div>{framesOpen && (project.frames.length ? <div className="frame-list">{project.frames.map((frame, index) => <button key={`${frame.name}-${index}`} className={selectedFrame === index ? "selected" : ""} aria-label={`Inspect frame ${index + 1}`} title={frame.name} onClick={() => onFrame(index)}><Image unoptimized width={640} height={400} src={frame.url} alt={`Source frame ${index + 1}`} loading="lazy" /><span>{frame.t_sec != null ? duration(frame.t_sec) : `#${index + 1}`}</span></button>)}</div> : <div className="frame-list-empty">Source views appear after camera reconstruction.</div>)}</div>}
  </section>;
}
