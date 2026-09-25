"use client";

import type { RefObject } from "react";
import Image from "next/image";
import { duration, type Mode, type ProjectDetail } from "@/lib/workspace";
import { Icon } from "./studio-icons";

export function ViewerPanel({ project, iframeRef, ready, mode, selectedFrame, message, framesOpen, onCommand, onFrame, onFramesToggle, onRun }: {
  project: ProjectDetail; iframeRef: RefObject<HTMLIFrameElement | null>; ready: boolean;
  mode: Mode; selectedFrame: number; message: string; framesOpen: boolean;
  onCommand: (command: string, options?: Record<string, unknown>) => void;
  onFrame: (index: number) => void; onFramesToggle: () => void; onRun: () => void;
}) {
  return <section className="viewer-section" aria-label="Reconstruction viewer">
    <div className="viewport">
      {project.viewer_url ? <iframe ref={iframeRef} key={project.model_revision} src={project.viewer_url} title="3D reconstruction" allow="fullscreen; pointer-lock" onLoad={() => onCommand("get-state")} /> : <div className="viewport-placeholder">{project.thumbnail_url && <Image unoptimized width={640} height={400} src={project.thumbnail_url} alt="" />}<Icon name="cube" size={48} /><h2>Your scene starts here.</h2><p>{project.video_count ? "Your capture is ready. Analyze the footage, then reconstruct it to open the 3D workspace." : "The model or required viewer assets are not available yet."}</p><button className="button primary" onClick={onRun} disabled={!project.video_count}><Icon name="play" size={15} />Set up reconstruction</button></div>}
      {project.viewer_url && <><div className="viewer-toolbar" aria-label="Navigation mode">{(["orbit", "fly", "walk"] as Mode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} disabled={!ready} onClick={() => onCommand("mode", { value: item })}>{item[0].toUpperCase() + item.slice(1)}</button>)}</div><div className="viewer-toolbar viewer-tools"><button className="icon-button" title="Fit model to view" aria-label="Fit model to view" disabled={!ready} onClick={() => onCommand("fit")}><Icon name="expand" size={16} /></button><button className="icon-button" title="Save viewport snapshot" aria-label="Save viewport snapshot" disabled={!ready} onClick={() => onCommand("snapshot")}><Icon name="camera" size={16} /></button><button className="icon-button" title="Fullscreen viewer" aria-label="Fullscreen viewer" onClick={() => void iframeRef.current?.requestFullscreen()}><Icon name="eye" size={16} /></button></div>{!ready && !message && <div className="viewport-loading">Opening reconstruction…</div>}<div className="viewport-caption"><span>{mode === "orbit" ? "DRAG TO ORBIT · SCROLL TO ZOOM · RIGHT DRAG TO PAN" : "WASD TO MOVE · DRAG TO LOOK"}</span></div></>}
      {message && <div className="viewport-message" role="status">{message}</div>}
    </div>
    <div className="frame-strip"><div className="frame-strip-heading"><button onClick={onFramesToggle}><Icon name="image" size={15} />Source frames<Icon name="chevron" size={12} style={{ transform: framesOpen ? "rotate(90deg)" : "rotate(-90deg)" }} /></button><span>{project.frames.length} source frames</span></div>{framesOpen && (project.frames.length ? <div className="frame-list">{project.frames.map((frame, index) => <button key={`${frame.name}-${index}`} className={selectedFrame === index ? "selected" : ""} aria-label={`Inspect frame ${index + 1}`} title={frame.name} onClick={() => onFrame(index)}><Image unoptimized width={640} height={400} src={frame.url} alt={`Source frame ${index + 1}`} loading="lazy" /><span>{frame.t_sec != null ? duration(frame.t_sec) : `#${index + 1}`}</span></button>)}</div> : <div className="frame-list-empty">Source views appear after camera reconstruction.</div>)}</div>
  </section>;
}
