"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { CAPTURES, WORKFLOWS, bytes, workspace, type Capture, type Workflow } from "@/lib/workspace";
import { Icon } from "./studio-icons";

export function ImportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const id = useRef<string | null>(null);
  const router = useRouter();
  const [name, setName] = useState("");
  const [workflow, setWorkflow] = useState<Workflow>("general");
  const [capture, setCapture] = useState<Capture>("unknown");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => { if (open) dialog.current?.showModal(); else dialog.current?.close(); }, [open]);
  const hasVideo = files.some((file) => /\.(mp4|mov|mkv|webm|m4v|avi)$/i.test(file.name));
  function addFiles(incoming: File[]) {
    const allowed = /\.(mp4|mov|mkv|webm|m4v|avi|gpx|srt|csv|jsonl|json)$/i;
    if (incoming.some((file) => !allowed.test(file.name))) { setError("Choose video files or GPS, pose, and calibration files in the supported formats."); return; }
    const combined = [...files, ...incoming].filter((file, index, all) => all.findIndex((other) => other.name === file.name) === index);
    if (combined.reduce((total, file) => total + file.size, 0) > 2 * 1024 ** 3) { setError("Upload up to 2 GB at a time. Use a shorter original clip for this project."); return; }
    setFiles(combined); setError("");
  }
  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim() || !hasVideo || busy) return;
    setBusy(true); setError("");
    try {
      id.current ??= `${name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 45) || "capture"}-${crypto.randomUUID().slice(0, 8)}`;
      await workspace.save({ scene: id.current, name: name.trim(), workflow, capture });
      await workspace.upload(id.current, files, setProgress);
      router.push(`/projects/${encodeURIComponent(id.current)}`);
      onClose();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <dialog ref={dialog} className="studio-dialog import-dialog" onCancel={(event) => { if (busy) event.preventDefault(); else onClose(); }}>
    <form onSubmit={(event) => void create(event)}>
      <div className="dialog-heading"><div><span className="eyebrow">NEW RECONSTRUCTION</span><h2>Start with a video.</h2><p>Turn a capture into a workspace you can explore.</p></div><button type="button" className="icon-button" aria-label="Close import" disabled={busy} onClick={onClose}><Icon name="close" /></button></div>
      <div className="dialog-body">
        <label className="field">Project name<input autoFocus value={name} onChange={(event) => setName(event.target.value)} maxLength={120} placeholder="e.g. North bridge inspection" disabled={busy} /></label>
        <div className={`dropzone ${dragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); if (!busy) addFiles(Array.from(event.dataTransfer.files)); }}>
          <span className="upload-symbol"><Icon name="upload" size={26} /></span><strong>Drop your video here</strong><span>or <button type="button" className="text-button" onClick={() => input.current?.click()} disabled={busy}>browse files</button></span><small>MP4, MOV, MKV, WebM, M4V, AVI · Up to 2 GB</small>
          <input ref={input} type="file" multiple accept=".mp4,.mov,.mkv,.webm,.m4v,.avi,.gpx,.srt,.csv,.jsonl,.json" aria-label="Video and optional sensor files" onChange={(event) => { addFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} className="visually-hidden" disabled={busy} />
        </div>
        {files.length > 0 && <ul className="upload-files">{files.map((file) => <li key={file.name}><Icon name={/\.(mp4|mov|mkv|webm|m4v|avi)$/i.test(file.name) ? "video" : "file"} /><span>{file.name}<small>{bytes(file.size)}</small></span><button type="button" className="icon-button" aria-label={`Remove ${file.name}`} disabled={busy} onClick={() => setFiles(files.filter((entry) => entry !== file))}><Icon name="close" size={14} /></button></li>)}</ul>}
        <div className="field-row"><label className="field">Captured with<select aria-label="Captured with" value={capture} onChange={(event) => setCapture(event.target.value as Capture)} disabled={busy}>{Object.entries(CAPTURES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label className="field">Your workflow<select aria-label="Your workflow" value={workflow} onChange={(event) => setWorkflow(event.target.value as Workflow)} disabled={busy}>{Object.entries(WORKFLOWS).map(([value, item]) => <option key={value} value={value}>{item.label} — {item.description}</option>)}</select></label></div>
        <div className="notice"><Icon name="info" /><p>GPS is optional. You can add telemetry or pose logs alongside the video. Without a reliable scale reference, measurements remain relative.</p></div>
        {error && <p role="alert" className="form-error">{error}</p>}
        {busy && <div className="upload-progress"><progress max={100} value={progress} /><span>{progress < 100 ? `Uploading · ${progress}%` : "Saving capture…"}</span></div>}
      </div>
      <div className="dialog-footer"><span>Private · stored on your machine</span><button className="button primary" type="submit" disabled={!name.trim() || !hasVideo || busy}>{busy ? "Creating project…" : "Create project"}<Icon name="arrow" size={16} /></button></div>
    </form>
  </dialog>;
}
