"use client";

import { useEffect, useRef, useState } from "react";
import { workspace, type ProjectDetail, type ProjectList, type RunOptions } from "@/lib/workspace";
import { Icon } from "./studio-icons";

export function RunDialog({ project, options, onClose, onStarted }: { project: ProjectDetail; options: ProjectList | null; onClose: () => void; onStarted: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [preset, setPreset] = useState("auto");
  const [quality, setQuality] = useState("standard");
  const [engine, setEngine] = useState<"pipeline" | "survey">("pipeline");
  const [dense, setDense] = useState("survey");
  const [anchor, setAnchor] = useState("none");
  const [anchorValue, setAnchorValue] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { dialog.current?.showModal(); }, []);
  const activeJob = project.job.status === "running" || project.job.status === "starting";
  const validAnchor = engine === "survey" || anchor === "none" || (Number.isFinite(Number(anchorValue)) && Number(anchorValue) > 0);
  const qualityNames: Record<string, string> = { smoke: "Quick preview · short training", standard: "Balanced · 640 px", high: "Detailed · 1280 px", ultra: "Maximum detail · 1440 px" };
  async function start(action: "run" | "scan") {
    setBusy(true); setError("");
    try {
      const body: RunOptions = { scene: project.id, preset, quality, action, engine: action === "scan" ? "pipeline" : engine };
      if (engine === "survey" && action === "run") body.dense_profile = dense;
      if (engine === "pipeline" && anchor !== "none") body.anchor = { kind: anchor as "height" | "speed", value: Number(anchorValue) };
      await workspace.run(body); onStarted(); onClose();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <dialog ref={dialog} className="studio-dialog" onCancel={(event) => { if (busy) event.preventDefault(); else onClose(); }}>
    <div className="dialog-heading"><div><span className="eyebrow">RECONSTRUCTION</span><h2>Build your next perspective.</h2><p>{project.name} · {project.video_count} source video{project.video_count !== 1 ? "s" : ""}</p></div><button className="icon-button" aria-label="Close reconstruction settings" disabled={busy} onClick={onClose}><Icon name="close" /></button></div>
    <div className="run-panel">
      <label className="field">Output<select value={engine} onChange={(event) => { setEngine(event.target.value as "pipeline" | "survey"); setConfirmed(false); }} disabled={busy}><option value="pipeline">Photorealistic 3D workspace · splats + walkable geometry</option><option value="survey">Survey surface & exports · prepared GPS survey required</option></select></label>
      {engine === "pipeline" ? <div className="field-row"><label className="field">Scene type<select aria-label="Scene type" value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>{Object.entries(options?.presets ?? { auto: {} }).map(([key, value]) => <option key={key} value={key}>{key === "auto" ? "Detect from capture" : String(value.label ?? key)}</option>)}</select></label><label className="field">Processing quality<select aria-label="Processing quality" value={quality} onChange={(event) => setQuality(event.target.value)} disabled={busy}>{Object.keys(options?.qualities ?? { standard: {} }).map((key) => <option key={key} value={key}>{qualityNames[key] ?? key}</option>)}</select></label></div> : <div className="field-row"><label className="field">Dense reconstruction<select value={dense} onChange={(event) => setDense(event.target.value)}><option value="survey">Geometric consistency · slower</option><option value="fast">Fast · reduced density</option><option value="budget">Resource-constrained</option></select></label></div>}
      {engine === "pipeline" && <div className="field-row"><label className="field">Optional scale reference<select value={anchor} onChange={(event) => setAnchor(event.target.value)} disabled={busy}><option value="none">Use available pose data / estimate</option><option value="height">Known camera height above ground</option><option value="speed">Known camera travel speed</option></select></label>{anchor !== "none" && <label className="field">{anchor === "height" ? "Height (metres)" : "Speed (metres/second)"}<input type="number" min="0.01" step="any" value={anchorValue} onChange={(event) => setAnchorValue(event.target.value)} disabled={busy} /><small>Supply a real reference—not a guessed value.</small></label>}</div>}
      <div className="notice"><Icon name="info" /><p>{engine === "survey" ? "Save and prepare telemetry in Project details first. Survey exports are separate from the visual splat; location and accuracy are not interchangeable." : "Processing runs on this machine. Duration depends on the video and GPU. Missing views cannot be recovered as measured geometry. Existing completed stages are reused when valid."}</p></div>
      {activeJob && <p className="form-error">Another job is running. Wait for it to finish or stop it from its project.</p>}
      {!project.video_count && <p className="form-error">The source video is not available. Existing model outputs can still be explored.</p>}
      <label className="run-confirm"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} disabled={busy} /><span>Run reconstruction on this machine using its GPU. A rerun may replace this project’s generated outputs.</span></label>
      {error && <p role="alert" className="form-error">{error}</p>}
    </div>
    <div className="dialog-footer"><button className="button secondary" onClick={() => void start("scan")} disabled={busy || activeJob || !project.video_count || !validAnchor}><Icon name="activity" size={16} />Analyze footage</button><button className="button primary" onClick={() => void start("run")} disabled={!confirmed || busy || activeJob || !project.video_count || !validAnchor}><Icon name="play" size={15} />{busy ? "Starting…" : "Start reconstruction"}</button></div>
  </dialog>;
}
