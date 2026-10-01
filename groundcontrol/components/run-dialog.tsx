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
  const [vertical, setVertical] = useState<"ellipsoidal" | "egm96">("ellipsoidal");
  const [anchor, setAnchor] = useState("none");
  const [anchorValue, setAnchorValue] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { dialog.current?.showModal(); }, []);
  const activeJob = project.job.status === "running" || project.job.status === "starting";
  const validAnchor = engine === "survey" || anchor === "none" || (Number.isFinite(Number(anchorValue)) && Number(anchorValue) > 0);
  // The quality tier's own numbers, read from the server, instead of strings written
  // beside them that drift the first time a tier is re-tuned. The card says what the run
  // will actually spend: pixels, steps and the collider voxel it will build with.
  const qualityLabel = (key: string) => {
    const q = options?.qualities?.[key];
    if (!q) return key;
    return `${key} · ${q.width ?? "?"} px · ${q.steps?.toLocaleString() ?? "?"} steps · ${q.voxel ?? "?"} m collider`;
  };
  const advice = preset !== "auto" ? String(options?.presets?.[preset]?.advice ?? "") : "";
  async function start(action: "run" | "scan") {
    setBusy(true); setError("");
    try {
      const body: RunOptions = { scene: project.id, preset, quality, action, engine: action === "scan" ? "pipeline" : engine };
      if (engine === "survey" && action === "run") { body.dense_profile = dense; body.vertical_datum = vertical; }
      if (engine === "pipeline" && anchor !== "none") body.anchor = { kind: anchor as "height" | "speed", value: Number(anchorValue) };
      await workspace.run(body); onStarted(); onClose();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <dialog ref={dialog} className="studio-dialog" onCancel={(event) => { if (busy) event.preventDefault(); else onClose(); }}>
    <div className="dialog-heading"><div><span className="eyebrow">Build the 3D model</span><h2>Build your next perspective.</h2><p>{project.name} · {project.video_count} source video{project.video_count !== 1 ? "s" : ""}</p>{project.scenario ? <p className="run-recorded">Last built as <b>{project.scenario.label || project.scenario.preset}</b> · {project.scenario.decided_by || "no record of how it was chosen"}{project.quality ? ` · world gate: ${project.quality.status === "failed" ? `blocked (${project.quality.hard_failures.join(", ")})` : project.quality.status === "warnings" ? `usable with ${project.quality.warnings.length} warning(s)` : "all checks passed"}` : ""}</p> : null}</div><button className="icon-button" aria-label="Close reconstruction settings" disabled={busy} onClick={onClose}><Icon name="close" /></button></div>
    <div className="run-panel">
      <label className="field">Output<select value={engine} onChange={(event) => { setEngine(event.target.value as "pipeline" | "survey"); setConfirmed(false); }} disabled={busy}><option value="pipeline">Photorealistic 3D workspace · splats + walkable geometry</option><option value="survey">Survey surface & exports · prepared GPS survey required</option></select></label>
      {engine === "pipeline" ? <div className="field-row"><label className="field">Scene type<select aria-label="Scene type" value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>{Object.entries(options?.presets ?? { auto: {} }).map(([key, value]) => <option key={key} value={key}>{key === "auto" ? "Detect from capture" : String(value.label ?? key)}</option>)}</select></label><label className="field">Processing quality<select aria-label="Processing quality" value={quality} onChange={(event) => setQuality(event.target.value)} disabled={busy}>{Object.keys(options?.qualities ?? { standard: {} }).map((key) => <option key={key} value={key}>{qualityLabel(key)}</option>)}</select></label></div> : <div className="field-row"><label className="field">Dense reconstruction<select value={dense} onChange={(event) => setDense(event.target.value)}><option value="survey">Geometric consistency · slower</option><option value="fast">Fast · reduced density</option><option value="budget">Resource-constrained</option></select></label><label className="field">Heights in the exports<select aria-label="Heights in the exports" value={vertical} onChange={(event) => setVertical(event.target.value as "ellipsoidal" | "egm96")} disabled={busy}><option value="ellipsoidal">Ellipsoidal (WGS84, as GPS measures)</option><option value="egm96">Mean sea level (EGM96 geoid)</option></select><small>{vertical === "egm96" ? "Heights on maps and GCP sheets. Needs the EGM96 grid in data/geoid; without it the run keeps ellipsoidal heights and says so." : "Differs from sea level by the local geoid: tens of metres in most places."}</small></label></div>}
      {/* The capture guidance each preset carries has shipped in the table since it was
          written and has never once been shown to a human. Choosing "Aerial Drone Orbit"
          without being told to fly 60-70% overlap is how a capture fails at the top of
          the pipeline instead of the bottom. */}
      {engine === "pipeline" && advice ? <p className="run-advice"><Icon name="camera" size={14} /><span><b>How to capture this scene</b>{advice}</span></p> : null}
      {engine === "pipeline" && preset === "auto" ? <p className="run-advice"><Icon name="info" size={14} /><span><b>Detect from capture</b>The probe measures the clip before anything runs: a pose log gives the metres walked and how fast, otherwise frame-to-frame steadiness separates a hand from a gimbal. The frames alone cannot tell an aerial orbit from a mapping grid or a corridor run; when the scene has a GPS log (Details → Georeferenced survey) the flight path decides it and sets the scale. Without one, a facade pass and the mission type stay your choice.</span></p> : null}
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
