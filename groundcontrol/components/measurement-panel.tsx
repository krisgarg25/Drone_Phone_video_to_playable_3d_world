"use client";

import { useState } from "react";
import { workspace, type MeasurementKind, type Point, type ProjectDetail } from "@/lib/workspace";
import { Icon, type IconName } from "./studio-icons";

const TOOLS: { kind: MeasurementKind; label: string; icon: IconName }[] = [
  { kind: "point", label: "Annotation", icon: "pin" }, { kind: "distance", label: "Distance", icon: "ruler" },
  { kind: "height", label: "Height", icon: "arrow" }, { kind: "area", label: "Plan area", icon: "layers" },
  { kind: "volume", label: "Volume", icon: "cube" },
];
export function MeasurementPanel({ project, ready, points, kind, onTool, onClear, onUndo, onSaved, onSelect }: {
  project: ProjectDetail; ready: boolean; points: Point[]; kind: MeasurementKind | null;
  onTool: (kind: MeasurementKind) => void; onClear: () => void; onUndo: () => void; onSaved: (project: ProjectDetail) => void;
  onSelect: (id: string | null) => void;
}) {
  const [label, setLabel] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const valid = kind === "point" ? points.length === 1 : kind === "height" ? points.length === 2 : kind === "distance" ? points.length >= 2 : (kind === "area" || kind === "volume") ? points.length >= 3 : false;
  const staleDraft = !!kind && revision !== project.model_revision;
  async function save() {
    if (!kind || !valid || !label.trim() || staleDraft) return;
    setBusy(true); setError("");
    try { onSaved(await workspace.measure(project.id, { label: label.trim(), kind, points, model_revision: revision })); onClear(); setLabel(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  async function remove(id: string) {
    setBusy(true); setError("");
    try { onSaved(await workspace.removeMeasurement(project.id, id)); setDeleting(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <>
    <section className="inspector-section"><h2>Measure & annotate</h2><div className="measure-tools">{TOOLS.map((tool) => <button key={tool.kind} className={`button secondary small ${kind === tool.kind ? "active" : ""}`} disabled={!ready || busy} onClick={() => { onTool(tool.kind); setRevision(project.model_revision); setLabel(""); setError(""); }}><Icon name={tool.icon} size={15} />{tool.label}</button>)}</div>
      {kind && <div className="measure-draft"><p>{kind === "point" ? "Click one point on the model." : kind === "height" ? "Pick two points to compare their elevation." : kind === "area" ? "Pick at least three points around a boundary. Area is projected onto the horizontal plane." : kind === "volume" ? "Trace a footprint of at least three points around a mound; volume is measured above the fitted ground plane." : "Pick two or more points along a line."} <strong>{points.length} selected.</strong></p><label className="field">{kind === "point" ? "Finding / annotation" : "Measurement name"}<input value={label} maxLength={120} placeholder={kind === "point" ? "Describe what you observed" : "e.g. Northern wall"} onChange={(event) => setLabel(event.target.value)} /></label>{staleDraft && <p className="form-error">The model changed. Select the tool again before measuring.</p>}<button className="button primary small full" disabled={!valid || !label.trim() || staleDraft || busy} onClick={() => void save()}>{busy ? "Saving…" : "Save to project"}</button>{points.length > 0 && <button className="button secondary small full" onClick={onUndo} disabled={busy}>Undo last point ({points.length})</button>}<button className="button secondary small full" onClick={onClear} disabled={busy}>Cancel selection</button></div>}
      <div className="notice warning"><Icon name="info" /><p>Points snap to the collision surface, an estimated proxy—not the visible splats. {project.scale.status === "relative" ? "Distances use relative units." : "Metre values inherit the model’s scale uncertainty."} Independent survey accuracy is not established.</p></div>
      {error && <p className="form-error" role="alert">{error}</p>}
    </section>
    <section className="inspector-section"><div className="section-label"><span>SAVED IN THIS PROJECT</span><span>{project.measurements.length}</span></div>{project.measurements.length === 0 ? <p className="inspector-copy">Select a tool and pick points in the model. Saved observations stay with this project.</p> : <ul className="measure-list">{project.measurements.map((measurement) => <li key={measurement.id} onMouseEnter={() => onSelect(measurement.id)} onMouseLeave={() => onSelect(null)}><Icon name={measurement.kind === "point" ? "pin" : "ruler"} size={16} /><div><strong>{measurement.label}</strong><small>{measurement.kind === "area" ? "Horizontal plan area" : measurement.kind} · model-derived</small>{measurement.value !== null && <span className="measure-value">{measurement.value.toLocaleString(undefined, { maximumFractionDigits: 3 })} <small>{measurement.unit}</small>{measurement.uncertainty && <small className="measure-unc"> ± {measurement.uncertainty.m.toLocaleString(undefined, { maximumFractionDigits: 3 })} {measurement.uncertainty.unit}</small>}</span>}{measurement.engine && measurement.support === false && <small>Weakly supported by measured geometry — treat as indicative.</small>}{!measurement.engine && measurement.value !== null && <small>Relative units · no point cloud to snap against</small>}{measurement.stale && <small className="stale">Previous model version · remeasure before use</small>}{deleting === measurement.id && <div><button className="button danger small" onClick={() => void remove(measurement.id)} disabled={busy}>Confirm delete</button><button className="text-button" onClick={() => setDeleting(null)}>Keep</button></div>}</div><button className="icon-button" aria-label={`Delete ${measurement.label}`} onClick={() => setDeleting(measurement.id)} disabled={busy}><Icon name="trash" size={14} /></button></li>)}</ul>}</section>
  </>;
}
