"use client";

import { useState } from "react";
import { workspace, measurementsCSV, measurementsGeoJSON, downloadBlob, type MeasurementKind, type Point, type ProjectDetail } from "@/lib/workspace";
import { previewMeasurement } from "../../viewer/measurement_math.js";
import { Icon, type IconName } from "./studio-icons";

const TOOLS: { kind: MeasurementKind; label: string; icon: IconName; hint: string }[] = [
  { kind: "distance", label: "Distance", icon: "ruler", hint: "Point-to-point or a path" },
  { kind: "height", label: "Height", icon: "arrow", hint: "Vertical difference" },
  { kind: "area", label: "Plan area", icon: "layers", hint: "Trace a horizontal boundary" },
  { kind: "volume", label: "Volume", icon: "cube", hint: "Above fitted ground" },
  { kind: "point", label: "Annotation", icon: "pin", hint: "Mark an observation" },
];
const number = (value: number) => value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export function MeasurementPanel({ project, ready, points, hover, kind, onTool, onClear, onUndo, onSaved, onSelect }: {
  project: ProjectDetail; ready: boolean; points: Point[]; hover: Point | null; kind: MeasurementKind | null;
  onTool: (kind: MeasurementKind) => void; onClear: () => void; onUndo: () => void; onSaved: (project: ProjectDetail) => void;
  onSelect: (id: string | null) => void;
}) {
  const [label, setLabel] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const unit = project.scale.unit === "m" ? "m" : "units";
  const valid = kind === "point" ? points.length === 1 : kind === "height" ? points.length === 2 : kind === "distance" ? points.length >= 2 : (kind === "area" || kind === "volume") ? points.length >= 3 : false;
  const staleDraft = !!kind && revision !== project.model_revision;
  const previewPoints = hover && points.length && !(kind === "height" && points.length === 2) && kind !== "point" ? [...points, hover] : points;
  const preview = previewMeasurement(kind, previewPoints, unit);
  const tool = TOOLS.find((t) => t.kind === kind);
  async function save() {
    if (!kind || !valid || staleDraft || busy) return;
    setBusy(true); setError("");
    try {
      onSaved(await workspace.measure(project.id, { label: label.trim() || `${tool?.label ?? "Measurement"} ${project.measurements.length + 1}`, kind, points, model_revision: revision }));
      onClear(); setLabel("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  async function remove(id: string) {
    setBusy(true); setError("");
    try { onSaved(await workspace.removeMeasurement(project.id, id)); setDeleting(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <>
    <section className="inspector-section measurement-intro"><span className="eyebrow">GEOMETRY / FIELD NOTES</span><h2>Measurement desk</h2><p className="inspector-copy">Pick on the model. Read the result as you move. Save only what you want to keep.</p>
      <div className="measure-tools">{TOOLS.map((t) => <button key={t.kind} className={`measurement-tool ${kind === t.kind ? "active" : ""}`} aria-label={t.label} aria-pressed={kind === t.kind} disabled={!ready || busy} onClick={() => { onTool(t.kind); setRevision(project.model_revision); setLabel(""); setError(""); }}><Icon name={t.icon} size={19} /><span>{t.label}<small>{t.hint}</small></span></button>)}</div>
      {!ready && <p className="inspector-copy">Picking needs a connected viewer and a collision surface.</p>}
    </section>
    {kind && <section className="inspector-section measure-draft">
      <div className="live-readout" role="status" aria-label="Live measurement"><span>{tool?.label} · {hover && previewPoints.length > points.length ? "cursor preview" : "picked points"}</span><strong>{preview.value == null ? "—" : number(preview.value)} <small>{preview.value == null ? "" : preview.unit}</small></strong><span>{kind === "volume" ? "Volume requires the cloud calculation on save" : kind === "point" ? "Annotation position" : "Surface estimate · not survey-verified"}</span></div>
      {kind === "distance" && preview.value != null && <div className="measurement-components"><span>Horizontal <b>{number(preview.horizontal)} {unit}</b></span><span>Elevation <b>{number(preview.vertical)} {unit}</b></span></div>}
      <p className="inspector-copy">{kind === "point" ? "Click one point to mark an observation." : kind === "height" ? "Click the bottom, then the top. Height is the vertical difference." : kind === "area" || kind === "volume" ? "Click around the footprint. Three or more points close the boundary." : "Click to add each endpoint. Move the cursor to preview the next segment."} <strong>{points.length} selected.</strong></p>
      <label className="field">{kind === "point" ? "Finding / annotation" : "Measurement name"}<input value={label} maxLength={120} placeholder={kind === "point" ? "Describe what you observed" : `${tool?.label} ${project.measurements.length + 1}`} onChange={(event) => setLabel(event.target.value)} /></label>
      {staleDraft && <p className="form-error">The model changed. Select the tool again.</p>}
      <div className="draft-actions"><button className="button primary small" disabled={!valid || staleDraft || busy || !ready} onClick={() => void save()}>{busy ? "Saving…" : "Save to project"}</button><button className="button secondary small" onClick={onUndo} disabled={!points.length || busy}>Undo last point ({points.length})</button></div>
      <button className="text-button" onClick={onClear} disabled={busy}>Cancel selection</button>
    </section>}
    <section className="inspector-section"><div className="section-label"><span>MEASUREMENT REGISTER</span><span>{project.measurements.length}</span></div>
      {!project.measurements.length && <div className="tool-empty"><Icon name="ruler" size={26} /><p>No saved measurements yet.</p><small>Live readings do not need to be saved.</small></div>}
      <ul className="measure-list">{project.measurements.map((m) => {
        const estimate = previewMeasurement(m.kind, m.points, unit);
        const supported = m.engine && m.valid !== false && m.support !== false && m.value != null;
        const value = supported ? m.value : estimate.value;
        return <li key={m.id}><div><button className="measurement-select" onClick={() => onSelect(m.id)}><Icon name={m.kind === "point" ? "pin" : "ruler"} size={16} /><strong>{m.label}</strong></button>
          {value != null && <span className="measure-value">{supported ? "" : "≈ "}{number(value)} <small>{supported ? m.unit : estimate.unit}</small></span>}
          <small>{m.kind === "point" ? "Observation" : supported ? "Cloud-supported · scale uncertainty applies" : "Surface estimate · cloud support unverified"}</small>
          {supported && m.uncertainty?.m != null && <small>Uncertainty ±{number(m.uncertainty.m)} {m.uncertainty.unit}</small>}
          {!supported && m.reason && <details><summary>Support details</summary><p className="inspector-copy">{m.reason}</p></details>}
          {m.stale && <small className="stale">Previous model version · remeasure before use</small>}
          {deleting === m.id && <div><button className="button danger small" onClick={() => void remove(m.id)} disabled={busy}>Confirm delete</button><button className="text-button" onClick={() => setDeleting(null)}>Keep</button></div>}
        </div><button className="icon-button" aria-label={`Delete ${m.label}`} onClick={() => setDeleting(m.id)} disabled={busy}><Icon name="trash" size={14} /></button></li>;
      })}</ul>
      {!!project.measurements.length && <div className="draft-actions"><button className="button secondary small" onClick={() => downloadBlob(`${project.id}-measurements.csv`, "text/csv", measurementsCSV(project.measurements))}>Export CSV</button><button className="button secondary small" onClick={() => downloadBlob(`${project.id}-measurements.geojson`, "application/geo+json", JSON.stringify(measurementsGeoJSON(project.measurements, project.id)))}>Export GeoJSON</button></div>}
    </section>
    <section className="inspector-section"><details className="measurement-evidence"><summary>What these numbers mean</summary><p className="inspector-copy">Live readings use the collision surface, not visible splats. Saving checks sparse-cloud support separately. Unsupported readings stay estimates; volume is never inferred from a flat outline. Scale: {project.scale.status}. {project.scale.source}</p></details>{error && <p className="form-error" role="alert">{error}</p>}</section>
  </>;
}
