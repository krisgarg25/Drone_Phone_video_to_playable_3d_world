"use client";

import { useState } from "react";
import Link from "next/link";
import { workspace, measurementsCSV, measurementsGeoJSON, downloadBlob, type Measurement, type MeasurementKind, type Point, type ProjectDetail } from "@/lib/workspace";
import { previewMeasurement } from "../../viewer/measurement_math.js";
import { Icon, type IconName } from "./studio-icons";
import { OnViewer } from "./on-viewer";
import "./inspect.css";

const TOOLS: { kind: MeasurementKind; label: string; icon: IconName; hint: string; clicks: string }[] = [
  { kind: "distance", label: "Distance", icon: "ruler", hint: "Straight line or a path", clicks: "2+ clicks" },
  { kind: "height", label: "Height", icon: "angle", hint: "Bottom, then top", clicks: "2 clicks" },
  { kind: "area", label: "Area", icon: "grid", hint: "Trace around a shape", clicks: "3+ clicks" },
  { kind: "volume", label: "Volume", icon: "cube", hint: "A pile above the ground", clicks: "3+ clicks" },
  { kind: "point", label: "Note", icon: "pin", hint: "Mark something you saw", clicks: "1 click" },
];
const HOW: Record<MeasurementKind, string> = {
  point: "Click the spot you want to note.",
  distance: "Click each point along the line. The reading follows your cursor.",
  height: "Click the bottom, then the top. Height is the straight-up difference.",
  area: "Click around the outline. Three or more points close the shape.",
  volume: "Click around the base of the pile. It is measured on save.",
};
const number = (value: number) => value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const centre = (m: Measurement): Point => {
  const n = m.points.length || 1;
  return m.points.reduce<Point>((sum, p) => [sum[0] + p[0] / n, sum[1] + p[1] / n, sum[2] + p[2] / n], [0, 0, 0]);
};

export function MeasurementPanel({ project, ready, points, hover, kind, onTool, onClear, onUndo, onSaved, onSelect, onFocus }: {
  project: ProjectDetail; ready: boolean; points: Point[]; hover: Point | null; kind: MeasurementKind | null;
  onTool: (kind: MeasurementKind) => void; onClear: () => void; onUndo: () => void; onSaved: (project: ProjectDetail) => void;
  onSelect: (id: string | null) => void; onFocus?: (point: Point) => void;
}) {
  const [label, setLabel] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [filter, setFilter] = useState<MeasurementKind | "all">("all");
  const [picked, setPicked] = useState<string | null>(null);
  const unit = project.scale.unit === "m" ? "m" : "units";
  const valid = kind === "point" ? points.length === 1 : kind === "height" ? points.length === 2 : kind === "distance" ? points.length >= 2 : (kind === "area" || kind === "volume") ? points.length >= 3 : false;
  const staleDraft = !!kind && revision !== project.model_revision;
  const previewPoints = hover && points.length && !(kind === "height" && points.length === 2) && kind !== "point" ? [...points, hover] : points;
  const preview = previewMeasurement(kind, previewPoints, unit);
  const tool = TOOLS.find((t) => t.kind === kind);
  const list = project.measurements.filter((m) => filter === "all" || m.kind === filter);
  const scale = project.scale.status;
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
  return <div className="measure-panel">
    <section className="inspector-section insp-top">
      <div className={`measure-scale scale-${scale}`}>
        <Icon name={scale === "metric" ? "check" : "alert"} size={16} />
        <span><b>{scale === "metric" ? "Real-world metres" : scale === "estimated" ? "Approximate metres" : "No real scale"}</b>
          {scale === "metric" ? "Scale comes from the flight's position data. Accuracy is not checked against surveyed points." : scale === "estimated" ? "Scale was estimated from the flight. Expect a few percent error." : "Readings are in model units, not metres. Add a scale reference and rebuild."}</span>
      </div>
      <div className="measure-grid" role="group" aria-label="What to measure">{TOOLS.map((t) => <button key={t.kind} className={`measure-card${kind === t.kind ? " on" : ""}`} aria-pressed={kind === t.kind} disabled={!ready || busy}
        onClick={() => { if (kind === t.kind) { onClear(); return; } onTool(t.kind); setRevision(project.model_revision); setLabel(""); setError(""); }}>
        <Icon name={t.icon} size={20} /><b>{t.label}</b><small>{t.hint}</small><em>{t.clicks}</em></button>)}</div>
      {!ready && <p className="insp-busy"><span className="spinner" />Tools unlock once the 3D view has loaded.</p>}
    </section>

    {kind && <OnViewer kicker={`${tool?.label ?? "Measure"} · click on the model`}><section className="inspector-section measure-draft">
      <div className="live-readout" role="status" aria-label="Live measurement"><span>{tool?.label}{hover && previewPoints.length > points.length ? " · following cursor" : ""}</span><strong>{preview.value == null ? "—" : number(preview.value)} <small>{preview.value == null ? "" : preview.unit}</small></strong><span>{kind === "volume" ? "Worked out when you save" : kind === "point" ? "Where the note goes" : "Live reading on the model surface"}</span></div>
      {kind === "distance" && preview.value != null && <div className="measurement-components"><span>Level distance <b>{number(preview.horizontal)} {unit}</b></span><span>Height change <b>{number(preview.vertical)} {unit}</b></span></div>}
      <p className="inspector-copy">{HOW[kind]} <strong>{points.length} point{points.length === 1 ? "" : "s"}.</strong></p>
      <label className="field">{kind === "point" ? "What did you see?" : "Name"}<input value={label} maxLength={120} placeholder={kind === "point" ? "e.g. Crack above the door" : `${tool?.label} ${project.measurements.length + 1}`} onChange={(event) => setLabel(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void save(); }} /></label>
      {staleDraft && <p className="form-error">The model was rebuilt. Pick the tool again.</p>}
      <div className="draft-actions"><button className="button primary small" disabled={!valid || staleDraft || busy || !ready} onClick={() => void save()}><Icon name="check" size={14} />{busy ? "Saving…" : "Save"}</button><button className="button secondary small" onClick={onUndo} disabled={!points.length || busy}>Undo point</button></div>
      <button className="text-button" onClick={onClear} disabled={busy}>Cancel</button>
    </section></OnViewer>}

    <section className="inspector-section insp-body">
      <header className="insp-head"><div><h3>Saved measurements</h3><p>{project.measurements.length ? "Click one to show it on the model." : "Readings you save appear here and in exports."}</p></div></header>
      {project.measurements.length > 0 && <div className="insp-chips">{(["all", ...TOOLS.map((t) => t.kind)] as const).map((k) => { const n = k === "all" ? project.measurements.length : project.measurements.filter((m) => m.kind === k).length;
        return n || k === "all" ? <button key={k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>{k === "all" ? "All" : TOOLS.find((t) => t.kind === k)?.label}<em>{n}</em></button> : null; })}</div>}
      {!project.measurements.length ? <div className="tool-empty"><Icon name="ruler" size={26} /><p>Nothing saved yet.</p><small>Pick a tool above, click on the model, then Save.</small></div>
        : <ul className="measure-rows">{list.map((m) => {
          const estimate = previewMeasurement(m.kind, m.points, unit);
          const supported = m.engine && m.valid !== false && m.support !== false && m.value != null;
          const value = supported ? m.value : estimate.value;
          const t = TOOLS.find((x) => x.kind === m.kind);
          const trust = m.stale ? ["bad", "Old model"] : m.kind === "point" ? ["", "Note"] : supported ? ["good", "Checked"] : ["warn", "Estimate"];
          return <li key={m.id} className={picked === m.id ? "on" : ""}>
            <button className="measure-row" onClick={() => { setPicked(m.id); onSelect(m.id); }}>
              <span className="measure-kind"><Icon name={t?.icon ?? "ruler"} size={16} /></span>
              <span className="measure-what"><strong>{m.label}</strong><small>{t?.label}{supported && m.uncertainty?.m != null ? ` · ±${number(m.uncertainty.m)} ${m.uncertainty.unit}` : ""}</small></span>
              {value != null && <span className="measure-num">{supported ? "" : "≈ "}{number(value)}<small>{supported ? m.unit : estimate.unit}</small></span>}
            </button>
            <div className="measure-row-foot">
              <span className={`measure-trust ${trust[0]}`} data-tip={m.stale ? "Measured on an older build of the model. Measure again before use." : m.kind === "point" ? "A note, not a measurement" : supported ? "Checked against the point cloud; scale uncertainty still applies" : m.reason ?? "The point cloud does not support this reading well, so it stays an estimate"}>{trust[1]}</span>
              {onFocus && m.points.length > 0 && <button className="text-button" onClick={() => { setPicked(m.id); onSelect(m.id); onFocus(centre(m)); }}><Icon name="target" size={13} />Go to</button>}
              {deleting === m.id ? <><button className="button danger small" onClick={() => void remove(m.id)} disabled={busy}>Delete</button><button className="text-button" onClick={() => setDeleting(null)}>Keep</button></>
                : <button className="icon-button" aria-label={`Delete ${m.label}`} data-tip="Delete" onClick={() => setDeleting(m.id)} disabled={busy}><Icon name="trash" size={14} /></button>}
            </div>
          </li>;
        })}</ul>}
      {project.measurements.length > 0 && <div className="insp-actions">
        <button className="button secondary small" onClick={() => downloadBlob(`${project.id}-measurements.csv`, "text/csv", measurementsCSV(project.measurements))}><Icon name="download" size={14} />CSV</button>
        <button className="button secondary small" onClick={() => downloadBlob(`${project.id}-measurements.geojson`, "application/geo+json", JSON.stringify(measurementsGeoJSON(project.measurements, project.id)))}><Icon name="download" size={14} />GeoJSON</button>
        <Link className="button ghost small" href={`/projects/${encodeURIComponent(project.id)}/export`}>More formats</Link>
      </div>}
      <details className="ops-notes"><summary>What &ldquo;Checked&rdquo; and &ldquo;Estimate&rdquo; mean</summary><p className="plan-note">Live readings are taken on the solid surface, not the photo-real model. On save, each reading is checked against the point cloud: <b>Checked</b> means the cloud supports it; <b>Estimate</b> means it does not, so treat it with care. Volume is never guessed from a flat outline. {project.scale.source}</p></details>
      {error && <p className="form-error" role="alert">{error}</p>}
    </section>
  </div>;
}
