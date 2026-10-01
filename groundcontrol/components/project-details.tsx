"use client";

import { useState } from "react";
import { bytes, CAPTURES, WORKFLOWS, workspace, measurementsGeoJSON, measurementsCSV, downloadBlob, type ProjectDetail, type Workflow, type Capture, APPLICATIONS, type ApplicationId } from "@/lib/workspace";
import type { GnssReport, Identifiability } from "@/lib/api";
import { Icon } from "./studio-icons";
import { SurveyControls } from "./survey-controls";

const UNPROVABLE: Record<keyof Identifiability, string> = {
  clock_offset: "clock offset", lever_arm_along_track: "along-track lever arm",
  lever_arm_cross_track: "cross-track lever arm", metric_scale: "metric scale",
  rotation_about_trajectory_axis: "rotation about the track axis",
};
const num = (value: number) => Number.isInteger(value) ? String(value) : value.toFixed(2);
const metres = (value: number | null) => value == null ? "—" : `${num(value)} m`;
const secs = (value: number | null) => value == null ? "—" : `${num(value)} s`;

/**
 * What the flight's own GPS proves, rendered in the panel that already carries the
 * coordinate reference. The numbers were computed at prepare time and reach here off
 * the same re-verified preparation manifest that the CRS line comes from.
 *
 * Two distinctions the wording is obliged to keep. An uncertainty here is a per-axis
 * standard deviation the receiver reported, not a covariance and not a radius. And the
 * clock figure is a *window* the two time series cannot escape — `survey_gnss` bounds
 * the offset and never estimates it, which is why `offset_estimate_s` is null by
 * design; so the row reads as a bound, and the metres quoted next to it are the
 * position error that bound admits at the caller's speed ceiling, not a measured
 * displacement. Accuracy therefore stays unverified here however tight the numbers
 * look: only surveyed checkpoints can lift that, and this scene's telemetry cannot.
 */
export function GnssReference({ project }: { project: ProjectDetail }) {
  const gnss: GnssReport | null | undefined = project.survey?.gnss;
  const absent = (state: string, reason: string) => <>
    <div className="datum-row"><span>GPS telemetry</span><span>{state}</span></div>
    <p className="inspector-copy">{reason}</p>
  </>;
  if (!project.survey) return absent("Not supplied", "No flight telemetry has been prepared for this scene, so there is no positioning uncertainty and no clock bound to report. Adding telemetry under Details produces both — it produces accuracy only when surveyed checkpoints are compared against the model.");
  const quality = gnss?.quality, clock = gnss?.clock, fix = gnss?.fix_quality, observed = gnss?.observability;
  if (!gnss || (!quality && !clock && !observed && !fix)) return absent("No diagnostics", "This preparation holds no GNSS gate report, so nothing below can be bounded from it. Prepare the scene again to recompute the telemetry diagnostics.");
  const unprovable = Object.entries(observed?.identifiability ?? {})
    .filter(([, separable]) => !separable)
    .map(([key]) => UNPROVABLE[key as keyof Identifiability] ?? key);
  const findings = [...(observed?.never_claim ?? []), ...(quality?.warnings ?? []),
    ...(clock?.warnings ?? []), ...(fix?.warnings ?? []), ...(observed?.warnings ?? [])];
  return <>
    {quality && <div className="datum-row"><span>GPS fixes</span><span title={gnss.std_basis}>{quality.count} · {secs(quality.duration_s)} · {metres(quality.path_length_m)} of path</span></div>}
    {quality && <div className="datum-row"><span>Reported σ / fix step</span><span title={gnss.std_basis}>{metres(quality.median_horizontal_std_m)} E&amp;N · {metres(quality.median_vertical_std_m)} up, steps {metres(quality.median_step_m)}</span></div>}
    {quality && <div className="datum-row"><span>Trajectory above noise</span><span>{quality.displacement_below_noise ? "No — per-fix motion sits below σ" : "Yes — steps exceed σ"}</span></div>}
    {quality && (quality.gaps_over_threshold_count > 0 || quality.suspicious_count > 0) && <div className="datum-row"><span>Dropouts / implausible</span><span>{quality.gaps_over_threshold_count} gap{quality.gaps_over_threshold_count === 1 ? "" : "s"} longer than {secs(quality.gap_threshold_s)} · {quality.suspicious_count} fix{quality.suspicious_count === 1 ? "" : "es"} above {num(quality.max_speed_m_s)} m/s</span></div>}
    {clock && <div className="datum-row"><span>GPS↔video clock offset</span><span>{clock.status === "bounded" ? `Bounded ${secs(clock.offset_bounds_s?.[0] ?? null)} to ${secs(clock.offset_bounds_s?.[1] ?? null)} · never estimated` : `Not bounded — ${secs(clock.infeasibility_s)} of the clip has no GNSS coverage`}</span></div>}
    {clock && <div className="datum-row"><span>Worst-case along-track error</span><span title={`${clock.convention}; the offset itself is never estimated, so this is the maximum displacement any admissible offset can hide, not a measured error`}>{clock.worst_case_along_track_error_m == null ? "—" : `≤ ${metres(clock.worst_case_along_track_error_m)} across the ${secs(clock.offset_width_s)} window at ${num(clock.max_speed_m_s)} m/s`}</span></div>}
    {observed && <div className="datum-row"><span>Separable on this path</span><span>{unprovable.length ? `Not identifiable: ${unprovable.join(", ")}` : "Clock offset, lever arms and metric scale"}</span></div>}
    <p className="inspector-copy">These are per-axis standard deviations the receiver reported and a coverage bound on the two clocks; neither is a measured error. {gnss.accuracy_validated ? "This preparation claims validated accuracy, which needs surveyed checkpoints behind it." : "Accuracy stays unverified here: no surveyed reference has been compared against this model."} {fix && (fix.hdop_supplied ? "Declared dilution widened the one-sided fix-quality floors, never narrowed them." : `No dilution was declared, so the floors assume HDOP ≤ 1 geometry${fix.fallback ? ` and ${fix.fallback} stand-in values cover the rows with no recorded fix type` : ""}.`)}</p>
    {findings.length > 0 && <details className="gnss-findings"><summary>Telemetry findings ({findings.length})</summary>{findings.map((finding, index) => <p key={index} className="inspector-copy warning-copy">{finding}</p>)}</details>}
  </>;
}

export function ExportPanel({ project }: { project: ProjectDetail }) {
  function exportMeasurements() {
    downloadBlob(`${project.id}-observations.json`, "application/json", JSON.stringify({ scene: project.id, coordinate_system: "viewer Y-up", scale: project.scale, measurements: project.measurements }, null, 2));
  }
  function exportGeoJSON() { downloadBlob(`${project.id}-measurements.geojson`, "application/geo+json", JSON.stringify(measurementsGeoJSON(project.measurements, project.id), null, 2)); }
  function exportCSV() { downloadBlob(`${project.id}-measurements.csv`, "text/csv", measurementsCSV(project.measurements)); }
  return <><section className="inspector-section"><a className="export-cta" href={`/projects/${encodeURIComponent(project.id)}/export`}><Icon name="download" size={20} /><span><strong>Open the export page</strong><small>Pick files, rename them, choose formats and download in one go.</small></span><Icon name="arrow" size={16} /></a></section><section className="inspector-section"><h2>Quick downloads</h2>{project.artifacts.length > 0 && <a className="button primary small full" href={`/api/backend/api/workspace/bundle?scene=${encodeURIComponent(project.id)}`} download={`${project.id}-bundle.zip`}><Icon name="download" size={15} />Download project bundle (.zip)</a>}{project.artifacts.length > 0 && <p className="inspector-copy">Model, measurements, quality reports and the coordinate reference in one file, with a README of every caveat and a SHA-256 for each file.</p>}</section><section className="inspector-section"><div className="artifact-list">{project.artifacts.map((file) => <a key={file.url} href={file.url.includes("download=1") ? file.url : `${file.url}${file.url.includes("?") ? "&" : "?"}download=1`} className="artifact-link"><Icon name="file" size={20} /><div><strong>{file.name}</strong><small>{file.kind} · {bytes(file.bytes)}</small></div><Icon name="download" size={15} /></a>)}</div>{!project.artifacts.length && <p className="inspector-copy">No generated files yet. Run reconstruction to create model outputs.</p>}{project.measurements.length > 0 && <div className="export-buttons"><button className="button secondary full" onClick={exportMeasurements}><Icon name="download" size={15} />Download JSON</button><button className="button secondary full" onClick={exportGeoJSON}><Icon name="globe" size={15} />Download GeoJSON (local)</button><button className="button secondary full" onClick={exportCSV}><Icon name="file" size={15} />Download CSV</button></div>}</section><section className="inspector-section"><div className="section-label"><span>Coordinates</span></div><div className="datum-row"><span>Interactive viewer</span><span>Local · Y-up</span></div><div className="datum-row"><span>Scale</span><span>{project.scale.status}</span></div><div className="datum-row"><span>Location</span><span>{project.georeference.status === "georeferenced" ? "Georeferenced deliverables" : "Local coordinates"}</span></div>{project.georeference.status === "georeferenced" && <div className="datum-row"><span>CRS</span><span>{project.georeference.crs}</span></div>}{project.georeference.status === "georeferenced" && <div className="datum-row"><span>Vertical datum</span><span>{project.georeference.vertical_datum ?? "—"}</span></div>}<div className="datum-row"><span>Accuracy</span><span>{project.accuracy.status === "verified" ? `${project.accuracy.rmse_m?.toFixed(3)} m 3D RMSE` : "Not independently verified"}</span></div>{project.accuracy.status === "verified" && <div className="datum-row"><span>Checkpoints</span><span>{project.accuracy.checkpoints ?? "—"}</span></div>}<p className="inspector-copy">{project.georeference.status === "georeferenced" ? "Validated survey evidence exists for this scene; the interactive viewer still stays in local Y-up coordinates." : "No validated survey evidence yet. Add telemetry in Details to produce georeferenced outputs."}</p><GnssReference project={project} /></section></>;
}

export function DetailsPanel({ project, onSaved, refresh }: { project: ProjectDetail; onSaved: (value: ProjectDetail) => void; refresh: () => void }) {
  const [name, setName] = useState(project.name);
  const [notes, setNotes] = useState(project.notes);
  const [workflow, setWorkflow] = useState(project.workflow);
  const [application, setApplication] = useState<ApplicationId | "">(project.application?.id ?? "");
  const [site, setSite] = useState(project.site ?? "");
  const [capture, setCapture] = useState(project.capture);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  async function save() {
    setBusy(true); setError(""); setMessage("");
    try { onSaved(await workspace.save({ scene: project.id, name, notes, workflow, capture, application: application || null, site })); setMessage("Project details saved."); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <><section className="inspector-section"><h2>Project details</h2><label className="field">Project name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} /></label><label className="field">Workflow<select value={workflow} onChange={(event) => setWorkflow(event.target.value as Workflow)}>{Object.entries(WORKFLOWS).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label><label className="field">Application<select value={application} onChange={(event) => setApplication(event.target.value as ApplicationId | "")}><option value="">Not set — all tools</option>{Object.entries(APPLICATIONS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label><label className="field">Site id<input value={site} onChange={(event) => setSite(event.target.value)} maxLength={80} placeholder="Links repeat flights of one site (epochs)" /></label><label className="field">Captured with<select value={capture} onChange={(event) => setCapture(event.target.value as Capture)}>{Object.entries(CAPTURES).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select></label><label className="field">Project notes<textarea aria-label="Project notes" value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={4000} placeholder="Site context, capture conditions, or inspection notes…" /></label><button className="button primary small full" onClick={() => void save()} disabled={busy || !name.trim()}>{busy ? "Saving…" : "Save details"}</button>{message && <p role="status" className="inspector-copy">{message}</p>}{error && <p role="alert" className="form-error">{error}</p>}</section>
    <section className="inspector-section"><h3>Source videos</h3>{project.videos.map((video) => <a className="artifact-link" key={video.url} href={video.url} target="_blank" rel="noreferrer"><Icon name="video" size={18} /><div><strong>{video.name}</strong><small>{bytes(video.bytes)} · open original</small></div><Icon name="arrow" size={14} /></a>)}{!project.videos.length && <p className="inspector-copy">Original capture is not available locally. Generated files remain accessible.</p>}</section>
    <SurveyControls scene={project.id} onChange={refresh} />
    {project.warnings.length > 0 && <section className="inspector-section"><h3>Capture & model notes</h3>{project.warnings.map((warning, index) => <p key={index} className="inspector-copy warning-copy">{warning}</p>)}</section>}
    {project.diagnostics && <details className="inspector-section"><summary>Capture analysis</summary><pre className="job-terminal">{JSON.stringify(project.diagnostics, null, 2)}</pre></details>}
  </>;
}
