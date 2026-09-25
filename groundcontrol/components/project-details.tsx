"use client";

import { useState } from "react";
import { bytes, CAPTURES, WORKFLOWS, workspace, measurementsGeoJSON, measurementsCSV, downloadBlob, type ProjectDetail, type Workflow, type Capture } from "@/lib/workspace";
import { Icon } from "./studio-icons";
import { SurveyControls } from "./survey-controls";

export function ExportPanel({ project }: { project: ProjectDetail }) {
  function exportMeasurements() {
    downloadBlob(`${project.id}-observations.json`, "application/json", JSON.stringify({ scene: project.id, coordinate_system: "viewer Y-up", scale: project.scale, measurements: project.measurements }, null, 2));
  }
  function exportGeoJSON() { downloadBlob(`${project.id}-measurements.geojson`, "application/geo+json", JSON.stringify(measurementsGeoJSON(project.measurements, project.id), null, 2)); }
  function exportCSV() { downloadBlob(`${project.id}-measurements.csv`, "text/csv", measurementsCSV(project.measurements)); }
  return <><section className="inspector-section"><h2>Project exports</h2><p className="inspector-copy">Download the files this project actually produced. A collision mesh supports navigation; it is not a textured survey surface.</p></section><section className="inspector-section"><div className="artifact-list">{project.artifacts.map((file) => <a key={file.url} href={file.url.includes("download=1") ? file.url : `${file.url}${file.url.includes("?") ? "&" : "?"}download=1`} className="artifact-link"><Icon name="file" size={20} /><div><strong>{file.name}</strong><small>{file.kind} · {bytes(file.bytes)}</small></div><Icon name="download" size={15} /></a>)}</div>{!project.artifacts.length && <p className="inspector-copy">No generated files yet. Run reconstruction to create model outputs.</p>}{project.measurements.length > 0 && <div className="export-buttons"><button className="button secondary full" onClick={exportMeasurements}><Icon name="download" size={15} />Download JSON</button><button className="button secondary full" onClick={exportGeoJSON}><Icon name="globe" size={15} />Download GeoJSON (local)</button><button className="button secondary full" onClick={exportCSV}><Icon name="file" size={15} />Download CSV</button></div>}</section><section className="inspector-section"><div className="section-label">COORDINATE REFERENCE</div><div className="datum-row"><span>Interactive viewer</span><span>Local · Y-up</span></div><div className="datum-row"><span>Scale</span><span>{project.scale.status}</span></div><div className="datum-row"><span>Location</span><span>{project.georeference.status === "georeferenced" ? "Georeferenced deliverables" : "Local coordinates"}</span></div>{project.georeference.status === "georeferenced" && <div className="datum-row"><span>CRS</span><span>{project.georeference.crs}</span></div>}{project.georeference.status === "georeferenced" && <div className="datum-row"><span>Vertical datum</span><span>{project.georeference.vertical_datum ?? "—"}</span></div>}<div className="datum-row"><span>Accuracy</span><span>{project.accuracy.status === "verified" ? `${project.accuracy.rmse_m?.toFixed(3)} m 3D RMSE` : "Not independently verified"}</span></div>{project.accuracy.status === "verified" && <div className="datum-row"><span>Checkpoints</span><span>{project.accuracy.checkpoints ?? "—"}</span></div>}<p className="inspector-copy">{project.georeference.status === "georeferenced" ? "Validated survey evidence exists for this scene; the interactive viewer still stays in local Y-up coordinates." : "No validated survey evidence yet. Add telemetry in Details to produce georeferenced outputs."}</p></section></>;
}

export function DetailsPanel({ project, onSaved, refresh }: { project: ProjectDetail; onSaved: (value: ProjectDetail) => void; refresh: () => void }) {
  const [name, setName] = useState(project.name);
  const [notes, setNotes] = useState(project.notes);
  const [workflow, setWorkflow] = useState(project.workflow);
  const [capture, setCapture] = useState(project.capture);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  async function save() {
    setBusy(true); setError(""); setMessage("");
    try { onSaved(await workspace.save({ scene: project.id, name, notes, workflow, capture })); setMessage("Project details saved."); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <><section className="inspector-section"><h2>Project details</h2><label className="field">Project name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} /></label><label className="field">Workflow<select value={workflow} onChange={(event) => setWorkflow(event.target.value as Workflow)}>{Object.entries(WORKFLOWS).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label><label className="field">Captured with<select value={capture} onChange={(event) => setCapture(event.target.value as Capture)}>{Object.entries(CAPTURES).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select></label><label className="field">Project notes<textarea aria-label="Project notes" value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={4000} placeholder="Site context, capture conditions, or inspection notes…" /></label><button className="button primary small full" onClick={() => void save()} disabled={busy || !name.trim()}>{busy ? "Saving…" : "Save details"}</button>{message && <p role="status" className="inspector-copy">{message}</p>}{error && <p role="alert" className="form-error">{error}</p>}</section>
    <section className="inspector-section"><h3>Source videos</h3>{project.videos.map((video) => <a className="artifact-link" key={video.url} href={video.url} target="_blank" rel="noreferrer"><Icon name="video" size={18} /><div><strong>{video.name}</strong><small>{bytes(video.bytes)} · open original</small></div><Icon name="arrow" size={14} /></a>)}{!project.videos.length && <p className="inspector-copy">Original capture is not available locally. Generated files remain accessible.</p>}</section>
    <SurveyControls scene={project.id} onChange={refresh} />
    {project.warnings.length > 0 && <section className="inspector-section"><h3>Capture & model notes</h3>{project.warnings.map((warning, index) => <p key={index} className="inspector-copy warning-copy">{warning}</p>)}</section>}
    {project.diagnostics && <details className="inspector-section"><summary>Capture analysis</summary><pre className="job-terminal">{JSON.stringify(project.diagnostics, null, 2)}</pre></details>}
  </>;
}
