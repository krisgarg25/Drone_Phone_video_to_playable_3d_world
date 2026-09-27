"use client";

import { useEffect, useState } from "react";
import { act, advance, surveyStatus, type SurveyState, type SurveyStep } from "@/lib/api";
import { Icon } from "./studio-icons";

const CHIP: Record<SurveyStep["status"], string> = {
  done: "Done", ready: "Ready", needs_you: "Needs you", running: "Running",
  failed: "Failed", optional: "Optional", waiting: "Waiting",
};

/** A blank sheet in the most common operator form: surveyed degrees against the UTM product. */
const TEMPLATE = "id,ref_lat_deg,ref_lon_deg,ref_height_m,model_easting_m,model_northing_m,model_height_m\n";

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  URL.revokeObjectURL(url);
}

/**
 * The georeferenced survey path as one guided list.
 *
 * Every row comes from the server's `steps`, derived from the same validated status the
 * rest of the console reads, so this panel cannot call a step done that the backend
 * would call stale. "Run next steps" performs every CPU step that can run now and stops
 * at the first one only the operator can unblock - the GPU reconstruction is always one
 * of those, and is started from the Run dialog with its own confirmation.
 */
export function SurveyControls({ scene, onChange }: { scene: string; onChange: () => void }) {
  const [state, setState] = useState<SurveyState | null>(null);
  const [csv, setCsv] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<File | null>(null);
  const [points, setPoints] = useState<File | null>(null);
  const [refDatum, setRefDatum] = useState("ellipsoidal");
  const [modelDatum, setModelDatum] = useState("ellipsoidal");
  const [withheld, setWithheld] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<{ stage: string; status: string; detail?: string }[]>([]);

  useEffect(() => {
    let live = true;
    surveyStatus(scene).then((value) => { if (live) setState(value); },
      (cause) => { if (live) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { live = false; };
  }, [scene]);

  async function perform(task: () => Promise<SurveyState>) {
    setBusy(true); setError(""); setTrace([]);
    try { setState(await task()); onChange(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  const saveInputs = () => perform(async () => {
    if (!csv || !metadata) throw new Error("Select telemetry CSV and metadata JSON.");
    if (csv.size + metadata.size > 1900000) throw new Error("Survey input files must total less than 1.9 MB.");
    return act("inputs", { scene, telemetry_csv: await csv.text(), metadata: JSON.parse(await metadata.text()) });
  });
  const runNext = () => perform(async () => {
    const result = await advance(scene);
    setTrace(result.trace);
    return result.state;
  });
  const uploadPoints = () => perform(async () => {
    if (!points) throw new Error("Select a checkpoint CSV.");
    if (points.size > 1900000) throw new Error("The checkpoint CSV must be below 1.9 MB.");
    return act("checkpoints", { scene, csv: await points.text(), reference_height_datum: refDatum,
                                model_height_datum: modelDatum, withheld });
  });

  const steps = state?.steps ?? [];
  const step = (id: SurveyStep["id"]) => steps.find((row) => row.id === id);
  const runnable = steps.some((row) => row.status === "ready");
  const accuracy = state?.evaluation?.criteria?.find((c) => c.id === "accuracy");
  const rmse = accuracy?.status === "measured" ? (accuracy.metrics as { rmse_3d_m?: number; count?: number } | undefined) : undefined;
  const alignedOrLater = step("checkpoints")?.status === "optional" || step("checkpoints")?.status === "done";

  return <details className="inspector-section survey-guide" open={Boolean(state && state.status !== "not_prepared")}>
    <summary>Georeferenced survey</summary>
    <p className="inspector-copy">Optional. The visual model works without GPS. Survey exports need timestamped telemetry; accuracy needs surveyed checkpoints.</p>
    <ol className="survey-steps" aria-label="Survey steps">
      {steps.map((row) => <li key={row.id} className={`survey-step is-${row.status}`}>
        <span className="survey-step-chip">{CHIP[row.status] ?? row.status}</span>
        <div><strong>{row.label}</strong><small>{row.detail}</small></div>
      </li>)}
    </ol>
    {!state && !error && <p className="inspector-copy" role="status">Reading survey status…</p>}

    {step("inputs")?.status === "needs_you" && <div className="survey-form">
      <label className="field">Telemetry CSV, GPX or DJI .srt<input type="file" accept=".csv,.gpx,.srt,.txt" onChange={(event) => setCsv(event.target.files?.[0] ?? null)} disabled={busy} /></label>
      <label className="field">Flight metadata JSON<input type="file" accept=".json" onChange={(event) => setMetadata(event.target.files?.[0] ?? null)} disabled={busy} /></label>
      <button className="button secondary small full" disabled={!csv || !metadata || busy} onClick={() => void saveInputs()}><Icon name="upload" size={14} />Save survey inputs</button>
    </div>}

    <button className="button primary small full" disabled={busy || !runnable} onClick={() => void runNext()}>
      <Icon name="play" size={14} />{busy ? "Working…" : runnable ? "Run next steps" : "Nothing to run on this machine now"}
    </button>
    {step("reconstruct")?.status === "needs_you" && <p className="inspector-copy">Next: open <b>Run</b>, choose <b>Survey surface &amp; exports</b> and approve the GPU reconstruction. Come back here afterwards to place it on the map.</p>}
    {trace.length > 0 && <ul className="survey-trace">{trace.map((row, index) => <li key={index}><b>{row.stage}</b> · {row.status.replaceAll("_", " ")}{row.detail ? ` — ${row.detail}` : ""}</li>)}</ul>}

    {alignedOrLater && <div className="survey-form">
      <div className="section-label">SURVEYED CHECKPOINTS</div>
      {rmse?.rmse_3d_m !== undefined && <div className="datum-row"><span>Measured 3D RMSE</span><span>{rmse.rmse_3d_m.toFixed(3)} m · {rmse.count} hold-out point{rmse.count === 1 ? "" : "s"}</span></div>}
      <p className="inspector-copy">One row per point: its surveyed position and the same feature read off the delivered model. Degrees, the scene&apos;s UTM zone or local ENU are all accepted.</p>
      <label className="field">Checkpoint CSV<input type="file" accept=".csv,.txt" onChange={(event) => setPoints(event.target.files?.[0] ?? null)} disabled={busy} /></label>
      <div className="field-row">
        <label className="field">Surveyed heights<select value={refDatum} onChange={(event) => setRefDatum(event.target.value)} disabled={busy}><option value="ellipsoidal">Ellipsoidal (GNSS)</option><option value="egm96" disabled={!state?.geoid?.available}>Mean sea level (EGM96)</option></select></label>
        <label className="field">Model heights<select value={modelDatum} onChange={(event) => setModelDatum(event.target.value)} disabled={busy}><option value="ellipsoidal">Ellipsoidal product</option><option value="egm96" disabled={!state?.geoid?.available}>MSL product (EGM96)</option></select></label>
      </div>
      <label className="run-confirm"><input type="checkbox" checked={withheld} onChange={(event) => setWithheld(event.target.checked)} disabled={busy} /><span>None of these points was used to build or align the model. Without this, the accuracy stays labelled as unverified independence.</span></label>
      <div className="export-buttons">
        <button className="button secondary small full" disabled={!points || busy} onClick={() => void uploadPoints()}><Icon name="upload" size={14} />Upload and evaluate</button>
        <button className="button secondary small full" onClick={() => download(`${scene}-checkpoints-template.csv`, TEMPLATE)}><Icon name="download" size={14} />Template CSV</button>
      </div>
    </div>}
    {error && <p className="form-error" role="alert">{error}</p>}
  </details>;
}
