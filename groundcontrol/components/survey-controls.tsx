"use client";

import { useState } from "react";
import { act, type SurveyState } from "@/lib/api";
import { Icon } from "./studio-icons";

export function SurveyControls({ scene, onChange }: { scene: string; onChange: () => void }) {
  const [csv, setCsv] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<File | null>(null);
  const [state, setState] = useState<SurveyState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function run(action: "inputs" | "prepare" | "align" | "evaluate") {
    setBusy(true); setError("");
    try {
      const body: Record<string, unknown> = { scene };
      if (action === "inputs") {
        if (!csv || !metadata) throw new Error("Select telemetry CSV and metadata JSON.");
        if (csv.size + metadata.size > 1900000) throw new Error("Survey input files must total less than 1.9 MB.");
        body.telemetry_csv = await csv.text(); body.metadata = JSON.parse(await metadata.text());
      }
      setState(await act(action, body)); onChange();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <details className="inspector-section"><summary>Georeferenced survey setup</summary><div className="survey-form"><p className="inspector-copy">Optional. The visual model works without GPS. Survey exports require timestamped telemetry and explicit camera-position and altitude conventions.</p><label className="field">Telemetry CSV<input type="file" accept=".csv" onChange={(event) => setCsv(event.target.files?.[0] ?? null)} disabled={busy} /></label><label className="field">Flight metadata JSON<input type="file" accept=".json" onChange={(event) => setMetadata(event.target.files?.[0] ?? null)} disabled={busy} /></label><button className="button secondary small full" disabled={!csv || !metadata || busy} onClick={() => void run("inputs")}><Icon name="upload" size={14} />Save survey inputs</button><button className="button secondary small full" disabled={busy} onClick={() => void run("prepare")}>Prepare survey</button><button className="button secondary small full" disabled={busy} onClick={() => void run("align")}>Align existing reconstruction</button><button className="button secondary small full" disabled={busy} onClick={() => void run("evaluate")}>Check available evidence</button>{busy && <p role="status" className="inspector-copy">Processing survey inputs…</p>}{state && <p className="inspector-copy">Survey: {state.status.replaceAll("_", " ")}</p>}{error && <p className="form-error" role="alert">{error}</p>}</div></details>;
}
