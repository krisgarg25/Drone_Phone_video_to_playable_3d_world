"use client";

// E: the quality screen. What this world was built as, what it actually became, and
// which of the gate's checks it passed - as numbers, not as a coloured step tile.
//
// Every row here existed before this component and none of it reached an operator.
// `check_world.py` printed "heightfield coverage: 19% measured (threshold 5%)" to a
// console and wrote a sentence into `viewer_assets/world_check.json` that no caller in
// the repository ever opened; a scene the gate BLOCKED arrived as a red tile labelled
// "gate" with no way to find out which rule it broke, and a scene that shipped with four
// quality warnings looked identical to a clean pass because a soft failure exits 0.
//
// The shape follows what shipping photogrammetry products settled on: a short list of
// numeric rows, each with the threshold it was judged against and the direction that is
// better (PIX4D's Quality Check, RealityCapture's alignment table, DroneDeploy's
// processing report). Blocked first, then warnings, then the passes collapsed - an
// operator reads this to find out what to re-shoot, not to be reassured.

import { Icon, type IconName } from "./studio-icons";
import type { ProjectDetail, QualityCheckRow } from "@/lib/workspace";

const VERDICT: Record<string, { label: string; tone: string; note: string }> = {
  failed: { label: "Blocked", tone: "bad", note: "The world gate stopped this scene. It loads and walks, but a hard check failed." },
  warnings: { label: "Usable, with limits", tone: "warn", note: "Every blocking check passed. These are quality warnings, not defects." },
  pass: { label: "Passed", tone: "good", note: "All checks passed at the scale this scene was built for." },
};

const READINGS: { key: string; label: string; unit: string; help: string }[] = [
  { key: "camera_agl_m", label: "Working height", unit: "m above the ground it filmed", help: "A hand holds a phone at 1-2 m; an airframe flies 10-60 m. This is the number that says what the capture actually was." },
  { key: "footprint_m", label: "Scene span", unit: "m across", help: "The walk grid's own longest side." },
  { key: "cell_m", label: "Grid resolution", unit: "m per cell", help: "The smallest thing the physics floor can represent. Above ~1.5 m it cannot see a doorway." },
  { key: "registration_pct", label: "Frames registered", unit: "%", help: "Views COLMAP placed out of views it was given. A view that did not register is a view the model cannot see." },
];

function State({ status }: { status: QualityCheckRow["status"] }) {
  const glyph: IconName = status === "pass" ? "check" : status === "fail" ? "close" : "info";
  return <span className={`quality-state state-${status}`}><Icon name={glyph} size={12} aria-hidden />{status === "na" ? "n/a" : status}</span>;
}

function Row({ row }: { row: QualityCheckRow }) {
  return <div className={`quality-row sev-${row.severity}`}>
    <State status={row.status} />
    <div className="quality-main">
      <b>{row.name}</b>
      {row.unit ? <span className="num">{row.value === null || row.value === undefined ? "—" : row.value}<small>{row.unit}</small></span>
        : <span className="num">{row.basis}</span>}
      {row.threshold ? <span className="quality-band">{row.better === "lower" ? "at most" : "at least"} {row.threshold}</span> : null}
    </div>
    {row.unit && row.basis ? <p className="quality-basis">{row.basis}</p> : null}
  </div>;
}

export function QualityPanel({ project }: { project: ProjectDetail }) {
  const quality = project.quality;
  const scenario = project.scenario;
  const verdict = quality ? VERDICT[quality.status] ?? VERDICT.pass : null;
  const rows = quality?.checks ?? [];
  const blocked = rows.filter((r) => r.status === "fail" && r.severity === "hard");
  const limited = rows.filter((r) => r.status === "fail" && r.severity !== "hard");
  const unknown = rows.filter((r) => r.status === "na");
  const okay = rows.filter((r) => r.status === "pass");
  const audit = scenario?.audit;
  const findings = audit?.findings ?? [];
  const thr = quality?.thresholds ?? {};

  return <div className="quality-panel">
    <section className="inspector-section quality-head">
      <div className="section-label"><span>WORLD QUALITY</span><Icon name={verdict?.tone === "good" ? "check" : "info"} size={14} /></div>
      {verdict ? <>
        <p className={`quality-verdict tone-${verdict.tone}`}><b>{verdict.label}</b><span>{verdict.note}</span></p>
        <div className="datum-row"><span>Built as</span><span>{scenario ? `${scenario.label || scenario.preset}${scenario.decided_by ? ` · ${scenario.decided_by}` : ""}` : "Not recorded - this run predates the scenario record"}</span></div>
        <div className="datum-row"><span>Walk body</span><span title={thr.character_height_source || ""}>{thr.character_height_m ? `${thr.character_height_m} m character` : "Unknown"}{thr.min_headroom_m ? ` · needs ${thr.min_headroom_m} m headroom` : ""}</span></div>
        <div className="datum-row"><span>Grid</span><span>{thr.grid ? `${thr.grid.join(" × ")} cells · ` : ""}{typeof thr.cell_m === "number" ? `${thr.cell_m.toFixed(3)} m each` : "cell size unknown"}</span></div>
        <div className="datum-row"><span>Scale</span><span>{project.scale.status === "metric" ? "Metric from " : project.scale.status === "estimated" ? "Estimated from " : "Relative — "}{project.scale.source}</span></div>
        {project.scale_check ? <div className="datum-row" title="An independent learned ruler, advisory only: on the one scene anyone measured with a tape it read 18% low while the AR scale read 4% low. It never changes a number above it."><span>Second ruler</span><span>{project.scale_check.ruler_m_per_unit} m/unit · {project.scale_check.gap_percent}% apart · does not change your measurements</span></div> : null}
      </> : <p className="empty-state">Nothing has been checked yet. The gate runs at the end of a reconstruction, after the collider is built.</p>}
    </section>

    {blocked.length ? <section className="inspector-section quality-block">
      <div className="section-label"><span>BLOCKING</span><Icon name="close" size={14} /></div>
      {blocked.map((r) => <Row key={r.name} row={r} />)}
    </section> : null}

    {limited.length ? <section className="inspector-section">
      <div className="section-label"><span>QUALITY WARNINGS</span><Icon name="info" size={14} /></div>
      {limited.map((r) => <Row key={r.name} row={r} />)}
    </section> : null}

    {unknown.length ? <section className="inspector-section">
      <div className="section-label"><span>NOT MEASURABLE HERE</span><Icon name="help" size={14} /></div>
      {unknown.map((r) => <Row key={r.name} row={r} />)}
      <p className="quality-basis">A check with no data is reported as not applicable rather than passed. A room scan that never got the ceiling in frame does not have a headroom measurement, and saying &ldquo;unknown&rdquo; is the difference between a caveat and an invention.</p>
    </section> : null}

    {okay.length ? <details className="inspector-section">
      <summary className="section-label"><span>CHECKS PASSED ({okay.length})</span><Icon name="check" size={14} /></summary>
      {okay.map((r) => <Row key={r.name} row={r} />)}
    </details> : null}

    {audit || scenario ? <section className="inspector-section">
      <div className="section-label"><span>WHAT THIS SCENE MEASURED</span><Icon name="activity" size={14} /></div>
      {READINGS.map((reading) => {
        const value = audit?.[reading.key as keyof typeof audit];
        return <div className="datum-row" key={reading.key} title={reading.help}><span>{reading.label}</span><span>{value === null || value === undefined ? "Not measured" : `${value} ${reading.unit}`}</span></div>;
      })}
      {findings.length ? <div className="quality-findings">{findings.map((f, i) => <p key={i} className={`finding tone-${f.severity === "fail" ? "bad" : f.severity === "warn" ? "warn" : "muted"}`}><b>{f.kind.replace(/-/g, " ")}</b><span>{f.message}</span>{f.fix ? <em>{f.fix}</em> : null}</p>)}</div>
        : <p className="quality-basis">No disagreement between the scenario this was built as and the world it produced.</p>}
    </section> : null}

    {scenario ? <section className="inspector-section">
      <div className="section-label"><span>CAPTURE DECISION</span><Icon name="compass" size={14} /></div>
      {scenario.evidence.length ? scenario.evidence.map((e, i) => <p className="quality-evidence" key={i}><b>{e.signal}</b>{e.value === null ? null : <span className="num">{String(e.value)}</span>}<span>{e.because}</span></p>)
        : <p className="quality-basis">The preset was chosen by hand, so no measurement decided it.</p>}
      {scenario.advice ? <p className="notice"><Icon name="camera" /><p>{scenario.advice}</p></p> : null}
      {scenario.applied.length ? <details className="quality-applied"><summary>Parameters this run applied, and who set each one</summary>
        <table><tbody>{scenario.applied.map((a) => <tr key={a.param}><td className="mono">{a.param}</td><td className="num">{a.value === null ? "default" : String(a.value)}</td><td>{a.set_by}</td></tr>)}</tbody></table>
        <p className="quality-basis">A parameter nobody chose shows as a script default rather than disappearing. Two of these - the SIFT detector thresholds - have been in the preset table since it was written and never reached COLMAP, because they were missing from the plan the mapper reads; every capture style ran the room detector until they were added.</p>
      </details> : null}
    </section> : null}
  </div>;
}
