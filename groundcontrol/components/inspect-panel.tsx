"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import {
  bearingName, DEFECT_TYPES, downloadFile, fmtDeg, inspectApi, pinMesh, SEVERITY_COLOR, STATUSES,
  type Annotation, type ArchiveRecord, type CrackResult, type DefectType, type FrameTrace, type M3c2Report, type Register,
  type SectionReport, type TerrainLayer, type TerrainResult, type TiltReport, type Vec3, type WireReport,
} from "@/lib/inspect";
import { opsApi } from "@/lib/ops";
import type { ExportFile, PlanMesh } from "@/lib/plan";
import type { Point } from "@/lib/workspace";
import { NumberField, Select } from "./plan-panel";
import { Icon, type IconName } from "./studio-icons";
import "./plan-panel.css";
import "./ops-panel.css";

export type InspectTool = "trace" | "tilt" | "wire" | "section" | "region";
export const INSPECT_TOOLS: Record<InspectTool, { hint: string; limit: number; min: number }> = {
  trace: { hint: "Click the defect or feature on the scan: the frames that saw it open below.", limit: 1, min: 1 },
  tilt: { hint: "Click the foot of the pole, mast or tower.", limit: 1, min: 1 },
  wire: { hint: "Click the two attachment points of the conductor (pole tops or insulators).", limit: 2, min: 2 },
  section: { hint: "Click the two ends of the section line.", limit: 2, min: 2 },
  region: { hint: "Outline the area to compare (three or more points), then Finish.", limit: 64, min: 3 },
};
type Mode = "inspection" | "archaeology";
type Props = {
  scene: string; ready: boolean; tool: InspectTool | null; points: Point[];
  onTool: (tool: InspectTool | null) => void; onUndoPoint: () => void; onOverlay: (meshes: PlanMesh[]) => void; onPlan: () => void;
};
const TYPE_LABEL = (t: string) => t.replace(/_/g, " ");
const SEV_LABEL = ["", "Note", "Minor", "Moderate", "Serious", "Urgent"];

function FrameCard({ frame }: { frame: FrameTrace["frames"][number] }) {
  return <figure className="insp-frame">
    {frame.url ? <span className="insp-frame-img">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={frame.url} alt={`Frame ${frame.name}`} loading="lazy" />
      <i style={{ left: `${(100 * frame.u) / frame.width}%`, top: `${(100 * frame.v) / frame.height}%` }} />
    </span> : <span className="insp-frame-missing">image not on disk</span>}
    <figcaption>{frame.name}<small>{frame.distance_m.toFixed(1)} m · {frame.gsd_mm.toFixed(1)} mm/px{frame.t_sec !== null ? ` · ${frame.t_sec.toFixed(1)} s` : ""}</small></figcaption>
  </figure>;
}

export function InspectPanel({ scene, ready, tool, points, onTool, onUndoPoint, onOverlay, onPlan }: Props) {
  const [mode, setMode] = useState<Mode>("inspection");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [register, setRegister] = useState<Register | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [trace, setTrace] = useState<{ at: Vec3; result: FrameTrace } | null>(null);
  const [draft, setDraft] = useState<{ type: DefectType; severity: number; title: string; note: string; element: string }>({ type: "crack", severity: 3, title: "", note: "", element: "" });
  const [crack, setCrack] = useState<{ id: string; result: CrackResult } | null>(null);
  const [tilt, setTilt] = useState<{ report: TiltReport; overlay: PlanMesh[] } | null>(null);
  const [radius, setRadius] = useState(0.8);
  const [wire, setWire] = useState<{ report: WireReport; overlay: PlanMesh[] } | null>(null);
  const [limit, setLimit] = useState<number | undefined>(5);
  const [section, setSection] = useState<{ report: SectionReport; overlay: PlanMesh[]; svg: ExportFile; dxf: ExportFile } | null>(null);
  const [slab, setSlab] = useState(0.25);
  const [terrainLayer, setTerrainLayer] = useState<TerrainLayer>("lrm");
  const [terrain, setTerrain] = useState<TerrainResult | null>(null);
  const [drape, setDrape] = useState(true);
  const [epochs, setEpochs] = useState<{ scene: string; name: string; compatible: boolean }[]>([]);
  const [before, setBefore] = useState("");
  const [region, setRegion] = useState<[number, number][] | null>(null);
  const [m3c2, setM3c2] = useState<{ report: M3c2Report; overlay: PlanMesh[] } | null>(null);
  const [record, setRecord] = useState<ArchiveRecord | null>(null);
  const [fixity, setFixity] = useState<string | null>(null);
  const [layers, setLayers] = useState<Record<string, boolean>>({ pins: true, tilt: true, wire: true, section: true, terrain: true, m3c2: true });

  const fail = (cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause));
  const loadRegister = useCallback(() => inspectApi.register(scene).then(setRegister).catch(fail), [scene]);
  useEffect(() => {
    const timer = setTimeout(() => {
      void loadRegister();
      opsApi.summary(scene).then((s) => { setEpochs(s.epochs); setBefore((b) => b || s.epochs.find((e) => e.compatible && e.same_site)?.scene || ""); }).catch(() => {});
    }, 0);
    return () => clearTimeout(timer);
  }, [scene, loadRegister]);

  // One overlay list for the viewer: annotation pins by severity + whatever analyses are shown.
  const overlay = useMemo(() => {
    const out: PlanMesh[] = [];
    if (layers.pins && register) for (const sev of [1, 2, 3, 4, 5]) {
      const pts = register.items.filter((a) => a.severity === sev && a.status !== "closed").map((a) => a.position);
      const m = pinMesh(pts, SEVERITY_COLOR[sev], `defect_${sev}`, 2.5);
      if (m) out.push(m);
    }
    if (layers.tilt && tilt) out.push(...tilt.overlay);
    if (layers.wire && wire) out.push(...wire.overlay);
    if (layers.section && section) out.push(...section.overlay);
    if (layers.terrain && drape && terrain) out.push(...terrain.overlay);
    if (layers.m3c2 && m3c2) out.push(...m3c2.overlay);
    return out;
  }, [layers, register, tilt, wire, section, terrain, drape, m3c2]);
  useEffect(() => { onOverlay(overlay); }, [overlay, onOverlay]);

  const run = async (key: string, fn: () => Promise<void>) => { setBusy(key); setError(""); try { await fn(); } catch (cause) { fail(cause); } finally { setBusy(null); } };
  const finish = useCallback(async () => {
    if (!tool || points.length < INSPECT_TOOLS[tool].min) return;
    const p = points.map((q) => [q[0], q[1], q[2]] as Vec3);
    const current = tool;
    onTool(null);
    if (current === "trace") await run("trace", async () => setTrace({ at: p[0], result: await inspectApi.frames(scene, p[0]) }));
    else if (current === "tilt") await run("tilt", async () => setTilt(await inspectApi.tilt(scene, p[0], radius)));
    else if (current === "wire") await run("wire", async () => setWire(await inspectApi.wire(scene, p[0], p[1], limit ?? null)));
    else if (current === "section") await run("section", async () => setSection(await inspectApi.section(scene, p[0], p[1], slab, "Section")));
    else if (current === "region") setRegion(p.map((q) => [q[0], q[2]] as [number, number]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, points, scene, radius, limit, slab, onTool]);
  useEffect(() => {
    if (!tool || tool === "region" || INSPECT_TOOLS[tool].limit !== points.length) return;
    const timer = setTimeout(() => void finish(), 0);
    return () => clearTimeout(timer);
  }, [tool, points, finish]);

  const addToRegister = () => trace && register && run("add", async () => {
    const out = await inspectApi.add(scene, register.revision, { position: trace.at, ...draft, title: draft.title || undefined });
    setRegister(out.register); setSelected(out.created); setTrace(null); setDraft((d) => ({ ...d, title: "", note: "" }));
  });
  const update = (a: Annotation, patch: Partial<Annotation>) => register && run("update", async () => setRegister((await inspectApi.update(scene, register.revision, a.id, patch)).register));
  const remove = (a: Annotation) => register && run("delete", async () => { setRegister((await inspectApi.remove(scene, register.revision, a.id)).register); setSelected(null); });

  const toolButton = (t: InspectTool, label: string, icon: IconName) =>
    <button className={`plan-tool${tool === t ? " active" : ""}`} aria-pressed={tool === t} disabled={!ready || !!busy} onClick={() => onTool(tool === t ? null : t)}><Icon name={icon} size={16} /><span>{label}</span></button>;
  const drawing = (owned: InspectTool[]) => tool && owned.includes(tool) ? <div className="plan-drawing" role="status"><p>{INSPECT_TOOLS[tool].hint}</p>
    <div className="plan-drawing-actions"><span>{points.length} point{points.length === 1 ? "" : "s"}</span>
      {tool === "region" && <><button className="text-button" disabled={!points.length} onClick={onUndoPoint}>Undo point</button>
        <button className="button primary small" disabled={points.length < 3} onClick={() => void finish()}><Icon name="check" size={13} />Finish</button></>}
      <button className="button secondary small" onClick={() => onTool(null)}>Cancel</button></div></div> : null;
  const toggle = (key: string) => <label className="plan-switch"><input type="checkbox" checked={layers[key]} onChange={(e) => setLayers((l) => ({ ...l, [key]: e.target.checked }))} />On scan</label>;
  const items = register?.items ?? [];
  const current = items.find((a) => a.id === selected) ?? null;

  const traceSection = <section className="inspector-section">
    <div className="section-label"><span>{mode === "inspection" ? "TRACE A DEFECT" : "TRACE A FEATURE"}</span></div>
    <div className="plan-tools ops-tools">{toolButton("trace", "Pick on scan", "search")}</div>
    {drawing(["trace"])}
    {busy === "trace" && <p className="plan-note">Finding the frames that saw it…</p>}
    {trace && <>
      <p className="plan-note">Seen by {trace.result.seen_by} of {trace.result.cameras} frames{trace.result.where ? ` · ${trace.result.where.mgrs}` : ""}. The best:</p>
      <div className="insp-frames">{trace.result.frames.slice(0, 4).map((f) => <FrameCard key={f.name} frame={f} />)}</div>
      {!trace.result.frames.length && <p className="plan-message">No recovered camera saw this point clearly; it may be occluded or outside every frame.</p>}
      <div className="plan-form">
        <Select label="Type" value={draft.type} options={DEFECT_TYPES.map((t) => [t, TYPE_LABEL(t)] as [DefectType, string])} onChange={(type) => setDraft({ ...draft, type })} />
        <Select label="Severity" value={String(draft.severity)} options={[1, 2, 3, 4, 5].map((s) => [String(s), `${s} · ${SEV_LABEL[s]}`] as [string, string])} onChange={(v) => setDraft({ ...draft, severity: Number(v) })} />
        <label className="plan-field plan-field-wide"><span>Title</span><input value={draft.title} maxLength={120} placeholder={TYPE_LABEL(draft.type)} onChange={(e) => setDraft({ ...draft, title: e.target.value })} /></label>
        <label className="plan-field plan-field-wide"><span>Element</span><input value={draft.element} maxLength={120} placeholder={mode === "inspection" ? "Pier 2, south face" : "East wall, 2nd course"} onChange={(e) => setDraft({ ...draft, element: e.target.value })} /></label>
        <label className="plan-field plan-field-wide"><span>Note</span><textarea value={draft.note} maxLength={4000} rows={2} onChange={(e) => setDraft({ ...draft, note: e.target.value })} /></label>
      </div>
      <button className="button primary small full" disabled={busy === "add" || !register} onClick={addToRegister}><Icon name="plus" size={13} />Add to {mode === "inspection" ? "defect register" : "annotations"}</button>
    </>}
  </section>;

  const registerSection = <section className="inspector-section">
    <div className="plan-shadow-head"><div className="section-label"><span>{mode === "inspection" ? "DEFECT REGISTER" : "ANNOTATIONS"}</span><span>{items.length}</span></div>{toggle("pins")}</div>
    {items.length ? <ul className="plan-existing insp-register">{[...items].sort((a, b) => b.severity - a.severity).map((a) => <li key={a.id} className={selected === a.id ? "active" : ""}>
      <button className="text-button insp-item" onClick={() => setSelected(selected === a.id ? null : a.id)}>
        {a.photo?.url && <Image unoptimized width={44} height={44} src={a.photo.url} alt="" />}
        <span><strong><i className="insp-sev" style={{ background: `rgb(${SEVERITY_COLOR[a.severity].join(",")})` }} />{a.title}</strong>
          <small>{TYPE_LABEL(a.type)} · severity {a.severity} · {a.status.replace("_", " ")}{a.element ? ` · ${a.element}` : ""}</small></span></button></li>)}</ul>
      : <p className="plan-note">Nothing recorded yet. Trace a point, then add it.</p>}
    {current && <div className="insp-detail">
      {current.photo?.url && <span className="insp-frame-img">{/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={crack?.id === current.id ? `data:image/jpeg;base64,${crack.result.overlay_jpeg_base64}` : current.photo.url} alt={`Photo of ${current.title}`} /></span>}
      <div className="plan-shadow-grid">
        <Select label="Status" value={current.status} options={STATUSES.map((s) => [s, s.replace("_", " ")] as [typeof s, string])} onChange={(status) => update(current, { status })} />
        <Select label="Severity" value={String(current.severity)} options={[1, 2, 3, 4, 5].map((s) => [String(s), `${s} · ${SEV_LABEL[s]}`] as [string, string])} onChange={(v) => update(current, { severity: Number(v) })} />
      </div>
      <p className="plan-note">{current.note || "No note."} Seen by {current.seen_by} frames{current.frames[0] ? `; photo from ${current.frames[0].name} at ${current.frames[0].distance_m} m (${current.frames[0].gsd_mm} mm/px)` : ""}.</p>
      {Object.entries(current.measurements).map(([k, v]) => <div className="datum-row" key={k}><span>{k.replace(/_/g, " ")}</span><span>{v}</span></div>)}
      <div className="plan-feature-actions">
        {current.photo && <button className="button secondary small" disabled={busy === "crack"} onClick={() => run("crack", async () => setCrack({ id: current.id, result: await inspectApi.crack(scene, current.id) }))}><Icon name="search" size={13} />{busy === "crack" ? "Looking…" : "Find cracks"}</button>}
        <button className="button danger small" disabled={!!busy} onClick={() => remove(current)}><Icon name="trash" size={13} />Delete</button>
      </div>
      {crack?.id === current.id && <>
        {crack.result.candidates.length ? <ul className="plan-existing">{crack.result.candidates.slice(0, 5).map((c, i) => <li key={i}><span><strong>{c.length_mm !== undefined ? `≈ ${c.length_mm.toFixed(0)} mm long, ${c.width_mm?.toFixed(1)} mm wide` : `${c.length_px.toFixed(0)} px`}</strong><small>elongation {c.elongation}</small></span>
          {c.length_mm !== undefined && <button className="button secondary small" onClick={() => update(current, { measurements: { ...current.measurements, crack_length_mm: Math.round(c.length_mm!), crack_width_mm: Math.round((c.width_mm ?? 0) * 10) / 10 } })}>Record</button>}</li>)}</ul>
          : <p className="plan-note">No thin dark lines found in this photo.</p>}
        <p className="plan-note">{crack.result.basis}. Scale from the frame: {crack.result.gsd_mm.toFixed(2)} mm per pixel at that range.</p>
      </>}
    </div>}
    <button className="button secondary small full" disabled={!items.length || busy === "report"} onClick={() => run("report", async () => downloadFile(await inspectApi.report(scene), scene))}><Icon name="download" size={13} />{mode === "inspection" ? "Inspection report (PDF)" : "Annotation register (PDF)"}</button>
  </section>;

  const compareSection = <section className="inspector-section">
    <div className="plan-shadow-head"><div className="section-label"><span>{mode === "inspection" ? "DEFORMATION VS BASELINE" : "SEASONAL COMPARISON"}</span></div>{toggle("m3c2")}</div>
    <Select label="Earlier scan" value={before} options={[["", epochs.length ? "Choose a scan" : "No other scan"], ...epochs.map((e) => [e.scene, `${e.name}${e.compatible ? "" : " · no common frame"}`] as [string, string])]} onChange={setBefore} />
    <div className="plan-tools ops-tools">{toolButton("region", region ? "Area set" : "Limit to area", "ruler")}
      {region && <button className="plan-tool" onClick={() => setRegion(null)}><Icon name="close" size={16} /><span>Whole scan</span></button>}</div>
    {drawing(["region"])}
    <button className="button primary small full" disabled={!before || !!busy} onClick={() => run("m3c2", async () => setM3c2(await inspectApi.m3c2(scene, before, region)))}><Icon name="activity" size={13} />{busy === "m3c2" ? "Measuring along normals…" : "Compare surfaces (M3C2)"}</button>
    {m3c2 && <>
      <div className="mission-kpis ops-kpis"><div className="warn"><b>{m3c2.report.moved_towards_viewer}</b><small>moved out</small></div><div className="cool"><b>{m3c2.report.moved_away}</b><small>moved in</small></div>
        <div><b>{m3c2.report.median_lod95_m !== null ? `${(m3c2.report.median_lod95_m * 1000).toFixed(0)} mm` : "—"}</b><small>LoD95</small></div><div><b>{m3c2.report.observed}</b><small>of {m3c2.report.core_points} seen</small></div></div>
      <p className="plan-note">Largest significant change {m3c2.report.max_abs_significant_m !== null ? `${(m3c2.report.max_abs_significant_m * 1000).toFixed(0)} mm` : "—"}; cylinder {m3c2.report.parameters.projection_diameter_m} m, normals over {m3c2.report.parameters.normal_scale_m} m (point spacing {m3c2.report.point_spacing_m} m). {m3c2.report.basis}. {m3c2.report.note}.</p>
    </>}
  </section>;

  const inspection = <>
    {traceSection}
    {registerSection}
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>POLE / TOWER TILT</span></div>{toggle("tilt")}</div>
      <div className="plan-shadow-grid"><NumberField label="Search radius" unit="m" value={radius} min={0.1} max={10} step={0.1} onCommit={(v) => setRadius(v ?? 0.8)} /></div>
      <div className="plan-tools ops-tools">{toolButton("tilt", "Pick the foot", "pin")}</div>
      {drawing(["tilt"])}
      {tilt && <><div className="mission-kpis ops-kpis"><div className={tilt.report.tilt_deg > 1 ? "warn" : "good"}><b>{fmtDeg(tilt.report.tilt_deg)}</b><small>from vertical</small></div>
        <div><b>{tilt.report.lean_mm_per_m.toFixed(0)}</b><small>mm per m</small></div><div><b>{bearingName(tilt.report.lean_bearing_deg)}</b><small>leans to</small></div><div><b>{tilt.report.height_m.toFixed(1)} m</b><small>fitted height</small></div></div>
        <p className="plan-note">Top is {(tilt.report.top_offset_m * 100).toFixed(0)} cm off plumb; {tilt.report.points} points, {(tilt.report.radius_rms_m * 100).toFixed(0)} cm scatter about the axis. {tilt.report.basis}</p></>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>CONDUCTOR SAG AND CLEARANCE</span></div>{toggle("wire")}</div>
      <div className="plan-shadow-grid"><NumberField label="Required clearance" unit="m" value={limit} min={0} max={100} step={0.5} onCommit={setLimit} /></div>
      <div className="plan-tools ops-tools">{toolButton("wire", "Pick the two ends", "arrow")}</div>
      {drawing(["wire"])}
      {wire && <><div className="mission-kpis ops-kpis"><div><b>{wire.report.span_m.toFixed(1)} m</b><small>span</small></div><div><b>{wire.report.sag_m.toFixed(2)} m</b><small>sag ({wire.report.sag_pct_of_span}%)</small></div>
        <div className={wire.report.clearance_ok === false ? "bad" : "good"}><b>{wire.report.min_clearance_m !== null ? `${wire.report.min_clearance_m.toFixed(1)} m` : "—"}</b><small>min clearance</small></div><div><b>{wire.report.points}</b><small>points</small></div></div>
        {wire.report.clearance_ok === false && <p className="plan-message" role="alert">Clearance below {wire.report.clearance_limit_m} m at {wire.report.clearance_at_m} m along the span.</p>}
        <p className="plan-note">Catenary fit RMS {(wire.report.fit_rms_m * 100).toFixed(0)} cm. {wire.report.basis}.</p></>}
    </section>
    {compareSection}
    <section className="inspector-section">
      <div className="section-label"><span>MEASURE AND ACCESS</span></div>
      <p className="plan-note">Distances, areas and point-to-plane measurements with their uncertainty are in the Measure tab. Scaffold bays, scissor and boom lifts for repair access are in the Plan tab&apos;s catalogue.</p>
      <button className="button secondary small full" onClick={onPlan}><Icon name="grid" size={13} />Plan access in the Plan tab</button>
    </section>
  </>;

  const archaeology = <>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>TERRAIN VISUALISATION</span></div>
        {terrain?.layer === "lrm" && <label className="plan-switch"><input type="checkbox" checked={drape} onChange={(e) => setDrape(e.target.checked)} />Drape on scan</label>}</div>
      <div className="plan-view" role="group" aria-label="Terrain layer">{([["ortho", "Ortho"], ["hillshade_dtm", "Ground relief"], ["hillshade_dsm", "Surface"], ["lrm", "Local relief"], ["slope_deg", "Slope"]] as [TerrainLayer, string][]).map(([k, label]) =>
        <button key={k} className={terrainLayer === k ? "active" : ""} aria-pressed={terrainLayer === k} onClick={() => { setTerrainLayer(k); void run("terrain", async () => setTerrain(await inspectApi.terrain(scene, k))); }}>{label}</button>)}</div>
      {busy === "terrain" && <p className="plan-note">Rendering…</p>}
      {terrain && <>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="ops-firstmap insp-raster" src={`data:image/png;base64,${terrain.png_base64}`} alt={`${terrain.layer} of the site`} />
        {terrain.layer === "lrm" && <div className="plan-legend"><span><i style={{ background: "rgb(30,70,200)" }} />Hollow (ditch, pit)</span><span><i style={{ background: "rgb(210,40,40)" }} />Raised (bank, wall line)</span>
          <span>{(terrain.lrm_range_m[0] * 100).toFixed(0)} to {(terrain.lrm_range_m[1] * 100).toFixed(0)} cm</span></div>}
        <p className="plan-note">{terrain.basis}. North is up the image only for a scene aligned to north; transparent = never observed.</p>
      </>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>SECTION</span></div>{toggle("section")}</div>
      <div className="plan-shadow-grid"><NumberField label="Slab half-width" unit="m" value={slab} min={0.02} max={5} step={0.05} onCommit={(v) => setSlab(v ?? 0.25)} /></div>
      <div className="plan-tools ops-tools">{toolButton("section", "Draw section line", "ruler")}</div>
      {drawing(["section"])}
      {section && <>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="insp-section" src={`data:image/svg+xml;base64,${section.svg.content}`} alt="Section drawing" />
        <p className="plan-note">{section.report.length_m.toFixed(2)} m at {section.report.azimuth_deg.toFixed(0)}° (scene), {section.report.points.toLocaleString()} points, heights {section.report.y_range[0].toFixed(2)}–{section.report.y_range[1].toFixed(2)} m{section.report.observed_pct !== null ? `; ground observed along ${section.report.observed_pct.toFixed(0)}%` : ""}.</p>
        <div className="plan-feature-actions"><button className="button secondary small" onClick={() => downloadFile(section.svg, scene)}><Icon name="download" size={13} />SVG</button>
          <button className="button secondary small" onClick={() => downloadFile(section.dxf, scene)}><Icon name="download" size={13} />DXF</button></div>
      </>}
    </section>
    {traceSection}
    {registerSection}
    {compareSection}
    <section className="inspector-section">
      <div className="section-label"><span>ARCHIVE RECORD</span></div>
      <button className="button secondary small full" disabled={busy === "record"} onClick={() => run("record", async () => { setRecord(await inspectApi.provenance(scene)); setFixity(null); })}><Icon name="file" size={13} />{busy === "record" ? "Hashing files…" : "Build paradata record"}</button>
      {record && <>
        {(["title", "date", "creator", "coverage"] as const).map((k) => <div className="datum-row" key={k}><span>{k}</span><span>{String(record.dublin_core[k] ?? "not recorded")}</span></div>)}
        <div className="datum-row"><span>Reference</span><span>{String((record.spatial_reference as { status?: string }).status ?? "—")}</span></div>
        <div className="datum-row"><span>Fixity</span><span>{record.fixity.length} files, SHA-256</span></div>
        <div className="plan-feature-actions">
          <button className="button secondary small" onClick={() => downloadFile({ filename: "archive-record.json", media_type: "application/json", encoding: "utf-8", content: JSON.stringify(record, null, 2) }, scene)}><Icon name="download" size={13} />JSON</button>
          <button className="button secondary small" onClick={() => run("verify", async () => { const r = await inspectApi.verify(scene, record); setFixity(r.changed.length ? `${r.changed.length} file(s) changed: ${r.changed.map((c) => c.file).join(", ")}` : "All files match the record."); })}><Icon name="check" size={13} />Check fixity</button>
        </div>
        {fixity && <p className="plan-note">{fixity}</p>}
        <p className="plan-note">{record.notes.join(" ")}</p>
      </>}
    </section>
    <section className="inspector-section">
      <div className="section-label"><span>HYPOTHESIS RECONSTRUCTION</span></div>
      <p className="plan-note">Draw a proposed restoration (a missing tower, a fallen wall) as a scheme in the Plan tab and mark it <b>Hypothesis</b>: it is shown and exported as inferred, never as measured, and can be switched on and off for public presentation.</p>
      <button className="button secondary small full" onClick={onPlan}><Icon name="grid" size={13} />Open the Plan tab</button>
    </section>
  </>;

  return <div className="plan-panel ops-panel inspect-panel">
    <section className="inspector-section plan-head">
      <div className="section-label"><span>INSPECT</span></div>
      <div className="plan-view" role="group" aria-label="Discipline">{(["inspection", "archaeology"] as Mode[]).map((m) => <button key={m} className={mode === m ? "active" : ""} aria-pressed={mode === m} onClick={() => { setMode(m); onTool(null); }}>{m === "inspection" ? "Inspection" : "Archaeology"}</button>)}</div>
      <p className="inspector-copy">{mode === "inspection" ? "Trace a defect to the frames that saw it, record it with a photo, measure tilt, sag and clearance, compare with a baseline scan, and issue the report." : "Relief and local relief models, true-scale sections, annotations with their frames, seasonal change, and the archive record of how the model was made."}</p>
      {error && <p className="plan-message" role="alert">{error}</p>}
    </section>
    {mode === "inspection" ? inspection : archaeology}
  </div>;
}
