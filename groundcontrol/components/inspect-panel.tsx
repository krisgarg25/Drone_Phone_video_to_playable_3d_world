"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import {
  bearingName, DEFECT_TYPES, downloadFile, fmtDeg, inspectApi, pinMesh, SEVERITY_COLOR, STATUSES,
  type Annotation, type ArchiveRecord, type CrackResult, type DefectType, type FrameTrace, type M3c2Report, type Register,
  type SectionReport, type TerrainLayer, type TerrainResult, type TiltReport, type Vec3, type WireReport,
} from "@/lib/inspect";
import { opsApi } from "@/lib/ops";
import type { ExportFile, PlanMesh } from "@/lib/plan";
import type { Point } from "@/lib/workspace";
import { NumberField } from "./plan-panel";
import { Icon, type IconName } from "./studio-icons";
import { OnViewer } from "./on-viewer";
import "./plan-panel.css";
import "./ops-panel.css";
import "./inspect.css";

export type InspectTool = "trace" | "tilt" | "wire" | "section" | "region";
export const INSPECT_TOOLS: Record<InspectTool, { hint: string; limit: number; min: number }> = {
  trace: { hint: "Click the damage or feature on the model. The photos that saw it will appear here.", limit: 1, min: 1 },
  tilt: { hint: "Click where the pole, mast or tower meets the ground.", limit: 1, min: 1 },
  wire: { hint: "Click where the cable is attached at each end (pole tops or insulators).", limit: 2, min: 2 },
  section: { hint: "Click where the cut should start, then where it should end.", limit: 2, min: 2 },
  region: { hint: "Click around the area to compare (three or more points), then press Finish.", limit: 64, min: 3 },
};
type Mode = "inspection" | "archaeology";
type Props = {
  scene: string; ready: boolean; tool: InspectTool | null; points: Point[];
  onTool: (tool: InspectTool | null) => void; onUndoPoint: () => void; onOverlay: (meshes: PlanMesh[]) => void; onPlan: () => void;
  onMeasure: () => void; onFrame?: (cameraIndex: number) => void;
};
type Task = "log" | "register" | "tilt" | "wire" | "compare" | "terrain" | "section" | "record" | "restore";
/** One job at a time: the panel shows the chosen task instead of every tool in one long scroll. */
const TASKS: Record<Mode, { id: Task; label: string; icon: IconName; what: string }[]> = {
  inspection: [
    { id: "log", label: "Log a defect", icon: "pin", what: "Click damage on the model, check the photos that saw it, then save it to the list." },
    { id: "register", label: "Defect list", icon: "list", what: "Everything logged so far. Open one to change its status or look for cracks." },
    { id: "tilt", label: "Pole tilt", icon: "angle", what: "Click the foot of a pole, mast or tower to see how far it leans." },
    { id: "wire", label: "Cable sag", icon: "wave", what: "Click both ends of a cable to measure how much it sags and how high it clears the ground." },
    { id: "compare", label: "What moved", icon: "history", what: "Compare with an older scan of the same site to find surfaces that moved." },
  ],
  archaeology: [
    { id: "terrain", label: "Relief maps", icon: "mountain", what: "Shaded maps that bring out faint banks, ditches and wall lines." },
    { id: "section", label: "Cut a section", icon: "cut", what: "Click two points to draw a true-scale profile through the site." },
    { id: "log", label: "Log a feature", icon: "pin", what: "Click a feature on the model, check the photos that saw it, then save it." },
    { id: "register", label: "Feature list", icon: "list", what: "Every feature logged so far, with photos and positions." },
    { id: "compare", label: "Seasonal change", icon: "history", what: "Compare with an earlier season's scan to find erosion or movement." },
    { id: "record", label: "Archive record", icon: "seal", what: "A record of how the model was made, with a checksum for every file." },
    { id: "restore", label: "Restoration idea", icon: "building", what: "Sketch a missing wall or tower, clearly marked as a guess." },
  ],
};
const TYPE_LABEL = (t: string) => t.replace(/_/g, " ");
const SEV_LABEL = ["", "Note", "Minor", "Moderate", "Serious", "Urgent"];

export function InspectPanel({ scene, ready, tool, points, onTool, onUndoPoint, onOverlay, onPlan, onMeasure, onFrame }: Props) {
  const [mode, setMode] = useState<Mode>("inspection");
  const [task, setTask] = useState<Task>("log");
  const [statusFilter, setStatusFilter] = useState<"all" | (typeof STATUSES)[number]>("all");
  const [zoom, setZoom] = useState<{ url: string; alt: string; caption?: string; marker?: { u: number; v: number; w: number; h: number } } | null>(null);
  const zoomDialog = useRef<HTMLDialogElement>(null);
  const bodyRef = useRef<HTMLElement>(null);
  useEffect(() => { if (zoom && !zoomDialog.current?.open) zoomDialog.current?.showModal(); if (!zoom && zoomDialog.current?.open) zoomDialog.current.close(); }, [zoom]);
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

  // An opened item starts at its top, not wherever the list was scrolled.
  useEffect(() => { if (selected) bodyRef.current?.scrollIntoView({ block: "start", behavior: "smooth" }); }, [selected]);
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

  const items = register?.items ?? [];
  const current = items.find((a) => a.id === selected) ?? null;
  const tasks = TASKS[mode];
  const active = tasks.find((t) => t.id === task) ?? tasks[0];
  const shown = statusFilter === "all" ? items : items.filter((a) => a.status === statusFilter);
  const sevColor = (s: number) => `rgb(${SEVERITY_COLOR[s].join(",")})`;
  const noun = mode === "inspection" ? "defect" : "feature";

  const pickButton = (t: InspectTool, label: string, icon: IconName) =>
    <button className={`insp-pick${tool === t ? " is-armed" : ""}`} aria-pressed={tool === t} disabled={!ready || !!busy} onClick={() => onTool(tool === t ? null : t)}>
      <Icon name={tool === t ? "close" : icon} size={18} /><span>{tool === t ? "Cancel picking" : label}<small>{tool === t ? "Or press Esc" : ready ? "Then click on the 3D view" : "Waiting for the 3D view"}</small></span></button>;
  const drawing = (owned: InspectTool[]) => tool && owned.includes(tool) ? <OnViewer><div className="plan-drawing" role="status"><p>{INSPECT_TOOLS[tool].hint}</p>
    <div className="plan-drawing-actions"><span>{points.length} point{points.length === 1 ? "" : "s"}</span>
      {tool === "region" && <><button className="text-button" disabled={!points.length} onClick={onUndoPoint}>Undo point</button>
        <button className="button primary small" disabled={points.length < 3} onClick={() => void finish()}><Icon name="check" size={13} />Finish</button></>}
      <button className="button secondary small" onClick={() => onTool(null)}>Cancel</button></div></div></OnViewer> : null;
  const showToggle = (key: string) => <label className="insp-show"><input type="checkbox" checked={layers[key]} onChange={(e) => setLayers((l) => ({ ...l, [key]: e.target.checked }))} />Show on model</label>;
  const kpi = (value: string, label: string, tone = "") => <div className={tone}><b>{value}</b><small>{label}</small></div>;
  const photo = (url: string, alt: string, marker?: { u: number; v: number; w: number; h: number }, caption?: string) =>
    <button className="insp-photo" onClick={() => setZoom({ url, alt, marker, caption })} aria-label={`Enlarge ${alt}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt={alt} loading="lazy" />
      {marker && <i style={{ left: `${(100 * marker.u) / marker.w}%`, top: `${(100 * marker.v) / marker.h}%` }} />}
      <span className="insp-photo-zoom"><Icon name="zoom" size={14} /></span>
    </button>;

  // Step-by-step: click the damage, check the photos that saw it, describe it and save.
  const step = tool === "trace" ? 1 : trace ? 3 : 1;
  const logTask = <>
    <ol className="insp-steps" aria-label="Steps">
      {["Click it on the model", "Check the photos", "Describe and save"].map((label, i) => <li key={label} className={step > i + 1 || (i === 1 && trace) ? "done" : step === i + 1 ? "now" : ""}><span>{step > i + 1 || (i === 1 && trace) ? <Icon name="check" size={12} /> : i + 1}</span>{label}</li>)}
    </ol>
    {pickButton("trace", trace ? `Pick a different spot` : `Click the ${noun} on the model`, "cursor")}
    {drawing(["trace"])}
    {busy === "trace" && <p className="insp-busy"><span className="spinner" />Finding the photos that saw this spot…</p>}
    {trace && <>
      <div className="insp-seen"><Icon name="photos" size={16} /><span><b>{trace.result.seen_by}</b> of {trace.result.cameras} photos saw this spot{trace.result.where ? <> · <code>{trace.result.where.mgrs}</code></> : null}</span></div>
      {!trace.result.frames.length && <p className="plan-message">No photo saw this point clearly. It may be hidden or outside the flight.</p>}
      <div className="insp-form">
        <label className="insp-field"><span>What is it?</span><select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as DefectType })}>{DEFECT_TYPES.map((t) => <option key={t} value={t}>{TYPE_LABEL(t)}</option>)}</select></label>
        <div className="insp-field"><span>How bad?</span><div className="insp-sev-pick" role="radiogroup" aria-label="Severity">{[1, 2, 3, 4, 5].map((s) => <button key={s} role="radio" aria-checked={draft.severity === s} className={draft.severity === s ? "on" : ""} style={{ ["--sev" as string]: sevColor(s) }} onClick={() => setDraft({ ...draft, severity: s })}><i />{SEV_LABEL[s]}</button>)}</div></div>
        <label className="insp-field"><span>Title</span><input value={draft.title} maxLength={120} placeholder={TYPE_LABEL(draft.type)} onChange={(e) => setDraft({ ...draft, title: e.target.value })} /></label>
        <label className="insp-field"><span>Where on the structure</span><input value={draft.element} maxLength={120} placeholder={mode === "inspection" ? "e.g. Pier 2, south face" : "e.g. East wall, 2nd course"} onChange={(e) => setDraft({ ...draft, element: e.target.value })} /></label>
        <label className="insp-field"><span>Note</span><textarea value={draft.note} maxLength={4000} rows={2} placeholder="Optional" onChange={(e) => setDraft({ ...draft, note: e.target.value })} /></label>
      </div>
      <div className="insp-actions"><button className="button primary full" disabled={busy === "add" || !register} onClick={() => { addToRegister(); setTask("register"); }}><Icon name="plus" size={15} />Save to the {noun} list</button><button className="button ghost" onClick={() => setTrace(null)}>Discard</button></div>
    </>}
  </>;

  const listTask = current ? <div className="insp-detail-view">
    <button className="insp-back" onClick={() => setSelected(null)}><Icon name="left" size={14} />All {noun}s</button>
    {current.photo?.url && photo(crack?.id === current.id ? `data:image/jpeg;base64,${crack.result.overlay_jpeg_base64}` : current.photo.url, `Photo of ${current.title}`, undefined, current.title)}
    <div className="insp-title-row"><i className="insp-sev" style={{ background: sevColor(current.severity) }} /><h3>{current.title}</h3></div>
    <p className="insp-sub">{TYPE_LABEL(current.type)}{current.element ? ` · ${current.element}` : ""} · seen by {current.seen_by} photos</p>
    <div className="insp-field"><span>Status</span><div className="insp-chips">{STATUSES.map((s) => <button key={s} className={current.status === s ? "on" : ""} onClick={() => update(current, { status: s })}>{s.replace("_", " ")}</button>)}</div></div>
    <div className="insp-field"><span>Severity</span><div className="insp-sev-pick" role="radiogroup" aria-label="Severity">{[1, 2, 3, 4, 5].map((s) => <button key={s} role="radio" aria-checked={current.severity === s} className={current.severity === s ? "on" : ""} style={{ ["--sev" as string]: sevColor(s) }} onClick={() => update(current, { severity: s })}><i />{SEV_LABEL[s]}</button>)}</div></div>
    {current.note && <p className="insp-note">{current.note}</p>}
    {Object.entries(current.measurements).map(([k, v]) => <div className="datum-row" key={k}><span>{k.replace(/_/g, " ")}</span><span>{v}</span></div>)}
    <div className="insp-actions">
      {current.frames[0] && onFrame && <button className="button secondary small" onClick={() => onFrame(current.frames[0].camera_index)}><Icon name="camera" size={14} />View from the photo</button>}
      {current.photo && <button className="button secondary small" disabled={busy === "crack"} onClick={() => run("crack", async () => setCrack({ id: current.id, result: await inspectApi.crack(scene, current.id) }))}><Icon name="scan" size={14} />{busy === "crack" ? "Looking…" : "Find cracks in photo"}</button>}
      <button className="button danger small" disabled={!!busy} onClick={() => remove(current)} aria-label="Delete"><Icon name="trash" size={14} /></button>
    </div>
    {crack?.id === current.id && <div className="insp-result">
      <h4>Crack candidates</h4>
      {crack.result.candidates.length ? <ul className="plan-existing">{crack.result.candidates.slice(0, 5).map((c, i) => <li key={i}><span><strong>{c.length_mm !== undefined ? `≈ ${c.length_mm.toFixed(0)} mm long, ${c.width_mm?.toFixed(1)} mm wide` : `${c.length_px.toFixed(0)} px long`}</strong><small>How line-like: {c.elongation}</small></span>
        {c.length_mm !== undefined && <button className="button secondary small" onClick={() => update(current, { measurements: { ...current.measurements, crack_length_mm: Math.round(c.length_mm!), crack_width_mm: Math.round((c.width_mm ?? 0) * 10) / 10 } })}>Record</button>}</li>)}</ul>
        : <p className="plan-note">No thin dark lines found in this photo.</p>}
      <p className="plan-note">{crack.result.basis}. Scale: {crack.result.gsd_mm.toFixed(2)} mm per pixel.</p>
    </div>}
  </div> : <>
    <div className="insp-sevbar" aria-label="By severity">{[5, 4, 3, 2, 1].map((s) => { const n = items.filter((a) => a.severity === s).length; return <div key={s} style={{ flexGrow: Math.max(n, 0.0001), ["--sev" as string]: sevColor(s) }} title={`${SEV_LABEL[s]}: ${n}`} className={n ? "" : "empty"} />; })}</div>
    <div className="insp-sevkey">{[5, 4, 3, 2, 1].map((s) => <span key={s}><i style={{ background: sevColor(s) }} />{SEV_LABEL[s]} <b>{items.filter((a) => a.severity === s).length}</b></span>)}</div>
    <div className="insp-chips insp-filter">{(["all", ...STATUSES] as const).map((s) => <button key={s} className={statusFilter === s ? "on" : ""} onClick={() => setStatusFilter(s)}>{s === "all" ? "All" : s.replace("_", " ")}<em>{s === "all" ? items.length : items.filter((a) => a.status === s).length}</em></button>)}</div>
    {shown.length ? <ul className="insp-list">{[...shown].sort((a, b) => b.severity - a.severity).map((a) => <li key={a.id}>
      <button onClick={() => setSelected(a.id)}>
        {a.photo?.url ? <Image unoptimized width={52} height={52} src={a.photo.url} alt="" /> : <span className="insp-nophoto"><Icon name="image" size={16} /></span>}
        <span><strong><i className="insp-sev" style={{ background: sevColor(a.severity) }} />{a.title}</strong><small>{TYPE_LABEL(a.type)}{a.element ? ` · ${a.element}` : ""}</small></span>
        <em className={`insp-status s-${a.status}`}>{a.status.replace("_", " ")}</em>
      </button></li>)}</ul>
      : <div className="tool-empty"><Icon name="pin" size={26} /><p>{items.length ? "Nothing with this status." : `No ${noun}s logged yet.`}</p>{!items.length && <button className="button secondary small" onClick={() => setTask("log")}><Icon name="plus" size={13} />Log the first one</button>}</div>}
    <button className="button secondary full" disabled={!items.length || busy === "report"} onClick={() => run("report", async () => downloadFile(await inspectApi.report(scene), scene))}><Icon name="download" size={15} />{busy === "report" ? "Writing report…" : mode === "inspection" ? "Download inspection report (PDF)" : "Download feature register (PDF)"}</button>
  </>;

  const tiltTask = <>
    <div className="insp-setting"><NumberField label="Search around the foot" unit="m" value={radius} min={0.1} max={10} step={0.1} onCommit={(v) => setRadius(v ?? 0.8)} /></div>
    {pickButton("tilt", "Click the foot of the pole", "cursor")}
    {drawing(["tilt"])}
    {busy === "tilt" && <p className="insp-busy"><span className="spinner" />Fitting the pole…</p>}
    {tilt && <div className="insp-result">
      <p className={`insp-verdict ${tilt.report.tilt_deg > 1 ? "warn" : "good"}`}><Icon name={tilt.report.tilt_deg > 1 ? "alert" : "check"} size={16} />Leans {fmtDeg(tilt.report.tilt_deg)} towards the {bearingName(tilt.report.lean_bearing_deg)}{tilt.report.tilt_deg > 1 ? " — more than 1°, worth a closer look." : " — within 1° of vertical."}</p>
      <div className="insp-kpis">{kpi(fmtDeg(tilt.report.tilt_deg), "from vertical", tilt.report.tilt_deg > 1 ? "warn" : "good")}{kpi(`${(tilt.report.top_offset_m * 100).toFixed(0)} cm`, "top off plumb")}{kpi(`${tilt.report.height_m.toFixed(1)} m`, "pole height")}{kpi(`${tilt.report.lean_mm_per_m.toFixed(0)} mm/m`, "lean rate")}</div>
      <p className="plan-note">{tilt.report.points} points, {(tilt.report.radius_rms_m * 100).toFixed(0)} cm scatter around the axis. {tilt.report.basis}</p>
    </div>}
  </>;

  const wireTask = <>
    <div className="insp-setting"><NumberField label="Required ground clearance" unit="m" value={limit} min={0} max={100} step={0.5} onCommit={setLimit} /></div>
    {pickButton("wire", "Click both ends of the cable", "cursor")}
    {drawing(["wire"])}
    {busy === "wire" && <p className="insp-busy"><span className="spinner" />Fitting the cable…</p>}
    {wire && <div className="insp-result">
      <p className={`insp-verdict ${wire.report.clearance_ok === false ? "bad" : "good"}`}><Icon name={wire.report.clearance_ok === false ? "alert" : "check"} size={16} />{wire.report.clearance_ok === false ? `Too low: under ${wire.report.clearance_limit_m} m at ${wire.report.clearance_at_m} m along the span.` : `Sags ${wire.report.sag_m.toFixed(2)} m over ${wire.report.span_m.toFixed(1)} m${wire.report.min_clearance_m !== null ? `, lowest point ${wire.report.min_clearance_m.toFixed(1)} m above ground` : ""}.`}</p>
      <div className="insp-kpis">{kpi(`${wire.report.span_m.toFixed(1)} m`, "span")}{kpi(`${wire.report.sag_m.toFixed(2)} m`, `sag (${wire.report.sag_pct_of_span}%)`)}{kpi(wire.report.min_clearance_m !== null ? `${wire.report.min_clearance_m.toFixed(1)} m` : "—", "lowest clearance", wire.report.clearance_ok === false ? "bad" : "good")}{kpi(String(wire.report.points), "points used")}</div>
      <p className="plan-note">Curve fit within {(wire.report.fit_rms_m * 100).toFixed(0)} cm. {wire.report.basis}.</p>
    </div>}
  </>;

  const compareTask = <>
    <label className="insp-field"><span>Older scan to compare with</span><select value={before} onChange={(e) => setBefore(e.target.value)}>
      <option value="">{epochs.length ? "Choose a scan" : "No other scan of this site"}</option>
      {epochs.map((e) => <option key={e.scene} value={e.scene}>{e.name}{e.compatible ? "" : " (no shared coordinates)"}</option>)}</select></label>
    <div className="insp-split">{pickButton("region", region ? "Area set — redraw" : "Limit to an area (optional)", "scan")}{region && <button className="button ghost small" onClick={() => setRegion(null)}>Use whole scan</button>}</div>
    {drawing(["region"])}
    <button className="button primary full" disabled={!before || !!busy} onClick={() => run("m3c2", async () => setM3c2(await inspectApi.m3c2(scene, before, region)))}><Icon name="activity" size={15} />{busy === "m3c2" ? "Comparing surfaces…" : "Find what moved"}</button>
    {m3c2 && <div className="insp-result">
      <p className={`insp-verdict ${m3c2.report.significant ? "warn" : "good"}`}><Icon name={m3c2.report.significant ? "alert" : "check"} size={16} />{m3c2.report.significant ? `${m3c2.report.significant} spots moved more than the measuring noise; the largest by ${m3c2.report.max_abs_significant_m !== null ? `${(m3c2.report.max_abs_significant_m * 1000).toFixed(0)} mm` : "—"}.` : "Nothing moved more than the measuring noise."}</p>
      <div className="insp-kpis">{kpi(String(m3c2.report.moved_towards_viewer), "moved outward", "warn")}{kpi(String(m3c2.report.moved_away), "moved inward", "cool")}{kpi(m3c2.report.median_lod95_m !== null ? `${(m3c2.report.median_lod95_m * 1000).toFixed(0)} mm` : "—", "smallest change we can trust")}{kpi(`${m3c2.report.observed}/${m3c2.report.core_points}`, "points compared")}</div>
      <p className="plan-note">Method: M3C2, cylinder {m3c2.report.parameters.projection_diameter_m} m, normals over {m3c2.report.parameters.normal_scale_m} m. {m3c2.report.basis}. {m3c2.report.note}.</p>
    </div>}
  </>;

  const terrainTask = <>
    <div className="insp-chips insp-layers" role="group" aria-label="Map type">{([["ortho", "Photo map"], ["hillshade_dtm", "Ground relief"], ["hillshade_dsm", "Surface relief"], ["lrm", "Local relief"], ["slope_deg", "Slope"]] as [TerrainLayer, string][]).map(([k, label]) =>
      <button key={k} className={terrainLayer === k && terrain ? "on" : ""} onClick={() => { setTerrainLayer(k); void run("terrain", async () => setTerrain(await inspectApi.terrain(scene, k))); }}>{label}</button>)}</div>
    {!terrain && busy !== "terrain" && <p className="plan-note">Pick a map type. Local relief is best for faint banks, ditches and wall lines.</p>}
    {busy === "terrain" && <p className="insp-busy"><span className="spinner" />Rendering the map…</p>}
    {terrain && <div className="insp-result">
      {photo(`data:image/png;base64,${terrain.png_base64}`, `${terrain.layer} of the site`, undefined, "Transparent areas were never seen by the drone.")}
      {terrain.layer === "lrm" && <><div className="plan-legend"><span><i style={{ background: "rgb(30,70,200)" }} />Lower: ditch, pit</span><span><i style={{ background: "rgb(210,40,40)" }} />Higher: bank, wall line</span><span>{(terrain.lrm_range_m[0] * 100).toFixed(0)} to {(terrain.lrm_range_m[1] * 100).toFixed(0)} cm</span></div>
        <label className="insp-show"><input type="checkbox" checked={drape} onChange={(e) => setDrape(e.target.checked)} />Drape over the model</label></>}
      <p className="plan-note">{terrain.basis}. Transparent means never observed.</p>
    </div>}
  </>;

  const sectionTask = <>
    <div className="insp-setting"><NumberField label="Slice thickness (each side)" unit="m" value={slab} min={0.02} max={5} step={0.05} onCommit={(v) => setSlab(v ?? 0.25)} /></div>
    {pickButton("section", "Click the two ends of the cut", "cursor")}
    {drawing(["section"])}
    {busy === "section" && <p className="insp-busy"><span className="spinner" />Cutting the section…</p>}
    {section && <div className="insp-result">
      {photo(`data:image/svg+xml;base64,${section.svg.content}`, "Section drawing", undefined, `${section.report.length_m.toFixed(2)} m long`)}
      <div className="insp-kpis">{kpi(`${section.report.length_m.toFixed(1)} m`, "length")}{kpi(`${(section.report.y_range[1] - section.report.y_range[0]).toFixed(2)} m`, "height range")}{kpi(section.report.points.toLocaleString(), "points")}{kpi(section.report.observed_pct !== null ? `${section.report.observed_pct.toFixed(0)}%` : "—", "ground seen")}</div>
      <div className="insp-actions"><button className="button secondary small" onClick={() => downloadFile(section.svg, scene)}><Icon name="download" size={14} />Drawing (SVG)</button><button className="button secondary small" onClick={() => downloadFile(section.dxf, scene)}><Icon name="download" size={14} />CAD (DXF)</button></div>
    </div>}
  </>;

  const recordTask = <>
    <button className="button primary full" disabled={busy === "record"} onClick={() => run("record", async () => { setRecord(await inspectApi.provenance(scene)); setFixity(null); })}><Icon name="seal" size={15} />{busy === "record" ? "Hashing every file…" : record ? "Rebuild the record" : "Build the archive record"}</button>
    {record && <div className="insp-result">
      {(["title", "date", "creator", "coverage"] as const).map((k) => <div className="datum-row" key={k}><span>{k[0].toUpperCase() + k.slice(1)}</span><span>{String(record.dublin_core[k] ?? "not recorded")}</span></div>)}
      <div className="datum-row"><span>Coordinates</span><span>{String((record.spatial_reference as { status?: string }).status ?? "—")}</span></div>
      <div className="datum-row"><span>Checksums</span><span>{record.fixity.length} files (SHA-256)</span></div>
      <div className="insp-actions"><button className="button secondary small" onClick={() => downloadFile({ filename: "archive-record.json", media_type: "application/json", encoding: "utf-8", content: JSON.stringify(record, null, 2) }, scene)}><Icon name="download" size={14} />Download record</button>
        <button className="button secondary small" onClick={() => run("verify", async () => { const r = await inspectApi.verify(scene, record); setFixity(r.changed.length ? `${r.changed.length} file(s) changed: ${r.changed.map((c) => c.file).join(", ")}` : "Every file still matches the record."); })}><Icon name="check" size={14} />Check files unchanged</button></div>
      {fixity && <p className={`insp-verdict ${fixity.startsWith("Every") ? "good" : "bad"}`}><Icon name={fixity.startsWith("Every") ? "check" : "alert"} size={16} />{fixity}</p>}
      <p className="plan-note">{record.notes.join(" ")}</p>
    </div>}
  </>;

  const restoreTask = <>
    <p className="insp-lead">Draw the missing tower or fallen wall as a scheme in <b>Plan</b> and tick <b>Hypothesis</b>. It is then always shown and exported as a guess, never as measured, and can be switched off for public viewing.</p>
    <button className="button primary full" onClick={onPlan}><Icon name="building" size={15} />Open Plan</button>
  </>;

  const body: Record<Task, React.ReactNode> = { log: logTask, register: listTask, tilt: tiltTask, wire: wireTask, compare: compareTask, terrain: terrainTask, section: sectionTask, record: recordTask, restore: restoreTask };
  const layerKey: Partial<Record<Task, string>> = { register: "pins", log: "pins", tilt: "tilt", wire: "wire", compare: "m3c2", section: "section" };

  return <div className="plan-panel ops-panel inspect-panel">
    <section className="inspector-section insp-top">
      <div className="plan-view" role="group" aria-label="Kind of work">{(["inspection", "archaeology"] as Mode[]).map((m) => <button key={m} className={mode === m ? "active" : ""} aria-pressed={mode === m} onClick={() => { setMode(m); setTask(TASKS[m][0].id); setSelected(null); onTool(null); }}>{m === "inspection" ? "Structures" : "Heritage"}</button>)}</div>
      <div className="insp-tasks" role="tablist" aria-label="What do you want to do?">{tasks.map((t) => <button key={t.id} role="tab" aria-selected={active.id === t.id} className={active.id === t.id ? "on" : ""} onClick={() => { setTask(t.id); onTool(null); }}>
        <Icon name={t.icon} size={18} /><span>{t.label}</span>{t.id === "register" && items.length > 0 && <em>{items.length}</em>}
        {((t.id === "tilt" && tilt) || (t.id === "wire" && wire) || (t.id === "compare" && m3c2) || (t.id === "section" && section) || (t.id === "terrain" && terrain) || (t.id === "record" && record)) && <i className="insp-has" aria-label="Has a result" />}</button>)}</div>
    </section>
    <section ref={bodyRef} className="inspector-section insp-body" key={`${mode}-${active.id}`}>
      <header className="insp-head"><div><h3>{active.label}</h3><p>{active.what}</p></div>{layerKey[active.id] && showToggle(layerKey[active.id]!)}</header>
      {error && <p className="form-error" role="alert">{error}</p>}
      {body[active.id]}
    </section>
    {mode === "inspection" && <section className="inspector-section insp-more">
      <button onClick={onMeasure}><Icon name="ruler" size={15} /><span>Measure lengths and areas<small>Opens Measure</small></span><Icon name="chevron" size={14} /></button>
      <button onClick={onPlan}><Icon name="building" size={15} /><span>Plan scaffold and lift access<small>Opens Plan</small></span><Icon name="chevron" size={14} /></button>
    </section>}

    {/* Photos that saw the traced spot, large, over the 3D view. */}
    {trace && trace.result.frames.length > 0 && !tool && <OnViewer kicker={`Photos that saw this spot · ${trace.result.seen_by} of ${trace.result.cameras}`}>
      <div className="insp-gallery">{trace.result.frames.slice(0, 4).map((f, i) => <figure key={f.name} className={i === 0 ? "best" : ""}>
        {f.url ? photo(f.url, `Photo ${f.name}`, { u: f.u, v: f.v, w: f.width, h: f.height }, `${f.name} · ${f.distance_m.toFixed(1)} m away · ${f.gsd_mm.toFixed(1)} mm per pixel`) : <span className="insp-frame-missing">image not on disk</span>}
        <figcaption><span>{i === 0 ? "Best view" : `Photo ${i + 1}`} · {f.distance_m.toFixed(0)} m away</span>{onFrame && <button className="text-button" onClick={() => onFrame(f.camera_index)}>View from here</button>}</figcaption>
      </figure>)}</div>
    </OnViewer>}

    <dialog ref={zoomDialog} className="insp-lightbox" onClick={(e) => { if (e.target === zoomDialog.current) setZoom(null); }} onCancel={() => setZoom(null)}>
      {zoom && <figure>
        <span className="insp-lightbox-img">{/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={zoom.url} alt={zoom.alt} />{zoom.marker && <i style={{ left: `${(100 * zoom.marker.u) / zoom.marker.w}%`, top: `${(100 * zoom.marker.v) / zoom.marker.h}%` }} />}</span>
        <figcaption>{zoom.caption ?? zoom.alt}<button className="icon-button" aria-label="Close" onClick={() => setZoom(null)}><Icon name="close" size={16} /></button></figcaption>
      </figure>}
    </dialog>
  </div>;
}
