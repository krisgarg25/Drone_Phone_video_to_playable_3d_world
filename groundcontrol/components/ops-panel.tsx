"use client";

import { useCallback, useEffect, useState } from "react";
import Image from "next/image";
import {
  fmtArea, fmtM3, opsApi, TOOL_HINT,
  type AccessReport, type ChangeReport, type CorridorReport, type CutFillReport, type DamageReport, type DetectionReport, type DeviationReport, type FirstMap,
  type FloodReport, type OpsApplication, type OpsSummary, type OpsTool, type PackKind, type Post, type Result, type VolumeReport,
} from "@/lib/ops";
import type { ExportFile, PlanMesh } from "@/lib/plan";
import type { Point } from "@/lib/workspace";
import { NumberField, Select } from "./plan-panel";
import { Icon } from "./studio-icons";
import "./plan-panel.css";
import "./ops-panel.css";

type Props = {
  scene: string; ready: boolean; tool: OpsTool | null; points: Point[];
  onTool: (tool: OpsTool | null) => void; onUndoPoint: () => void;
  onOverlay: (meshes: PlanMesh[]) => void; onDownload: (file: ExportFile) => void; onPlan: () => void;
};
type Key = "change" | "volume" | "damage" | "detections" | "access" | "flood" | "cutfill" | "corridor" | "deviation";
const APPS: { key: OpsApplication; label: string; copy: string }[] = [
  { key: "disaster", label: "Disaster", copy: "Damage, debris, access, flooding and people seen — from this flight against the one before." },
  { key: "construction", label: "Construction", copy: "Stockpiles, cut and fill against the design, progress since the last flight." },
  { key: "border", label: "Border", copy: "Coverage of a line from observation posts, blind stretches, change and tiled products." },
];
const xz = (p: Point) => [p[0], p[2]] as [number, number];
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
const GRADE_CLASS: Record<string, string> = { intact: "good", partial: "warn", collapsed: "bad", unknown: "muted" };

function Kpis({ items }: { items: [string, string, string?][] }) {
  return <div className="mission-kpis ops-kpis">{items.map(([value, label, tone]) => <div key={label} className={tone ?? ""}><b>{value}</b><small>{label}</small></div>)}</div>;
}
function Where({ row }: { row: { mgrs?: string } }) {
  return row.mgrs ? <small className="ops-mgrs">{row.mgrs}</small> : null;
}
function Notes({ notes }: { notes?: string[] }) {
  return notes?.length ? <details className="ops-notes"><summary>Limits</summary>{notes.map((n) => <p key={n} className="plan-note">{n}</p>)}</details> : null;
}

export function OpsPanel({ scene, ready, tool, points, onTool, onUndoPoint, onOverlay, onDownload, onPlan }: Props) {
  const [app, setApp] = useState<OpsApplication>("disaster");
  const [summary, setSummary] = useState<OpsSummary | null>(null);
  const [before, setBefore] = useState("");
  const [busy, setBusy] = useState<Key | "pack" | "tiles" | "design" | null>(null);
  const [error, setError] = useState("");
  const [shown, setShown] = useState<Key | null>(null);
  const [overlays, setOverlays] = useState<Partial<Record<Key, PlanMesh[]>>>({});
  const [change, setChange] = useState<ChangeReport | null>(null);
  const [volume, setVolume] = useState<VolumeReport | null>(null);
  const [damage, setDamage] = useState<DamageReport | null>(null);
  const [detections, setDetections] = useState<DetectionReport | null>(null);
  const [access, setAccess] = useState<AccessReport | null>(null);
  const [flood, setFlood] = useState<FloodReport | null>(null);
  const [cutfill, setCutfill] = useState<CutFillReport | null>(null);
  const [corridor, setCorridor] = useState<CorridorReport | null>(null);
  const [deviation, setDeviation] = useState<DeviationReport | null>(null);
  const [model, setModel] = useState("");
  const [modelFrame, setModelFrame] = useState<"scene" | "enu">("scene");
  const [devTol, setDevTol] = useState(0.05);
  const [firstmap, setFirstmap] = useState<FirstMap | null>(null);
  const [width, setWidth] = useState(2.5);
  const [rise, setRise] = useState(2);
  const [design, setDesign] = useState("");
  const [designFrame, setDesignFrame] = useState<"scene" | "utm">("scene");
  const [line, setLine] = useState<[number, number][]>([]);
  const [posts, setPosts] = useState<Post[]>([]);
  const [postHeight, setPostHeight] = useState(4);
  const [postRange, setPostRange] = useState(600);
  const [tileSize, setTileSize] = useState(250);

  const loadSummary = useCallback(async () => {
    try {
      const s = await opsApi.summary(scene);
      setSummary(s);
      setBefore((b) => b || s.epochs.find((e) => e.compatible)?.scene || "");
      setDesign((d) => d || s.designs.find((n) => !n.toLowerCase().endsWith(".glb")) || "");
      setModel((m) => m || s.designs.find((n) => n.toLowerCase().endsWith(".glb")) || "");
      setFirstmap(await opsApi.firstmap(scene));
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, [scene]);
  useEffect(() => {
    const timer = setTimeout(() => void loadSummary(), 0);
    return () => clearTimeout(timer);
  }, [loadSummary]);
  useEffect(() => { onOverlay(shown ? overlays[shown] ?? [] : []); }, [shown, overlays, onOverlay]);

  async function run<R>(key: Key, call: () => Promise<Result<R>>, set: (r: R) => void) {
    setBusy(key); setError("");
    try {
      const out = await call();
      set(out.report);
      setOverlays((o) => ({ ...o, [key]: out.overlay }));
      setShown(key);
      return out.report;
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); return null; }
    finally { setBusy(null); }
  }
  async function pack(kind: PackKind) {
    setBusy("pack"); setError("");
    try { onDownload(await opsApi.pack(scene, kind)); } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); } finally { setBusy(null); }
  }
  async function tiles() {
    setBusy("tiles"); setError("");
    try { onDownload(await opsApi.tiles(scene, tileSize)); } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); } finally { setBusy(null); }
  }
  async function upload(file: File) {
    if (file.size > 1_500_000) { setError("Design files are limited to 1.5 MB here; clip it to the site first."); return; }
    setBusy("design"); setError("");
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      const out = await opsApi.design(scene, file.name, btoa(binary));
      setSummary((s) => (s ? { ...s, designs: out.designs } : s));
      if (out.design.toLowerCase().endsWith(".glb")) setModel(out.design); else setDesign(out.design);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); } finally { setBusy(null); }
  }

  // Finished drawings: the host collects picks; this panel turns them into requests.
  const finish = useCallback(async () => {
    if (!tool) return;
    const pts = points.map(xz);
    if (pts.length < TOOL_HINT[tool].min) return;
    if (tool === "polygon") {
      onTool(null);
      if (!before) { setError("Choose the earlier flight to measure against."); return; }
      await run("volume", () => opsApi.volume(scene, before, pts), setVolume);
    } else if (tool === "access") {
      onTool(null);
      await run("access", () => opsApi.access(scene, { before, start: pts[0], end: pts[1], vehicle_width_m: width }), setAccess);
    } else if (tool === "seed") {
      onTool(null);
      await run("flood", () => opsApi.flood(scene, pts[0], rise), setFlood);
    } else if (tool === "line") {
      setLine(pts); onTool(null);
    } else if (tool === "post") {
      setPosts((p) => [...p, { id: `OP${p.length + 1}`, at: pts[0], height_m: postHeight, range_m: postRange, bearing_deg: null, fov_deg: 360 }]);
      onTool("post");
    }
  }, [tool, points, before, scene, width, rise, postHeight, postRange, onTool]);
  // Fixed-count tools (a seed, a post, staging + site) finish on their last click.
  useEffect(() => {
    if (!tool || TOOL_HINT[tool].limit !== points.length || tool === "polygon" || tool === "line") return;
    const timer = setTimeout(() => void finish(), 0);
    return () => clearTimeout(timer);
  }, [tool, points, finish]);
  useEffect(() => {
    const keys = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.key === "Enter" && tool) void finish();
      else if (event.key === "Escape" && tool) onTool(null);
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  }, [tool, finish, onTool]);

  // The drawing status sits under the tool that owns it, where the operator is looking.
  const drawing = (owned: OpsTool[]) => tool && owned.includes(tool) ? <div className="plan-drawing" role="status"><p>{TOOL_HINT[tool].hint}</p>
    <div className="plan-drawing-actions"><span>{points.length} point{points.length === 1 ? "" : "s"}</span>
      {(tool === "polygon" || tool === "line") && <><button className="text-button" disabled={!points.length} onClick={onUndoPoint}>Undo point</button>
        <button className="button primary small" disabled={points.length < TOOL_HINT[tool].min} onClick={() => void finish()}><Icon name="check" size={13} />Finish</button></>}
      <button className="button secondary small" onClick={() => onTool(null)}>{tool === "post" ? "Done" : "Cancel"}</button></div></div> : null;
  const epochs = summary?.epochs ?? [];
  const geo = summary?.frame.status === "georeferenced";
  const toolButton = (t: OpsTool, label: string, icon: "pin" | "ruler" | "arrow" | "eye" | "grid") =>
    <button className={`plan-tool${tool === t ? " active" : ""}`} aria-pressed={tool === t} disabled={!ready || !!busy} onClick={() => onTool(tool === t ? null : t)}><Icon name={icon} size={16} /><span>{label}</span></button>;
  const layerToggle = (key: Key) => overlays[key]?.length ? <label className="plan-switch"><input type="checkbox" checked={shown === key} onChange={(e) => setShown(e.target.checked ? key : null)} />On scan</label> : null;
  const packButton = (kind: PackKind, available: boolean) => <button className="button secondary small" disabled={!available || busy === "pack"} onClick={() => void pack(kind)}><Icon name="download" size={13} />Field pack</button>;

  const beforePicker = <div className="ops-epoch">
    <Select label="Earlier flight" value={before} options={[["", epochs.length ? "Choose a flight" : "No other flight of this site"], ...epochs.map((e) => [e.scene, `${e.name}${e.same_site ? " · same site" : ""}${e.compatible ? "" : " · no common frame"}`] as [string, string])]} onChange={setBefore} />
  </div>;

  const changeSection = <section className="inspector-section">
    <div className="plan-shadow-head"><div className="section-label"><span>CHANGE SINCE EARLIER FLIGHT</span></div>{layerToggle("change")}</div>
    <button className="button primary small full" disabled={!before || !!busy} onClick={() => void run("change", () => opsApi.change(scene, before), setChange)}><Icon name="activity" size={13} />{busy === "change" ? "Comparing…" : "Detect change"}</button>
    {change && <>
      <Kpis items={[[fmtM3(change.volume.gain_m3), "gained", "warn"], [fmtM3(change.volume.loss_m3), "lost", "cool"], [change.lod95_median_m === null ? "—" : `${(change.lod95_median_m * 100).toFixed(0)} cm`, "LoD95"], [String(change.regions.length), "regions"]]} />
      <div className="plan-legend"><span><i style={{ background: "rgb(214,60,50)" }} />Higher now</span><span><i style={{ background: "rgb(50,110,220)" }} />Lower now</span><span>Unmarked = no change beyond LoD, or not seen</span></div>
      <ul className="plan-existing">{change.regions.slice(0, 8).map((r) => <li key={r.id}><span><strong>{r.id} · {r.kind === "gain" ? "raised" : "lowered"} {Math.abs(r.mean_dh_m).toFixed(1)} m</strong><small>{fmtArea(r.area_m2)} · {fmtM3(r.volume_m3)} ± {r.volume_sigma_m3.toFixed(0)}</small><Where row={r} /></span></li>)}</ul>
      <p className="plan-note">Aligned {change.registration.dx_m.toFixed(2)} / {change.registration.dy_m.toFixed(2)} / {change.registration.dz_m.toFixed(2)} m (E/N/U) on {change.registration.stable_cells.toLocaleString()} stable cells. {change.epochs.basis}.</p>
      <div className="plan-feature-actions">{packButton("change", true)}</div>
      <Notes notes={change.notes} />
    </>}
  </section>;

  const volumeSection = (title: string) => <section className="inspector-section">
    <div className="plan-shadow-head"><div className="section-label"><span>{title}</span></div>{layerToggle("volume")}</div>
    <div className="plan-tools ops-tools">{toolButton("polygon", "Outline", "ruler")}</div>
      {drawing(["polygon"])}
    {volume && <>
      <Kpis items={[[fmtM3(volume.net_m3), "net change", volume.net_m3 >= 0 ? "warn" : "cool"], [volume.sigma_m3 === null ? "—" : `± ${volume.sigma_m3.toFixed(1)}`, "m³ (1σ)"], [fmtArea(volume.polygon_area_m2), "outline"], [pct(volume.observed_fraction), "observed"]]} />
      {!volume.valid && <p className="plan-message" role="alert">{volume.reason}</p>}
      <p className="plan-note">Gained {fmtM3(volume.gain_m3)}, removed {fmtM3(volume.loss_m3)} against the earlier flight; {pct(volume.filled_fraction)} of cells interpolated across sampling gaps.</p>
    </>}
  </section>;

  const disaster = <>
    <section className="inspector-section">
      <div className="section-label"><span>FIRST MAP</span>{firstmap?.available && <span>WINDOW {firstmap.window + 1}</span>}</div>
      {firstmap?.available ? <>
        {firstmap.png_base64 && <Image unoptimized width={320} height={220} className="ops-firstmap" src={`data:image/png;base64,${firstmap.png_base64}`} alt="First map from the sparse model" />}
        <p className="plan-note">{firstmap.label} {firstmap.registered_images} frames, {firstmap.points.toLocaleString()} points, {firstmap.georeferenced ? firstmap.crs : "local frame, unscaled"}{firstmap.elapsed_s !== null ? `, ready ${firstmap.elapsed_s.toFixed(0)} s into the run` : ""}.</p>
      </> : <p className="plan-note">{firstmap?.reason ?? "Checking for a progressive run…"}</p>}
    </section>
    {changeSection}
    {volumeSection("DEBRIS / LANDSLIDE VOLUME")}
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>BUILDING DAMAGE</span></div>{layerToggle("damage")}</div>
      <button className="button primary small full" disabled={!!busy} onClick={() => void run("damage", () => opsApi.damage(scene, before || null), setDamage)}><Icon name="activity" size={13} />{busy === "damage" ? "Grading…" : before ? "Grade buildings (footprints from earlier flight)" : "Grade buildings"}</button>
      {damage && <>
        <Kpis items={[[String(damage.counts.collapsed ?? 0), "collapsed", "bad"], [String(damage.counts.partial ?? 0), "partial", "warn"], [String(damage.counts.intact ?? 0), "intact", "good"], [String(damage.counts.unknown ?? 0), "unknown"]]} />
        <ul className="plan-existing">{damage.buildings.map((b) => <li key={b.id}><span><strong><span className={`ops-grade ${GRADE_CLASS[b.grade]}`}>{b.grade}</span> {b.id}{b.height_m !== undefined ? ` · ${b.height_m.toFixed(1)} m` : ""}</strong><small>{b.reason}</small><Where row={b} /></span></li>)}</ul>
        <p className="plan-note">Footprints: {damage.footprints}.</p>
        <div className="plan-feature-actions">{packButton("damage", true)}</div>
        <Notes notes={damage.notes} />
      </>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>PEOPLE AND VEHICLES</span></div>{layerToggle("detections")}</div>
      <button className="button secondary small full" disabled={!!busy} onClick={() => void run("detections", () => opsApi.detections(scene), setDetections)}><Icon name="eye" size={13} />{busy === "detections" ? "Placing…" : `Place detections (${summary?.detections.raw ?? 0} raw)`}</button>
      {detections && <>
        {detections.objects.length ? <ul className="plan-existing">{detections.objects.map((o) => <li key={o.id}><span><strong>{o.id} · {o.class}{o.moving ? " · moving" : ""}</strong><small>seen {o.sightings}× · {o.first_t ?? "?"}–{o.last_t ?? "?"} s</small><Where row={o} /></span></li>)}</ul> : <p className="plan-note">No detections could be placed.</p>}
        <p className="plan-note">Source: {detections.source}. {detections.unplaced ? `${detections.unplaced} boxes had no camera or missed the surface.` : ""}</p>
        <div className="plan-feature-actions">{packButton("detections", detections.objects.length > 0)}</div>
        <Notes notes={detections.notes} />
      </>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>ROAD ACCESS</span></div>{layerToggle("access")}</div>
      <div className="plan-shadow-grid"><NumberField label="Vehicle width" unit="m" value={width} min={1} max={6} step={0.5} onCommit={(v) => setWidth(v ?? 2.5)} /></div>
      <div className="plan-tools ops-tools">{toolButton("access", "Staging → site", "arrow")}
        <button className="plan-tool" disabled={!!busy} onClick={() => void run("access", () => opsApi.access(scene, { before, vehicle_width_m: width }), setAccess)}><Icon name="search" size={16} /><span>Blocked roads</span></button></div>
      {drawing(["access"])}
      {access && <>
        <Kpis items={[[String(access.blocked.length), "blocked", access.blocked.length ? "bad" : "good"], [fmtArea(access.blocked_m2), "obstructed"], [access.route ? `${access.route.length_m.toFixed(0)} m` : "—", "route"], [access.route ? `${(access.route.eta_s / 60).toFixed(1)} min` : "—", "ETA"]]} />
        {access.route_error && <p className="plan-message" role="alert">No route: {access.route_error}</p>}
        {access.route && <p className="plan-note">{access.route.road_m.toFixed(0)} m on road, {access.route.offroad_m.toFixed(0)} m off road, for a {access.route.vehicle_width_m} m vehicle.{Object.entries(access.route.snapped_m).filter(([, m]) => m > 0).map(([name, m]) => ` The ${name} moved ${m.toFixed(1)} m to drivable ground.`).join("")}</p>}
        <ul className="plan-existing">{access.blocked.map((b) => <li key={b.id}><span><strong>{b.id} · obstacle {b.max_height_m.toFixed(1)} m</strong><small>{fmtArea(b.area_m2)}{b.new_since_before ? " · new since earlier flight" : ""}</small><Where row={b} /></span></li>)}</ul>
        <p className="plan-note">Roads: {access.road_source}.</p>
        <div className="plan-feature-actions">{packButton("access", true)}</div>
        <Notes notes={access.route?.notes} />
      </>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>FLOOD</span></div>{layerToggle("flood")}</div>
      <div className="plan-shadow-grid"><NumberField label="Water rise at seed" unit="m" value={rise} min={0.1} max={30} step={0.5} onCommit={(v) => setRise(v ?? 2)} /></div>
      <div className="plan-tools ops-tools">{toolButton("seed", "Water seed", "pin")}</div>
      {drawing(["seed"])}
      {flood && <>
        <Kpis items={[[fmtArea(flood.flooded_m2), "flooded", "cool"], [`${flood.max_depth_m.toFixed(1)} m`, "max depth"], [fmtM3(flood.volume_m3), "water"], [String(flood.buildings.length), "buildings wet"]]} />
        <div className="plan-legend">{Object.entries(flood.depth_bands_m2).map(([band, area], i) => <span key={band}><i style={{ background: ["rgb(110,170,240)", "rgb(50,110,210)", "rgb(20,60,170)"][i] }} />{band}: {fmtArea(area)}</span>)}</div>
        <ul className="plan-existing">{flood.buildings.slice(0, 8).map((b) => <li key={b.id}><span><strong>{b.id} · {b.max_depth_m.toFixed(1)} m at the walls</strong><Where row={b} /></span></li>)}</ul>
        <p className="plan-note">Level {flood.level_m.toFixed(2)} m; {fmtArea(flood.isolated_low_ground_m2)} of low ground is not connected to the seed and stays dry in this model.</p>
        <div className="plan-feature-actions">{packButton("flood", true)}</div>
        <Notes notes={flood.notes} />
      </>}
    </section>
    <section className="inspector-section">
      <div className="section-label"><span>RELIEF PLANNING</span></div>
      <p className="plan-note">Tents, medical tents, water bladders, generators, helipads and staging areas are in the Plan tab&apos;s object catalogue, with hazard zones as plots. They stay proposals, never part of the scan.</p>
      <button className="button secondary small full" onClick={onPlan}><Icon name="grid" size={13} />Open the Plan tab</button>
    </section>
  </>;

  const construction = <>
    {volumeSection("STOCKPILE VOLUME")}
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>CUT / FILL AGAINST DESIGN</span></div>{layerToggle("cutfill")}</div>
      <label className="button secondary small full ops-upload"><Icon name="upload" size={13} />{busy === "design" ? "Reading…" : "Upload design (LandXML, DXF 3DFACE, GeoTIFF, GLB model)"}
        <input type="file" accept=".xml,.landxml,.dxf,.tif,.tiff,.glb" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); e.target.value = ""; }} /></label>
      {summary && summary.designs.length > 0 && <div className="plan-shadow-grid">
        <Select label="Design surface" value={design} options={summary.designs.filter((d) => !d.toLowerCase().endsWith(".glb")).map((d) => [d, d] as [string, string])} onChange={setDesign} />
        <Select label="Design coordinates" value={designFrame} options={[["scene", "Same as the scan"], ...(geo ? [["utm", "UTM (from GPS fit)"] as [string, string]] : [])]} onChange={(v) => setDesignFrame(v as "scene" | "utm")} />
      </div>}
      <button className="button primary small full" disabled={!design || !!busy} onClick={() => void run("cutfill", () => opsApi.cutfill(scene, design, designFrame, before || null), setCutfill)}><Icon name="activity" size={13} />{busy === "cutfill" ? "Comparing…" : before ? "Compare, with progress since earlier flight" : "Compare to design"}</button>
      {cutfill && <>
        <Kpis items={[[fmtM3(cutfill.cut_m3), "cut left", "warn"], [fmtM3(cutfill.fill_m3), "fill left", "cool"], [fmtM3(cutfill.net_m3), "net"], [pct(cutfill.design_observed_fraction), "design seen"]]} />
        <div className="plan-legend"><span><i style={{ background: "rgb(214,70,60)" }} />Above design (cut)</span><span><i style={{ background: "rgb(60,110,214)" }} />Below (fill)</span><span><i style={{ background: "rgb(80,200,110)" }} />On grade ±{(cutfill.tolerance_m * 100).toFixed(0)} cm</span></div>
        {cutfill.zones?.map((z) => <div key={z.zone} className="datum-row"><span>Progress · {z.zone}</span><span>{z.progress_pct === null || z.progress_pct === undefined ? "—" : `${z.progress_pct.toFixed(0)}%`} ({fmtM3(z.remaining_m3)} left of {z.start_m3 !== undefined ? fmtM3(z.start_m3) : "?"})</span></div>)}
        <p className="plan-note">Cut ± {cutfill.cut_sigma_m3.toFixed(1)} m³, fill ± {cutfill.fill_sigma_m3.toFixed(1)} m³ · design in {cutfill.design_frame}.</p>
        <div className="plan-feature-actions">{packButton("cutfill", true)}</div>
        <Notes notes={cutfill.notes} />
      </>}
    </section>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>AS-BUILT VS DESIGN MODEL</span></div>{layerToggle("deviation")}</div>
      {summary && summary.designs.some((d) => d.toLowerCase().endsWith(".glb")) ? <div className="plan-shadow-grid">
        <Select label="Design model" value={model} options={summary.designs.filter((d) => d.toLowerCase().endsWith(".glb")).map((d) => [d, d] as [string, string])} onChange={setModel} />
        <Select label="Model coordinates" value={modelFrame} options={[["scene", "Same as the scan"], ...(geo ? [["enu", "ENU at the GPS origin"] as [string, string]] : [])]} onChange={(v) => setModelFrame(v as "scene" | "enu")} />
        <NumberField label="Tolerance" unit="m" value={devTol} min={0.001} max={1} step={0.005} onCommit={(v) => setDevTol(v ?? 0.05)} />
      </div> : <p className="plan-note">Upload the design model (GLB, from BIM or CAD) with the button above.</p>}
      <button className="button primary small full" disabled={!model || !!busy} onClick={() => void run("deviation", () => opsApi.deviation(scene, model, modelFrame, devTol), setDeviation)}><Icon name="activity" size={13} />{busy === "deviation" ? "Measuring…" : "Compare scan to model"}</button>
      {deviation && <>
        <Kpis items={[[deviation.within_tolerance_pct !== null ? `${deviation.within_tolerance_pct.toFixed(0)}%` : "—", `within ±${(deviation.tolerance_m * 100).toFixed(0)} cm`, "good"],
          [deviation.proud_pct !== null ? `${deviation.proud_pct.toFixed(0)}%` : "—", "proud", "warn"], [deviation.short_pct !== null ? `${deviation.short_pct.toFixed(0)}%` : "—", "short", "cool"],
          [deviation.rms_m !== null ? `${(deviation.rms_m * 100).toFixed(1)} cm` : "—", "RMS"]]} />
        <div className="plan-legend"><span><i style={{ background: "rgb(214,60,50)" }} />Proud of design</span><span><i style={{ background: "rgb(50,110,220)" }} />Short of design</span><span><i style={{ background: "rgb(70,190,90)" }} />Within tolerance</span></div>
        <p className="plan-note">{deviation.scan_points_compared.toLocaleString()} scan points compared ({deviation.scan_points_unrelated.toLocaleString()} near the model but not on it); sampling error ≈ {(deviation.sampling_error_m * 100).toFixed(1)} cm. {deviation.basis}.</p>
      </>}
    </section>
    <section className="inspector-section">
      <div className="section-label"><span>SITE LOGISTICS</span></div>
      <p className="plan-note">Place tower cranes, site offices and material yards in the Plan tab: each crane shows its jib radius and lists every scanned structure the jib would hit.</p>
      <button className="button secondary small full" onClick={onPlan}><Icon name="grid" size={13} />Open the Plan tab</button>
    </section>
    {changeSection}
  </>;

  const border = <>
    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>LINE COVERAGE</span></div>{layerToggle("corridor")}</div>
      <div className="plan-tools ops-tools">{toolButton("line", line.length ? `Line (${line.length})` : "Draw line", "ruler")}{toolButton("post", `Posts (${posts.length})`, "pin")}</div>
      {drawing(["line", "post"])}
      <div className="plan-shadow-grid">
        <NumberField label="Post eye height" unit="m" value={postHeight} min={0} max={100} step={0.5} onCommit={(v) => setPostHeight(v ?? 4)} />
        <NumberField label="Post range" unit="m" value={postRange} min={10} max={20000} step={50} onCommit={(v) => setPostRange(v ?? 600)} />
      </div>
      {posts.length > 0 && <ul className="plan-existing">{posts.map((p, i) => <li key={p.id}><span><strong>{p.id}</strong><small>{p.height_m} m eye · {p.range_m} m range</small></span><button className="icon-button" aria-label={`Remove ${p.id}`} onClick={() => setPosts(posts.filter((_, k) => k !== i))}><Icon name="trash" size={13} /></button></li>)}</ul>}
      <button className="button primary small full" disabled={line.length < 2 || !posts.length || !!busy} onClick={() => void run("corridor", () => opsApi.corridor(scene, line, posts), setCorridor)}><Icon name="eye" size={13} />{busy === "corridor" ? "Tracing sight lines…" : "Analyse coverage"}</button>
      {corridor && <>
        <Kpis items={[[pct(corridor.covered_fraction), "line covered", corridor.covered_fraction > 0.9 ? "good" : "warn"], [`${corridor.line_length_m.toFixed(0)} m`, "line"], [String(corridor.blind_stretches.length), "blind"], [`${corridor.profile.climb_m.toFixed(0)} m`, "climb"]]} />
        <div className="mission-profile ops-profile" aria-label="Ground profile along the line; red where no post sees it">
          {corridor.profile.z.map((z, i) => { const zs = corridor.profile.z.filter((v): v is number => v !== null); const lo = Math.min(...zs), hi = Math.max(...zs); return <i key={i} className={z === null ? "gap" : corridor.profile.covered[i] ? "" : "hot2"} style={{ height: z === null ? "4%" : `${20 + 80 * ((z - lo) / Math.max(0.5, hi - lo))}%` }} />; })}
        </div>
        <ul className="plan-existing">{corridor.blind_stretches.map((b, i) => <li key={i}><span><strong>{b.from_m.toFixed(0)}–{b.to_m.toFixed(0)} m · {b.length_m.toFixed(0)} m blind</strong><small>{b.unobserved_ground ? "includes ground the flight never saw" : "hidden by terrain or structures"}</small><Where row={b} /></span></li>)}</ul>
        <div className="plan-feature-actions">{packButton("corridor", true)}</div>
        <Notes notes={corridor.notes} />
      </>}
    </section>
    {changeSection}
    <section className="inspector-section">
      <div className="section-label"><span>TILED PRODUCTS</span></div>
      <div className="plan-export"><NumberField label="Tile size" unit="m" value={tileSize} min={20} max={5000} step={50} onCommit={(v) => setTileSize(v ?? 250)} />
        <button className="button secondary small" disabled={busy === "tiles"} onClick={() => void tiles()}><Icon name="download" size={13} />{busy === "tiles" ? "Tiling…" : "DSM + LAS tiles"}</button></div>
      <p className="plan-note">{geo ? "Tiles are written in UTM from the GPS fit." : "Local scene: tiles stay in scene coordinates (no GeoTIFF CRS)."} Plain GeoTIFF and LAS, not COG/COPC.</p>
    </section>
  </>;

  return <div className="plan-panel ops-panel">
    <section className="inspector-section plan-head">
      <div className="section-label"><span>OPERATIONS</span>{summary && <span>{summary.grid.extent_m[0].toFixed(0)} × {summary.grid.extent_m[1].toFixed(0)} m</span>}</div>
      <div className="plan-view" role="group" aria-label="Application">{APPS.map((a) => <button key={a.key} className={app === a.key ? "active" : ""} aria-pressed={app === a.key} onClick={() => { setApp(a.key); onTool(null); }}>{a.label}</button>)}</div>
      <p className="inspector-copy">{APPS.find((a) => a.key === app)?.copy}</p>
      {summary && <div className={`plan-frame plan-frame-${summary.frame.status}`}><Icon name={geo ? "globe" : "info"} size={13} />
        <span>{geo ? "Georeferenced — every result carries MGRS and exports to KMZ" : "No GPS fit — results stay in scene coordinates, no KMZ"}</span></div>}
      {beforePicker}
      {error && <p className="plan-message" role="alert">{error}</p>}
    </section>
    {app === "disaster" ? disaster : app === "construction" ? construction : border}
  </div>;
}
