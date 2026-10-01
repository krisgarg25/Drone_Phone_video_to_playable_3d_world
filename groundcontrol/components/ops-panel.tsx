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
import { NumberField } from "./plan-panel";
import { Icon, type IconName } from "./studio-icons";
import { OnViewer } from "./on-viewer";
import "./plan-panel.css";
import "./ops-panel.css";
import "./inspect.css";

type Props = {
  scene: string; ready: boolean; tool: OpsTool | null; points: Point[];
  onTool: (tool: OpsTool | null) => void; onUndoPoint: () => void;
  onOverlay: (meshes: PlanMesh[]) => void; onDownload: (file: ExportFile) => void; onPlan: () => void;
  onFocus?: (point: [number, number, number]) => void;
};
type Key = "change" | "volume" | "damage" | "detections" | "access" | "flood" | "cutfill" | "corridor" | "deviation";
const APPS: { key: OpsApplication; label: string; icon: IconName }[] = [
  { key: "disaster", label: "Disaster", icon: "alert" },
  { key: "construction", label: "Construction", icon: "crane" },
  { key: "border", label: "Border", icon: "route" },
];
type Task = "firstmap" | "change" | "volume" | "damage" | "detections" | "access" | "flood" | "relief" | "cutfill" | "deviation" | "logistics" | "corridor" | "tiles";
/** One job at a time, named for what the person wants to know, not for the algorithm. */
const TASKS: Record<OpsApplication, { id: Task; label: string; icon: IconName; what: string }[]> = {
  disaster: [
    { id: "change", label: "What changed", icon: "history", what: "Compare with the flight before the event: what got higher and what got lower." },
    { id: "damage", label: "Building damage", icon: "building", what: "Grade every building as intact, partly damaged or collapsed." },
    { id: "volume", label: "Debris volume", icon: "mountain", what: "Outline a debris pile or landslide to get its volume." },
    { id: "detections", label: "People & vehicles", icon: "eye", what: "Put the people and vehicles seen in the video onto the map." },
    { id: "access", label: "Road access", icon: "route", what: "Find blocked roads and a route a vehicle can still drive." },
    { id: "flood", label: "Flood", icon: "wave", what: "Raise the water from one point and see what goes under." },
    { id: "firstmap", label: "First map", icon: "image", what: "The quick map made while the drone was still flying." },
    { id: "relief", label: "Relief camp", icon: "flag", what: "Lay out tents, helipads and staging areas on the scan." },
  ],
  construction: [
    { id: "volume", label: "Stockpile volume", icon: "mountain", what: "Outline a stockpile to get its volume against the earlier flight." },
    { id: "cutfill", label: "Cut & fill", icon: "crane", what: "How much earth is still to dig or fill to reach the design." },
    { id: "deviation", label: "Built vs design", icon: "twin", what: "Compare what was built with the BIM or CAD model." },
    { id: "change", label: "What changed", icon: "history", what: "Everything that moved on site since the earlier flight." },
    { id: "logistics", label: "Site logistics", icon: "building", what: "Place cranes, offices and yards and check crane clearances." },
  ],
  border: [
    { id: "corridor", label: "Line coverage", icon: "eye", what: "Which stretches of a border or fence your observation posts can see." },
    { id: "change", label: "What changed", icon: "history", what: "New tracks, digging or structures since the earlier flight." },
    { id: "tiles", label: "Map tiles", icon: "grid", what: "Cut the site into height-map and point-cloud tiles for GIS." },
  ],
};
const GRADE_LABEL: Record<string, string> = { intact: "Intact", partial: "Partly damaged", collapsed: "Collapsed", unknown: "Not judged" };
const xz = (p: Point) => [p[0], p[2]] as [number, number];
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
const GRADE_CLASS: Record<string, string> = { intact: "good", partial: "warn", collapsed: "bad", unknown: "muted" };

function Notes({ notes }: { notes?: string[] }) {
  return notes?.length ? <details className="ops-notes"><summary>How this was worked out</summary>{notes.map((n) => <p key={n} className="plan-note">{n}</p>)}</details> : null;
}

export function OpsPanel({ scene, ready, tool, points, onTool, onUndoPoint, onOverlay, onDownload, onPlan, onFocus }: Props) {
  const [app, setApp] = useState<OpsApplication>("disaster");
  const [task, setTask] = useState<Task>("change");
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

  const drawing = (owned: OpsTool[]) => tool && owned.includes(tool) ? <OnViewer><div className="plan-drawing" role="status"><p>{TOOL_HINT[tool].hint}</p>
    <div className="plan-drawing-actions"><span>{points.length} point{points.length === 1 ? "" : "s"}</span>
      {(tool === "polygon" || tool === "line") && <><button className="text-button" disabled={!points.length} onClick={onUndoPoint}>Undo point</button>
        <button className="button primary small" disabled={points.length < TOOL_HINT[tool].min} onClick={() => void finish()}><Icon name="check" size={13} />Finish</button></>}
      <button className="button secondary small" onClick={() => onTool(null)}>{tool === "post" ? "Done" : "Cancel"}</button></div></div></OnViewer> : null;
  const epochs = summary?.epochs ?? [];
  const geo = summary?.frame.status === "georeferenced";
  const tasks = TASKS[app];
  const active = tasks.find((t) => t.id === task) ?? tasks[0];
  const results: Partial<Record<Task, boolean>> = { firstmap: !!firstmap?.available, change: !!change, volume: !!volume, damage: !!damage, detections: !!detections, access: !!access, flood: !!flood, cutfill: !!cutfill, deviation: !!deviation, corridor: !!corridor };

  const pickButton = (t: OpsTool, label: string, sub?: string) =>
    <button className={`insp-pick${tool === t ? " is-armed" : ""}`} aria-pressed={tool === t} disabled={!ready || !!busy} onClick={() => onTool(tool === t ? null : t)}>
      <Icon name={tool === t ? "close" : "cursor"} size={18} /><span>{tool === t ? (t === "post" ? "Done placing posts" : "Cancel picking") : label}<small>{tool === t ? "Or press Esc" : ready ? sub ?? "Then click on the 3D view" : "Waiting for the 3D view"}</small></span></button>;
  const runButton = (label: string, busyLabel: string, key: Key | "tiles", disabled: boolean, onClick: () => void, icon: IconName = "activity") =>
    <button className="button primary full" disabled={disabled || !!busy} onClick={onClick}>{busy === key ? <><span className="spinner" />{busyLabel}</> : <><Icon name={icon} size={15} />{label}</>}</button>;
  const showToggle = (key: Key) => overlays[key]?.length ? <label className="insp-show"><input type="checkbox" checked={shown === key} onChange={(e) => setShown(e.target.checked ? key : null)} />Show on model</label> : null;
  const packButton = (kind: PackKind, available: boolean) => <button className="button secondary small" disabled={!available || busy === "pack"} onClick={() => void pack(kind)} data-tip={geo ? "KMZ for Google Earth and GeoJSON, with MGRS on every item" : "GeoJSON in scene coordinates (no GPS fit, so no KMZ)"}><Icon name="download" size={14} />{busy === "pack" ? "Packing…" : "Download field pack"}</button>;
  const verdict = (tone: "good" | "warn" | "bad", text: string) => <p className={`insp-verdict ${tone}`}><Icon name={tone === "good" ? "check" : "alert"} size={16} />{text}</p>;
  const kpis = (items: [string, string, string?][]) => <div className="insp-kpis">{items.map(([value, label, tone]) => <div key={label} className={tone ?? ""}><b>{value}</b><small>{label}</small></div>)}</div>;
  const beforeField = (required: boolean, why: string) => <label className="insp-field ops-before"><span>Compare with an earlier flight{required ? "" : " (optional)"}</span>
    <select value={before} onChange={(e) => setBefore(e.target.value)}>
      <option value="">{epochs.length ? (required ? "Choose a flight" : "None") : "No other flight of this site"}</option>
      {epochs.map((e) => <option key={e.scene} value={e.scene} disabled={!e.compatible}>{e.name}{e.same_site ? " · same site" : ""}{e.compatible ? "" : " · can't line up"}</option>)}</select>
    <small>{why}</small></label>;
  const rows = (list: RowItem[], empty: string) => list.length ? <RowList rows={list} onFocus={onFocus} /> : <p className="plan-note">{empty}</p>;

  const firstmapTask = firstmap?.available ? <div className="insp-result">
    {firstmap.png_base64 && <Image unoptimized width={320} height={220} className="ops-firstmap" src={`data:image/png;base64,${firstmap.png_base64}`} alt="First map from the sparse model" />}
    {kpis([[String(firstmap.registered_images), "photos placed"], [firstmap.points.toLocaleString(), "3D points"], [firstmap.elapsed_s !== null ? `${firstmap.elapsed_s.toFixed(0)} s` : "—", "into the flight"], [firstmap.georeferenced ? "GPS" : "Local", "coordinates"]])}
    <p className="plan-note">{firstmap.label}{firstmap.georeferenced && firstmap.crs ? ` · ${firstmap.crs}` : ""}.</p>
  </div> : <div className="tool-empty"><Icon name="image" size={26} /><p>No quick map for this scene.</p><small>{firstmap?.reason ?? "Checking…"}</small></div>;

  const changeTask = <>
    {beforeField(true, "The two scans are lined up on ground that did not move, then compared cell by cell.")}
    {runButton("Find what changed", "Comparing the two flights…", "change", !before, () => void run("change", () => opsApi.change(scene, before), setChange))}
    {change && <div className="insp-result">
      {verdict(change.regions.length ? "warn" : "good", change.regions.length ? `${change.regions.length} area${change.regions.length === 1 ? "" : "s"} changed: ${fmtM3(change.volume.gain_m3)} added and ${fmtM3(change.volume.loss_m3)} removed.` : "Nothing changed more than the measuring noise.")}
      {kpis([[fmtM3(change.volume.gain_m3), "added (higher now)", "warn"], [fmtM3(change.volume.loss_m3), "removed (lower now)", "cool"], [change.lod95_median_m === null ? "—" : `${(change.lod95_median_m * 100).toFixed(0)} cm`, "smallest change we trust"], [String(change.regions.length), "changed areas"]])}
      {rows(change.regions.map((r) => ({ id: r.id, title: `${r.kind === "gain" ? "Raised" : "Lowered"} ${Math.abs(r.mean_dh_m).toFixed(1)} m`, sub: `${fmtArea(r.area_m2)} · ${fmtM3(r.volume_m3)} ± ${r.volume_sigma_m3.toFixed(0)}`, tone: r.kind === "gain" ? "warn" : "cool", geo: r })), "No changed areas.")}
      <div className="insp-actions">{packButton("change", true)}</div>
      <Notes notes={[`Lined up by ${change.registration.dx_m.toFixed(2)} / ${change.registration.dy_m.toFixed(2)} / ${change.registration.dz_m.toFixed(2)} m (east / north / up) on ${change.registration.stable_cells.toLocaleString()} stable cells. ${change.epochs.basis}.`, ...change.notes]} />
    </div>}
  </>;

  const volumeTask = (thing: string) => <>
    {beforeField(true, `The ${thing} is measured as the difference from the earlier flight inside your outline.`)}
    {pickButton("polygon", `Outline the ${thing} on the model`, "Click around it, then press Finish")}
    {drawing(["polygon"])}
    {busy === "volume" && <p className="insp-busy"><span className="spinner" />Measuring the volume…</p>}
    {volume && <div className="insp-result">
      {volume.valid ? verdict("good", `${fmtM3(Math.abs(volume.net_m3))} ${volume.net_m3 >= 0 ? "added" : "removed"} inside the outline${volume.sigma_m3 !== null ? `, give or take ${volume.sigma_m3.toFixed(1)} m³` : ""}.`) : verdict("bad", volume.reason ?? "This outline could not be measured.")}
      {kpis([[fmtM3(volume.net_m3), "net change", volume.net_m3 >= 0 ? "warn" : "cool"], [volume.sigma_m3 === null ? "—" : `± ${volume.sigma_m3.toFixed(1)} m³`, "uncertainty (1σ)"], [fmtArea(volume.polygon_area_m2), "outlined area"], [pct(volume.observed_fraction), "seen by the drone"]])}
      <p className="plan-note">Added {fmtM3(volume.gain_m3)}, removed {fmtM3(volume.loss_m3)}; {pct(volume.filled_fraction)} of the area was filled in across small gaps.</p>
    </div>}
  </>;

  const damageTask = <>
    {beforeField(false, "With an earlier flight, building outlines come from before the event; without one, from this scan.")}
    {runButton("Grade every building", "Grading buildings…", "damage", false, () => void run("damage", () => opsApi.damage(scene, before || null), setDamage), "building")}
    {damage && <div className="insp-result">
      {verdict((damage.counts.collapsed ?? 0) ? "bad" : (damage.counts.partial ?? 0) ? "warn" : "good", `${damage.counts.collapsed ?? 0} collapsed, ${damage.counts.partial ?? 0} partly damaged, ${damage.counts.intact ?? 0} intact${damage.counts.unknown ? `, ${damage.counts.unknown} could not be judged` : ""}.`)}
      {kpis([[String(damage.counts.collapsed ?? 0), "collapsed", "bad"], [String(damage.counts.partial ?? 0), "partly damaged", "warn"], [String(damage.counts.intact ?? 0), "intact", "good"], [String(damage.counts.unknown ?? 0), "not judged"]])}
      {rows(damage.buildings.map((b) => ({ id: b.id, title: `${b.id}${b.height_m !== undefined ? ` · ${b.height_m.toFixed(1)} m tall` : ""}`, sub: b.reason, badge: GRADE_LABEL[b.grade], tone: GRADE_CLASS[b.grade], geo: b })), "No buildings found.")}
      <div className="insp-actions">{packButton("damage", true)}</div>
      <Notes notes={[`Building outlines: ${damage.footprints}.`, ...damage.notes]} />
    </div>}
  </>;

  const detectionsTask = <>
    <p className="insp-lead">The video was scanned for people and vehicles{summary ? ` (${summary.detections.raw} sightings)` : ""}. This places each one on the ground where the cameras saw it.</p>
    {runButton("Place them on the map", "Placing sightings…", "detections", false, () => void run("detections", () => opsApi.detections(scene), setDetections), "eye")}
    {detections && <div className="insp-result">
      {verdict(detections.objects.length ? "warn" : "good", detections.objects.length ? `${detections.objects.length} placed: ${detections.objects.filter((o) => o.class === "person").length} people, ${detections.objects.filter((o) => o.class !== "person").length} vehicles${detections.objects.some((o) => o.moving) ? `, ${detections.objects.filter((o) => o.moving).length} moving` : ""}.` : "Nothing could be placed on the ground.")}
      {rows(detections.objects.map((o) => ({ id: o.id, title: `${o.class[0].toUpperCase()}${o.class.slice(1)}${o.moving ? " · moving" : ""}`, sub: `seen ${o.sightings}× between ${o.first_t ?? "?"} s and ${o.last_t ?? "?"} s`, badge: o.id, geo: o })), "No detections could be placed.")}
      <div className="insp-actions">{packButton("detections", detections.objects.length > 0)}</div>
      <Notes notes={[`Source: ${detections.source}.${detections.unplaced ? ` ${detections.unplaced} sightings had no camera or missed the ground.` : ""}`, ...detections.notes]} />
    </div>}
  </>;

  const accessTask = <>
    <div className="insp-setting"><NumberField label="Vehicle width" unit="m" value={width} min={1} max={6} step={0.5} onCommit={(v) => setWidth(v ?? 2.5)} /></div>
    {beforeField(false, "With an earlier flight, obstacles that are new since then are flagged.")}
    {pickButton("access", "Click where the vehicle starts, then where it must reach", "Two clicks on the 3D view")}
    <button className="button secondary full" disabled={!!busy} onClick={() => void run("access", () => opsApi.access(scene, { before, vehicle_width_m: width }), setAccess)}><Icon name="search" size={15} />Only find blocked roads</button>
    {drawing(["access"])}
    {busy === "access" && <p className="insp-busy"><span className="spinner" />Checking the roads…</p>}
    {access && <div className="insp-result">
      {access.route_error ? verdict("bad", `No drivable route: ${access.route_error}`) : access.route ? verdict(access.blocked.length ? "warn" : "good", `Route found: ${access.route.length_m.toFixed(0)} m, about ${(access.route.eta_s / 60).toFixed(1)} min for a ${access.route.vehicle_width_m} m vehicle. ${access.blocked.length} blocked spot${access.blocked.length === 1 ? "" : "s"} on the roads.`) : verdict(access.blocked.length ? "warn" : "good", `${access.blocked.length} blocked spot${access.blocked.length === 1 ? "" : "s"} on the roads.`)}
      {kpis([[String(access.blocked.length), "blocked spots", access.blocked.length ? "bad" : "good"], [fmtArea(access.blocked_m2), "road covered"], [access.route ? `${access.route.length_m.toFixed(0)} m` : "—", "route length"], [access.route ? `${(access.route.eta_s / 60).toFixed(1)} min` : "—", "drive time"]])}
      {access.route && <p className="plan-note">{access.route.road_m.toFixed(0)} m on road, {access.route.offroad_m.toFixed(0)} m off road.{Object.entries(access.route.snapped_m).filter(([, m]) => m > 0).map(([name, m]) => ` The ${name} was moved ${m.toFixed(1)} m to drivable ground.`).join("")}</p>}
      {rows(access.blocked.map((b) => ({ id: b.id, title: `Obstacle ${b.max_height_m.toFixed(1)} m high`, sub: `${fmtArea(b.area_m2)}${b.new_since_before ? " · new since the earlier flight" : ""}`, badge: b.id, tone: b.new_since_before ? "bad" : "warn", geo: b })), "No blocked roads.")}
      <div className="insp-actions">{packButton("access", true)}</div>
      <Notes notes={[`Roads: ${access.road_source}.`, ...(access.route?.notes ?? [])]} />
    </div>}
  </>;

  const floodTask = <>
    <div className="insp-setting"><NumberField label="Water rises by" unit="m" value={rise} min={0.1} max={30} step={0.5} onCommit={(v) => setRise(v ?? 2)} /></div>
    {pickButton("seed", "Click where the water comes in", "A river bank or the lowest street")}
    {drawing(["seed"])}
    {busy === "flood" && <p className="insp-busy"><span className="spinner" />Filling the ground from that point…</p>}
    {flood && <div className="insp-result">
      {verdict(flood.buildings.length ? "bad" : "warn", `${fmtArea(flood.flooded_m2)} under water, up to ${flood.max_depth_m.toFixed(1)} m deep. ${flood.buildings.length} building${flood.buildings.length === 1 ? "" : "s"} get wet.`)}
      {kpis([[fmtArea(flood.flooded_m2), "under water", "cool"], [`${flood.max_depth_m.toFixed(1)} m`, "deepest"], [fmtM3(flood.volume_m3), "water"], [String(flood.buildings.length), "buildings wet", flood.buildings.length ? "bad" : "good"]])}
      {rows(flood.buildings.map((b) => ({ id: b.id, title: `${b.id}`, sub: `${b.max_depth_m.toFixed(1)} m of water at the walls`, tone: "cool", geo: b })), "No buildings reached.")}
      <div className="insp-actions">{packButton("flood", true)}</div>
      <Notes notes={[`Water level ${flood.level_m.toFixed(2)} m. ${fmtArea(flood.isolated_low_ground_m2)} of low ground is not connected to your point and stays dry here.`, ...flood.notes]} />
    </div>}
  </>;

  const designUpload = <label className="button secondary full ops-upload"><Icon name="upload" size={15} />{busy === "design" ? "Reading the file…" : "Upload a design (LandXML, DXF, GeoTIFF or GLB)"}
    <input type="file" accept=".xml,.landxml,.dxf,.tif,.tiff,.glb" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); e.target.value = ""; }} /></label>;
  const surfaces = summary?.designs.filter((d) => !d.toLowerCase().endsWith(".glb")) ?? [];
  const models = summary?.designs.filter((d) => d.toLowerCase().endsWith(".glb")) ?? [];

  const cutfillTask = <>
    {designUpload}
    {surfaces.length > 0 ? <div className="insp-form">
      <label className="insp-field"><span>Design surface</span><select value={design} onChange={(e) => setDesign(e.target.value)}>{surfaces.map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
      <label className="insp-field"><span>Its coordinates</span><select value={designFrame} onChange={(e) => setDesignFrame(e.target.value as "scene" | "utm")}><option value="scene">Same as the scan</option>{geo && <option value="utm">UTM (from the GPS)</option>}</select></label>
    </div> : <p className="plan-note">Upload the finished-ground design from your CAD or survey software.</p>}
    {beforeField(false, "With an earlier flight, progress per zone since then is shown too.")}
    {runButton(before ? "Compare with design and show progress" : "Compare with design", "Comparing with the design…", "cutfill", !design, () => void run("cutfill", () => opsApi.cutfill(scene, design, designFrame, before || null), setCutfill))}
    {cutfill && <div className="insp-result">
      {verdict(cutfill.cut_m3 + cutfill.fill_m3 > 0 ? "warn" : "good", `${fmtM3(cutfill.cut_m3)} still to dig and ${fmtM3(cutfill.fill_m3)} still to fill to reach the design.`)}
      {kpis([[fmtM3(cutfill.cut_m3), "to dig (cut)", "warn"], [fmtM3(cutfill.fill_m3), "to fill", "cool"], [fmtM3(cutfill.net_m3), "net"], [pct(cutfill.design_observed_fraction), "design area seen"]])}
      {cutfill.zones?.map((z) => <div key={z.zone} className="ops-progress"><span>{z.zone}<b>{z.progress_pct == null ? "—" : `${z.progress_pct.toFixed(0)}%`}</b></span><i><em style={{ width: `${Math.max(0, Math.min(100, z.progress_pct ?? 0))}%` }} /></i><small>{fmtM3(z.remaining_m3)} left{z.start_m3 !== undefined ? ` of ${fmtM3(z.start_m3)}` : ""}</small></div>)}
      <div className="insp-actions">{packButton("cutfill", true)}</div>
      <Notes notes={[`Cut ± ${cutfill.cut_sigma_m3.toFixed(1)} m³, fill ± ${cutfill.fill_sigma_m3.toFixed(1)} m³. Design in ${cutfill.design_frame}.`, ...cutfill.notes]} />
    </div>}
  </>;

  const deviationTask = <>
    {designUpload}
    {models.length > 0 ? <div className="insp-form">
      <label className="insp-field"><span>Design model</span><select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
      <label className="insp-field"><span>Its coordinates</span><select value={modelFrame} onChange={(e) => setModelFrame(e.target.value as "scene" | "enu")}><option value="scene">Same as the scan</option>{geo && <option value="enu">East-north-up at the GPS origin</option>}</select></label>
      <div className="insp-setting"><NumberField label="Allowed difference" unit="m" value={devTol} min={0.001} max={1} step={0.005} onCommit={(v) => setDevTol(v ?? 0.05)} /></div>
    </div> : <p className="plan-note">Upload the design as a GLB exported from your BIM or CAD tool.</p>}
    {runButton("Compare scan with model", "Measuring the difference…", "deviation", !model, () => void run("deviation", () => opsApi.deviation(scene, model, modelFrame, devTol), setDeviation))}
    {deviation && <div className="insp-result">
      {verdict((deviation.within_tolerance_pct ?? 0) >= 90 ? "good" : "warn", deviation.within_tolerance_pct !== null ? `${deviation.within_tolerance_pct.toFixed(0)}% of what was built is within ±${(deviation.tolerance_m * 100).toFixed(0)} cm of the design.` : "Too little of the model was scanned to compare.")}
      {kpis([[deviation.within_tolerance_pct !== null ? `${deviation.within_tolerance_pct.toFixed(0)}%` : "—", `within ±${(deviation.tolerance_m * 100).toFixed(0)} cm`, "good"], [deviation.proud_pct !== null ? `${deviation.proud_pct.toFixed(0)}%` : "—", "sticks out", "warn"], [deviation.short_pct !== null ? `${deviation.short_pct.toFixed(0)}%` : "—", "falls short", "cool"], [deviation.rms_m !== null ? `${(deviation.rms_m * 100).toFixed(1)} cm` : "—", "typical difference"]])}
      <Notes notes={[`${deviation.scan_points_compared.toLocaleString()} scan points compared (${deviation.scan_points_unrelated.toLocaleString()} near the model but not on it). Sampling error about ${(deviation.sampling_error_m * 100).toFixed(1)} cm. ${deviation.basis}.`]} />
    </div>}
  </>;

  const corridorTask = <>
    <ol className="insp-steps" aria-label="Steps">
      {[["Draw the line", line.length >= 2], ["Place posts", posts.length > 0], ["Check coverage", !!corridor]].map(([label, done], i) => <li key={String(label)} className={done ? "done" : (i === 0 || (i === 1 && line.length >= 2) || (i === 2 && posts.length > 0)) ? "now" : ""}><span>{done ? <Icon name="check" size={12} /> : i + 1}</span>{label}</li>)}
    </ol>
    {pickButton("line", line.length ? `Redraw the line (${line.length} points)` : "Draw the border or fence line", "Click along it, then press Finish")}
    <div className="insp-form ops-post-settings"><div className="insp-setting"><NumberField label="Post eye height" unit="m" value={postHeight} min={0} max={100} step={0.5} onCommit={(v) => setPostHeight(v ?? 4)} /></div>
      <div className="insp-setting"><NumberField label="How far a post can see" unit="m" value={postRange} min={10} max={20000} step={50} onCommit={(v) => setPostRange(v ?? 600)} /></div></div>
    {pickButton("post", posts.length ? `Add more posts (${posts.length} placed)` : "Place observation posts", "One click per post")}
    {drawing(["line", "post"])}
    {posts.length > 0 && <div className="insp-chips">{posts.map((p, i) => <button key={p.id} onClick={() => setPosts(posts.filter((_, k) => k !== i))} data-tip="Remove this post">{p.id} · {p.height_m} m<Icon name="close" size={11} /></button>)}</div>}
    {runButton("Check what the posts can see", "Tracing sight lines…", "corridor", line.length < 2 || !posts.length, () => void run("corridor", () => opsApi.corridor(scene, line, posts), setCorridor), "eye")}
    {corridor && <div className="insp-result">
      {verdict(corridor.covered_fraction > 0.9 ? "good" : "warn", `The posts see ${pct(corridor.covered_fraction)} of the ${corridor.line_length_m.toFixed(0)} m line. ${corridor.blind_stretches.length} blind stretch${corridor.blind_stretches.length === 1 ? "" : "es"}.`)}
      <div className="mission-profile ops-profile" aria-label="Ground height along the line; red where no post can see">
        {corridor.profile.z.map((z, i) => { const zs = corridor.profile.z.filter((v): v is number => v !== null); const lo = Math.min(...zs), hi = Math.max(...zs); return <i key={i} className={z === null ? "gap" : corridor.profile.covered[i] ? "" : "hot2"} style={{ height: z === null ? "4%" : `${20 + 80 * ((z - lo) / Math.max(0.5, hi - lo))}%` }} />; })}
      </div>
      <div className="plan-legend"><span><i style={{ background: "hsl(218 8% 57% / .6)" }} />Seen by a post</span><span><i style={{ background: "rgb(255,90,80)" }} />Blind</span><span>Ground height along the line</span></div>
      {kpis([[pct(corridor.covered_fraction), "of the line seen", corridor.covered_fraction > 0.9 ? "good" : "warn"], [`${corridor.line_length_m.toFixed(0)} m`, "line length"], [String(corridor.blind_stretches.length), "blind stretches", corridor.blind_stretches.length ? "bad" : "good"], [`${corridor.profile.climb_m.toFixed(0)} m`, "total climb"]])}
      {rows(corridor.blind_stretches.map((b, i) => ({ id: String(i), title: `${b.from_m.toFixed(0)}–${b.to_m.toFixed(0)} m along the line`, sub: `${b.length_m.toFixed(0)} m blind · ${b.unobserved_ground ? "the drone never saw this ground" : "hidden by terrain or buildings"}`, tone: "bad", geo: b })), "No blind stretches.")}
      <div className="insp-actions">{packButton("corridor", true)}</div>
      <Notes notes={corridor.notes} />
    </div>}
  </>;

  const tilesTask = <>
    <p className="insp-lead">Cut the whole site into square tiles of height map (DSM) and point cloud (LAS) for GIS tools. {geo ? "Tiles are in UTM from the GPS." : "No GPS fit, so tiles stay in scene coordinates."}</p>
    <div className="insp-setting"><NumberField label="Tile size" unit="m" value={tileSize} min={20} max={5000} step={50} onCommit={(v) => setTileSize(v ?? 250)} /></div>
    {runButton("Make and download tiles", "Cutting tiles…", "tiles", false, () => void tiles(), "download")}
    <p className="plan-note">Plain GeoTIFF and LAS, not cloud-optimised formats.</p>
  </>;

  const planLink = (text: string) => <>
    <p className="insp-lead">{text}</p>
    <button className="button primary full" onClick={onPlan}><Icon name="building" size={15} />Open Plan</button>
  </>;

  const body: Record<Task, React.ReactNode> = {
    firstmap: firstmapTask, change: changeTask, volume: volumeTask(app === "construction" ? "stockpile" : "debris"), damage: damageTask, detections: detectionsTask,
    access: accessTask, flood: floodTask, relief: planLink("Place tents, medical tents, water bladders, generators, helipads and staging areas from the Plan catalogue, with hazard zones drawn as plots. They stay proposals and never change the scan."),
    cutfill: cutfillTask, deviation: deviationTask, logistics: planLink("Place tower cranes, site offices and material yards in Plan. Each crane shows its jib circle and lists every scanned structure the jib would hit."),
    corridor: corridorTask, tiles: tilesTask,
  };
  const overlayKey: Partial<Record<Task, Key>> = { change: "change", volume: "volume", damage: "damage", detections: "detections", access: "access", flood: "flood", cutfill: "cutfill", deviation: "deviation", corridor: "corridor" };
  const legendOf: Partial<Record<Key, [string, string][]>> = {
    change: [["rgb(214,60,50)", "Higher now"], ["rgb(50,110,220)", "Lower now"]],
    damage: [["rgb(80,200,110)", "Intact"], ["rgb(240,200,70)", "Partly damaged"], ["rgb(230,80,70)", "Collapsed"]],
    flood: flood ? Object.keys(flood.depth_bands_m2).map((band, i) => [["rgb(110,170,240)", "rgb(50,110,210)", "rgb(20,60,170)"][i], band] as [string, string]) : [],
    cutfill: [["rgb(214,70,60)", "Above design (dig)"], ["rgb(60,110,214)", "Below design (fill)"], ["rgb(80,200,110)", "On grade"]],
    deviation: [["rgb(214,60,50)", "Sticks out"], ["rgb(50,110,220)", "Falls short"], ["rgb(70,190,90)", "Within tolerance"]],
  };
  const shownTask = shown ? tasks.find((t) => overlayKey[t.id] === shown) : undefined;

  return <div className="plan-panel ops-panel">
    <section className="inspector-section insp-top">
      <div className="plan-view ops-apps" role="group" aria-label="Kind of work">{APPS.map((a) => <button key={a.key} className={app === a.key ? "active" : ""} aria-pressed={app === a.key} onClick={() => { setApp(a.key); setTask(TASKS[a.key][0].id); setShown(null); onTool(null); }}><Icon name={a.icon} size={14} />{a.label}</button>)}</div>
      {summary && <div className={`ops-frame ${geo ? "is-geo" : ""}`}><Icon name={geo ? "globe" : "info"} size={14} /><span>{geo ? "GPS-referenced: results carry grid references and export to Google Earth" : "No GPS fit: results stay in scene coordinates"}</span><em>{summary.grid.extent_m[0].toFixed(0)} × {summary.grid.extent_m[1].toFixed(0)} m</em></div>}
      <div className="insp-tasks" role="tablist" aria-label="What do you want to do?">{tasks.map((t) => <button key={t.id} role="tab" aria-selected={active.id === t.id} className={active.id === t.id ? "on" : ""} onClick={() => { setTask(t.id); onTool(null); if (overlayKey[t.id] && overlays[overlayKey[t.id]!]?.length) setShown(overlayKey[t.id]!); }}>
        <Icon name={t.icon} size={18} /><span>{t.label}</span>{results[t.id] && <i className="insp-has" aria-label="Has a result" />}</button>)}</div>
    </section>
    <section className="inspector-section insp-body" key={`${app}-${active.id}`}>
      <header className="insp-head"><div><h3>{active.label}</h3><p>{active.what}</p></div>{overlayKey[active.id] && showToggle(overlayKey[active.id]!)}</header>
      {error && <p className="form-error" role="alert">{error}</p>}
      {body[active.id]}
    </section>

    {/* What the colours on the model mean, next to the model. */}
    {shown && !tool && shownTask && <OnViewer kicker={`On the model: ${shownTask.label}`}>
      <div className="ops-hud">{(legendOf[shown] ?? []).length > 0 && <div className="plan-legend">{legendOf[shown]!.map(([colour, label]) => <span key={label}><i style={{ background: colour }} />{label}</span>)}</div>}
        <button className="button ghost small" onClick={() => setShown(null)}><Icon name="eye" size={14} />Hide</button></div>
    </OnViewer>}
  </div>;
}

type RowItem = { id: string; title: string; sub?: string; badge?: string; tone?: string; geo?: { mgrs?: string; viewer?: [number, number, number] } };
function RowList({ rows, onFocus }: { rows: RowItem[]; onFocus?: (point: [number, number, number]) => void }) {
  const [all, setAll] = useState(false);
  const list = all ? rows : rows.slice(0, 5);
  return <div className="ops-rows">
    <ul>{list.map((r) => <li key={r.id} className={r.tone ? `tone-${r.tone}` : ""}>
      <i />
      <span><strong>{r.title}{r.badge && <em className={`ops-grade ${r.tone ?? "muted"}`}>{r.badge}</em>}</strong>{r.sub && <small>{r.sub}</small>}{r.geo?.mgrs && <small className="ops-mgrs">{r.geo.mgrs}</small>}</span>
      {r.geo?.viewer && onFocus && <button className="button ghost small" onClick={() => onFocus(r.geo!.viewer!)} data-tip="Fly the view to this spot"><Icon name="target" size={14} />Go to</button>}
    </li>)}</ul>
    {rows.length > 5 && <button className="text-button" onClick={() => setAll(!all)}>{all ? "Show fewer" : `Show all ${rows.length}`}</button>}
  </div>;
}
