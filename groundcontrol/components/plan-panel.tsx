"use client";

import { useEffect, useRef, useState } from "react";
import {
  plan, TOOL_COPY,
  type BuildingParams, type Catalogue, type ClipParams, type Coordinates, type ExistingBuilding, type ExportFormat, type ObjectParams,
  type FacadeResult, type PlanFeature, type PlanState, type PlanTool, type ProposalIndex, type RoadParams, type ShadowResult, type ShadowSettings,
  type ZoneParams, type ZoneRules,
} from "@/lib/plan";
import { Icon, type IconName } from "./studio-icons";
import "./plan-panel.css";

export type PlanView = "proposal" | "existing" | "flicker" | "swipe" | "side";
export type ShadowPanelState = { on: boolean; settings: ShadowSettings; result: ShadowResult | null; busy: boolean; error: string; stale: boolean };

type Props = {
  scene: string; state: PlanState | null; index: ProposalIndex | null; catalogue: Catalogue;
  selected: string | null; tool: PlanTool | null; points: number; view: PlanView; busy: boolean;
  message: string; ready: boolean; canUndo: boolean; canRedo: boolean;
  objectItem: string; arraySpacing: number; arrayOffset: number;
  onProposal: (id: string) => void; onCreate: (name: string, source?: string) => void;
  onRename: (name: string) => void; onDeleteProposal: () => void;
  onTool: (tool: PlanTool | null) => void; onFinish: () => void; onUndoPoint: () => void;
  onObjectItem: (item: string) => void; onArray: (spacing: number, offset: number) => void;
  onUpdate: (feature: PlanFeature) => void; onDeleteFeature: (id: string) => void;
  onSelect: (id: string | null) => void; onView: (view: PlanView) => void;
  onUndo: () => void; onRedo: () => void; onExport: (format: ExportFormat) => void; onDemolish: (building: ExistingBuilding) => void;
  onImport: (file: File, epsg?: number) => void;
  onInferred: (inferred: boolean, basis: string) => void;
  shadow: ShadowPanelState; onShadowToggle: (on: boolean) => void; onShadowSettings: (settings: ShadowSettings) => void; onShadowRun: () => void;
  facades: FacadeResult | null; facadesBusy: boolean; showFacades: boolean; onFacades: () => void; onShowFacades: (on: boolean) => void;
};

const VIEWS: [PlanView, string, string][] = [
  ["existing", "Existing", "The scan as it is"], ["proposal", "Proposed", "The scheme drawn over the scan"],
  ["flicker", "Flicker", "Alternate every 1.4 s"], ["swipe", "Swipe", "Drag the divider across one view"],
  ["side", "Split", "Existing and proposed side by side, cameras locked together"],
];
const EXPORTS: [ExportFormat, string][] = [
  ["geojson", "GeoJSON · footprints"], ["cityjson", "CityJSON 2.0 · 3D city model"], ["dxf", "DXF · CAD drawing"], ["3dtiles", "3D Tiles · web globe (zip)"],
];
const SHADOW_KEY: [string, number[], string][] = [
  ["Shadow", [28, 36, 64], "instant"], ["New shadow from the scheme", [150, 60, 190], "instant"], ["Sunlit again", [250, 214, 90], "instant"],
  ["Loses ½–1 h sun", [252, 220, 120], "day"], ["1–2 h", [248, 160, 70], "day"], ["2–3 h", [232, 96, 60], "day"], ["3 h +", [180, 40, 60], "day"],
];

function ShadowStudy({ shadow, georeferenced, disabled, onToggle, onSettings, onRun }: {
  shadow: ShadowPanelState; georeferenced: boolean; disabled: boolean;
  onToggle: (on: boolean) => void; onSettings: (s: ShadowSettings) => void; onRun: () => void;
}) {
  const s = shadow.settings;
  const set = (patch: Partial<ShadowSettings>) => onSettings({ ...s, ...patch });
  const r = shadow.result;
  return <section className="inspector-section">
    <div className="plan-shadow-head"><div className="section-label"><span>SHADOW STUDY</span></div>
      <label className="plan-switch"><input type="checkbox" checked={shadow.on} disabled={disabled} onChange={(e) => onToggle(e.target.checked)} />Show</label></div>
    {shadow.on && <>
      <div className="plan-view" role="group" aria-label="Shadow study mode">
        {(["instant", "day"] as const).map((m) => <button key={m} className={s.mode === m ? "active" : ""} aria-pressed={s.mode === m} onClick={() => set({ mode: m })}>{m === "instant" ? "One moment" : "Sun hours over a day"}</button>)}
      </div>
      <div className="plan-shadow-grid">
        <label className="plan-field"><span>Date</span><input type="date" value={s.date} onChange={(e) => e.target.value && set({ date: e.target.value })} /></label>
        {s.mode === "instant"
          ? <label className="plan-field"><span>Local time</span><input type="time" value={s.time} onChange={(e) => e.target.value && set({ time: e.target.value })} /></label>
          : <label className="plan-field"><span>Window</span><select value={`${s.start}-${s.end}`} onChange={(e) => { const [a, b] = e.target.value.split("-").map(Number); set({ start: a, end: b }); }}>
            {[[9, 15], [8, 16], [10, 14], [6, 18]].map(([a, b]) => <option key={a} value={`${a}-${b}`}>{a}:00 – {b}:00</option>)}</select></label>}
        <NumberField label="UTC offset" unit="h" value={s.utc_offset} min={-14} max={14} step={0.25} onCommit={(v) => set({ utc_offset: v ?? 0 })} />
        {s.mode === "instant" && <label className="plan-field"><span>Scrub</span><input type="range" min={6 * 60} max={19 * 60} step={15} aria-label="Time of day"
          value={Number(s.time.slice(0, 2)) * 60 + Number(s.time.slice(3, 5))}
          onChange={(e) => { const m = Number(e.target.value); set({ time: `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}` }); }} /></label>}
        {!georeferenced && <>
          <NumberField label="Site latitude" unit="°" value={s.lat} min={-90} max={90} step={0.0001} onCommit={(v) => set({ lat: v })} />
          <NumberField label="Site longitude" unit="°" value={s.lon} min={-180} max={180} step={0.0001} onCommit={(v) => set({ lon: v })} />
          <NumberField label="North from −Z" unit="°" value={s.north_deg} min={-180} max={180} step={1} onCommit={(v) => set({ north_deg: v ?? 0 })} />
        </>}
      </div>
      {s.mode === "day" && <button className="button primary small full plan-run" disabled={disabled || shadow.busy} onClick={onRun}><Icon name="play" size={13} />{shadow.busy ? "Counting sun hours…" : shadow.stale ? "Re-run for the current scheme" : "Run day study"}</button>}
      {shadow.busy && s.mode === "instant" && <p className="plan-note">Casting shadows…</p>}
      {shadow.error && <p className="plan-message" role="alert">{shadow.error}</p>}
      {r && <>
        {r.mode === "instant" && typeof r.sun.elevation_deg === "number" && <div className="plan-sun"><span>SUN {r.sun.elevation_deg.toFixed(1)}° UP</span><span>AZ {r.sun.azimuth_deg?.toFixed(0)}°</span></div>}
        <div className="plan-legend">{SHADOW_KEY.filter(([, , m]) => m === r.mode).map(([label, c]) => <span key={label}><i style={{ background: `rgb(${c.join(",")})` }} />{label}</span>)}</div>
        {r.metrics.length > 0 && <table className="plan-table"><thead><tr><th>Metric</th><th>Existing</th><th>Proposed</th><th>Change</th></tr></thead>
          <tbody>{r.metrics.map((row) => <tr key={row.metric}><td>{row.metric}<small>{row.unit}</small></td><td>{fmt(row.existing)}</td><td>{fmt(row.proposal)}</td><td className={row.change > 0 ? "up" : row.change < 0 ? "down" : ""}>{row.change > 0 ? "+" : ""}{fmt(row.change)}</td></tr>)}</tbody></table>}
        {r.notes.map((note) => <p className="plan-note" key={note}>{note}</p>)}
      </>}
    </>}
  </section>;
}

const TOOLS: { tool: PlanTool; icon: IconName }[] = [
  { tool: "road", icon: "arrow" }, { tool: "building", icon: "cube" }, { tool: "zone", icon: "grid" },
  { tool: "clip", icon: "trash" }, { tool: "object", icon: "pin" }, { tool: "array", icon: "layers" },
];
const TYPE_LABEL: Record<string, string> = { road: "Roads", building: "Buildings", zone: "Plots", object: "Objects", clip: "Demolitions" };
const RULE_LABEL: Record<string, string> = { max_height_m: "Height limit", max_fsi: "FSI / FAR", max_coverage_pct: "Ground coverage", setback_m: "Setback", max_floors: "Floor limit", crane_swing_clearance: "Crane jib clearance (highest structure vs jib minus clearance)" };
const fmt = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });

/** A numeric field that commits on blur or Enter, never on every keystroke. */
export function NumberField({ label, value, min, max, step = 1, unit, disabled, onCommit }: {
  label: string; value: number | undefined; min: number; max: number; step?: number; unit?: string;
  disabled?: boolean; onCommit: (value: number | undefined) => void;
}) {
  const [text, setText] = useState(value === undefined ? "" : String(value));
  // A new value from the backend replaces the draft text (adjusting state during render,
  // which React prefers to an effect that re-renders twice).
  const [shown, setShown] = useState(value);
  if (shown !== value) { setShown(value); setText(value === undefined ? "" : String(value)); }
  const commit = () => {
    if (text.trim() === "") { if (value !== undefined) onCommit(undefined); return; }
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) { setText(value === undefined ? "" : String(value)); return; }
    const clamped = Math.min(max, Math.max(min, parsed));
    setText(String(clamped));
    if (clamped !== value) onCommit(clamped);
  };
  return <label className="plan-field">
    <span>{label}</span>
    <span className="plan-input">
      <input type="number" inputMode="decimal" min={min} max={max} step={step} value={text} disabled={disabled}
        onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
      {unit && <small>{unit}</small>}
    </span>
  </label>;
}

export function Select<T extends string>({ label, value, options, disabled, onChange }: {
  label: string; value: T; options: [T, string][]; disabled?: boolean; onChange: (value: T) => void;
}) {
  return <label className="plan-field"><span>{label}</span>
    <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
      {options.map(([v, text]) => <option key={v} value={v}>{text}</option>)}
    </select></label>;
}

function FeatureForm({ feature, onUpdate, disabled, metrics }: { feature: PlanFeature; onUpdate: (f: PlanFeature) => void; disabled: boolean; metrics?: Record<string, unknown> }) {
  const set = (patch: Record<string, unknown>) => onUpdate({ ...feature, params: { ...feature.params, ...patch } as PlanFeature["params"] });
  if (feature.type === "road") {
    const p = feature.params as RoadParams;
    return <div className="plan-form">
      <NumberField label="Carriageway width" unit="m" value={p.width_m} min={2.5} max={60} step={0.5} disabled={disabled} onCommit={(v) => set({ width_m: v ?? 7 })} />
      <NumberField label="Lanes" value={p.lanes} min={1} max={12} disabled={disabled} onCommit={(v) => set({ lanes: v ?? 2 })} />
      <NumberField label="Footpath each side" unit="m" value={p.footpath_m} min={0} max={8} step={0.25} disabled={disabled} onCommit={(v) => set({ footpath_m: v ?? 0 })} />
      <NumberField label="Median" unit="m" value={p.median_m} min={0} max={20} step={0.5} disabled={disabled} onCommit={(v) => set({ median_m: v ?? 0 })} />
      <Select label="Vertical alignment" value={p.mode} disabled={disabled} options={[["drape", "Follow ground"], ["graded", "Graded profile"]]} onChange={(mode) => set({ mode })} />
      {p.mode === "graded" && <NumberField label="Max grade" unit="%" value={p.max_grade_pct} min={0.5} max={20} step={0.5} disabled={disabled} onCommit={(v) => set({ max_grade_pct: v ?? 6 })} />}
      <Select label="Surface" value={p.surface} disabled={disabled} options={[["asphalt", "Asphalt"], ["concrete", "Concrete"], ["gravel", "Gravel"]]} onChange={(surface) => set({ surface })} />
    </div>;
  }
  if (feature.type === "building") {
    const p = feature.params as BuildingParams;
    return <div className="plan-form">
      <div className="plan-stepper"><span>Floors</span>
        <button className="icon-button" aria-label="One floor fewer" disabled={disabled || p.floors <= 1} onClick={() => set({ floors: p.floors - 1 })}>−</button>
        <strong>{p.floors}</strong>
        <button className="icon-button" aria-label="One floor more" disabled={disabled || p.floors >= 200} onClick={() => set({ floors: p.floors + 1 })}>+</button>
      </div>
      <NumberField label="Floor height" unit="m" value={p.floor_height_m} min={2.4} max={8} step={0.1} disabled={disabled} onCommit={(v) => set({ floor_height_m: v ?? 3.2 })} />
      <Select label="Use" value={p.use} disabled={disabled} options={[["residential", "Residential"], ["commercial", "Commercial"], ["mixed", "Mixed use"], ["institutional", "Institutional"], ["industrial", "Industrial"]]} onChange={(use) => set({ use })} />
      <Select label="Roof" value={p.roof} disabled={disabled} options={[["flat", "Flat"], ["gable", "Gable"], ["hip", "Hip"]]} onChange={(roof) => set({ roof })} />
      {p.roof !== "flat" && <NumberField label="Roof pitch" unit="°" value={p.roof_pitch_deg} min={5} max={60} disabled={disabled} onCommit={(v) => set({ roof_pitch_deg: v ?? 30 })} />}
      {p.roof !== "flat" && p.footprint.length !== 4 && <p className="plan-note">Pitched roofs need a four-cornered footprint; this one is drawn flat.</p>}
      <MoveRotate feature={feature} points={p.footprint} disabled={disabled} onPoints={(footprint) => set({ footprint })} />
    </div>;
  }
  if (feature.type === "zone") {
    const p = feature.params as ZoneParams;
    const rule = (key: keyof ZoneRules) => (v: number | undefined) => {
      const rules = { ...p.rules };
      if (v === undefined) delete rules[key]; else rules[key] = v;
      set({ rules });
    };
    return <div className="plan-form">
      <p className="plan-note">Leave a rule empty to not check it. Values come from the local development-control regulations.</p>
      <NumberField label="Max building height" unit="m" value={p.rules.max_height_m} min={1} max={1000} step={0.5} disabled={disabled} onCommit={rule("max_height_m")} />
      <NumberField label="Max floors" value={p.rules.max_floors} min={1} max={200} disabled={disabled} onCommit={rule("max_floors")} />
      <NumberField label="Max FSI / FAR" value={p.rules.max_fsi} min={0.05} max={30} step={0.05} disabled={disabled} onCommit={rule("max_fsi")} />
      <NumberField label="Max ground coverage" unit="%" value={p.rules.max_coverage_pct} min={1} max={100} disabled={disabled} onCommit={rule("max_coverage_pct")} />
      <NumberField label="Min setback" unit="m" value={p.rules.setback_m} min={0} max={100} step={0.5} disabled={disabled} onCommit={rule("setback_m")} />
    </div>;
  }
  if (feature.type === "object") {
    const p = feature.params as ObjectParams;
    return <div className="plan-form">
      <NumberField label="Heading" unit="°" value={p.yaw_deg} min={-360} max={360} step={5} disabled={disabled} onCommit={(v) => set({ yaw_deg: v ?? 0 })} />
      <NumberField label="Scale" unit="×" value={p.scale} min={0.1} max={10} step={0.1} disabled={disabled} onCommit={(v) => set({ scale: v ?? 1 })} />
      {p.item === "tower_crane" && <>
        <NumberField label="Jib radius" unit="m" value={p.jib_radius_m ?? 40} min={5} max={90} step={1} disabled={disabled} onCommit={(v) => set({ jib_radius_m: v ?? 40 })} />
        <NumberField label="Clearance under jib" unit="m" value={p.clearance_m ?? 3} min={0} max={20} step={0.5} disabled={disabled} onCommit={(v) => set({ clearance_m: v ?? 3 })} />
        {metrics && <p className={Number(metrics.swing_conflict_m2) > 0 ? "plan-message" : "plan-note"}>
          {Number(metrics.swing_conflict_m2) > 0
            ? `Jib swing clashes: ${metrics.swing_conflict_m2} m² of scanned structure reaches within ${metrics.clearance_m} m of the jib (highest ${metrics.swing_highest_obstacle_m} m above the crane's base; jib at ${metrics.jib_height_m} m).`
            : `Jib swing clear: nothing scanned within ${metrics.clearance_m} m of the jib at ${metrics.jib_height_m} m.`}
          {Number(metrics.swing_unobserved_pct) > 0 ? ` ${metrics.swing_unobserved_pct}% of the swing circle was never observed and is not checked.` : ""}</p>}
      </>}
      <MoveRotate feature={feature} points={[p.position]} disabled={disabled} rotate={false} onPoints={([position]) => set({ position })} />
    </div>;
  }
  const p = feature.params as ClipParams;
  return <div className="plan-form"><p className="plan-note">The scanned model inside this outline is hidden in the proposal view and counted as demolished in the before/after table. {p.polygon.length} corners.</p>
    <MoveRotate feature={feature} points={p.polygon} disabled={disabled} onPoints={(polygon) => set({ polygon })} /></div>;
}

/** Nudge and rotate any footprint in metres/degrees: exact, keyboard-reachable edits. */
function MoveRotate({ points, disabled, rotate = true, onPoints }: {
  feature: PlanFeature; points: [number, number][]; disabled: boolean; rotate?: boolean; onPoints: (p: [number, number][]) => void;
}) {
  const [step, setStep] = useState(1);
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cz = points.reduce((s, p) => s + p[1], 0) / points.length;
  const move = (dx: number, dz: number) => onPoints(points.map(([x, z]) => [x + dx, z + dz]));
  const turn = (deg: number) => {
    const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
    onPoints(points.map(([x, z]) => [cx + (x - cx) * c - (z - cz) * s, cz + (x - cx) * s + (z - cz) * c]));
  };
  return <div className="plan-move">
    <div className="plan-move-pad" role="group" aria-label="Move">
      <span /><button className="button secondary small" disabled={disabled} aria-label="Move −Z" onClick={() => move(0, -step)}>↑</button><span />
      <button className="button secondary small" disabled={disabled} aria-label="Move −X" onClick={() => move(-step, 0)}>←</button>
      <select aria-label="Move step" value={step} onChange={(e) => setStep(Number(e.target.value))}>{[0.25, 1, 5, 10].map((v) => <option key={v} value={v}>{v} m</option>)}</select>
      <button className="button secondary small" disabled={disabled} aria-label="Move +X" onClick={() => move(step, 0)}>→</button>
      <span /><button className="button secondary small" disabled={disabled} aria-label="Move +Z" onClick={() => move(0, step)}>↓</button><span />
    </div>
    {rotate && <div className="plan-rotate"><button className="button secondary small" disabled={disabled} onClick={() => turn(-15)}>⟲ 15°</button><button className="button secondary small" disabled={disabled} onClick={() => turn(15)}>⟳ 15°</button></div>}
  </div>;
}

function Metrics({ feature, state }: { feature: PlanFeature; state: PlanState }) {
  const m = state.evaluation.features[feature.id]?.metrics ?? {};
  const rows: [string, string][] = [];
  const add = (label: string, value: unknown, unit = "", digits = 1) => { if (typeof value === "number") rows.push([label, `${fmt(value, digits)}${unit ? " " + unit : ""}`]); };
  if (feature.type === "road") {
    add("Length", m.length_m, "m"); add("Total width", m.width_total_m, "m"); add("Paved area", m.total_area_m2, "m²");
    add("Steepest grade", m.max_grade_pct, "%"); add("Cut", m.cut_m3, "m³"); add("Fill", m.fill_m3, "m³");
    if (typeof m.profile_bridged_m === "number" && m.profile_bridged_m > 0) add("Bridged over unscanned ground", m.profile_bridged_m, "m");
    add("Buildings in the way", m.existing_buildings_hit, "", 0); add("Trees removed", m.trees_removed, "", 0);
  } else if (feature.type === "building") {
    add("Footprint", m.footprint_area_m2, "m²"); add("Gross floor area", m.gfa_m2, "m²"); add("Height", m.height_m, "m");
    add("Volume", m.volume_m3, "m³"); add("Buildings overlapped", m.existing_buildings_hit, "", 0); add("Trees removed", m.trees_removed, "", 0);
  } else if (feature.type === "zone") {
    add("Plot area", m.area_m2, "m²"); add("Buildings inside", m.buildings, "", 0); add("Ground coverage", m.coverage_pct, "%"); add("FSI", m.fsi, "", 2);
  } else if (feature.type === "clip") add("Area", m.area_m2, "m²");
  else add("Height", m.height_m, "m");
  if (typeof m.ground_supported_fraction === "number" && m.ground_supported_fraction < 0.9)
    rows.push(["Ground under it", `${Math.round(m.ground_supported_fraction * 100)}% measured`]);
  return <div className="plan-metrics">{rows.map(([k, v]) => <div className="datum-row" key={k}><span>{k}</span><span>{v}</span></div>)}</div>;
}

function Location({ scene, feature, state }: { scene: string; feature: PlanFeature; state: PlanState }) {
  const [found, setFound] = useState<{ key: string; value: Coordinates } | null>(null);
  const derived = state.evaluation.features[feature.id];
  const ring = derived?.footprint ?? [];
  const cx = ring.length ? ring.reduce((s, p) => s + p[0], 0) / ring.length : 0;
  const cz = ring.length ? ring.reduce((s, p) => s + p[1], 0) / ring.length : 0;
  const base = typeof derived?.metrics.base_y === "number" ? derived.metrics.base_y : 0;
  const geo = state.frame.status === "georeferenced";
  const key = `${cx.toFixed(3)},${base.toFixed(3)},${cz.toFixed(3)}`;
  useEffect(() => {
    if (!geo || !ring.length) return;
    let live = true;
    plan.coords(scene, [cx, base, cz]).then((value) => { if (live) setFound({ key, value }); }).catch(() => {});
    return () => { live = false; };
  }, [scene, geo, cx, cz, base, ring.length, key]);
  const where = found?.key === key ? found.value : null;
  if (!geo) return <div className="datum-row"><span>Location</span><span>Local {fmt(cx)} m, {fmt(-cz)} m · no GPS fit</span></div>;
  return <>
    <div className="datum-row"><span>MGRS</span><span><code>{where?.mgrs ?? "…"}</code></span></div>
    <div className="datum-row"><span>Lat / lon</span><span>{where?.lat_deg !== undefined ? `${where.lat_deg.toFixed(6)}, ${where.lon_deg?.toFixed(6)}` : "…"}</span></div>
  </>;
}

function ExchangeSection({ busy, locked, georeferenced, onExport, onImport }: {
  busy: boolean; locked: boolean; georeferenced: boolean; onExport: (format: ExportFormat) => void; onImport: (file: File, epsg?: number) => void;
}) {
  const [epsg, setEpsg] = useState<number | undefined>(undefined);
  const [format, setFormat] = useState<ExportFormat>("cityjson");
  const file = useRef<HTMLInputElement>(null);
  return <section className="inspector-section">
    <div className="section-label"><span>EXCHANGE</span></div>
    <div className="plan-export">
      <select aria-label="Export format" value={format} onChange={(e) => setFormat(e.target.value as ExportFormat)}>{EXPORTS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}</select>
      <button className="button secondary small" disabled={busy} onClick={() => onExport(format)}><Icon name="download" size={14} />Export</button>
    </div>
    <p className="plan-note">{georeferenced ? "Written in the scene's UTM zone (3D Tiles on the globe); heights are ellipsoidal." : "No GPS fit: files are in LOCAL metres and say so."}</p>
    <input ref={file} type="file" accept=".geojson,.json,.kml,.dxf,.zip" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onImport(f, epsg); e.target.value = ""; }} />
    <div className="plan-export"><NumberField label="Source CRS (EPSG, optional)" value={epsg} min={1024} max={999999} disabled={locked} onCommit={setEpsg} />
      <button className="button secondary small plan-import" disabled={locked} onClick={() => file.current?.click()}><Icon name="upload" size={14} />Import parcels…</button></div>
    <p className="plan-note">GeoJSON, KML, DXF or a zipped shapefile (.shp + .dbf + .prj) become plots; FSI, height, coverage and setback in their attributes become plot rules. A .prj or GeoJSON CRS is read automatically; give an EPSG code for a DXF in a projected system other than the scene&apos;s UTM zone.</p>
  </section>;
}

export function PlanPanel(props: Props) {
  const { scene, state, index, catalogue, selected, tool, points, view, busy, message, ready } = props;
  const [naming, setNaming] = useState<"new" | "rename" | null>(null);
  const [name, setName] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [spacing, setSpacing] = useState(props.arraySpacing);
  const [offset, setOffset] = useState(props.arrayOffset);
  const proposal = state?.proposal ?? null;
  const feature = proposal?.features.find((f) => f.id === selected) ?? null;
  const violations = state?.evaluation.violations ?? [];
  const violating = new Set(violations.map((v) => v.feature));
  const demolished = new Set(state?.evaluation.demolished ?? []);
  const locked = busy || !ready;
  const scale = state?.frame.scale_status;

  const submitName = () => {
    const value = name.trim();
    if (!value) return;
    if (naming === "rename") props.onRename(value); else props.onCreate(value);
    setNaming(null); setName("");
  };

  return <div className="plan-panel">
    <section className="inspector-section plan-head">
      <div className="section-label"><span>PLANNING SCHEME</span>
        <span className="plan-history">
          <button className="icon-button" title="Undo (Ctrl+Z)" aria-label="Undo" disabled={!props.canUndo || busy} onClick={props.onUndo}><Icon name="reset" size={14} /></button>
          <button className="icon-button plan-redo" title="Redo (Ctrl+Y)" aria-label="Redo" disabled={!props.canRedo || busy} onClick={props.onRedo}><Icon name="reset" size={14} /></button>
        </span>
      </div>
      {index && index.proposals.length > 0 && !naming && <div className="plan-scheme">
        <select aria-label="Active scheme" value={proposal?.id ?? ""} disabled={busy} onChange={(e) => props.onProposal(e.target.value)}>
          {index.proposals.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        <button className="icon-button" title="New scheme" aria-label="New scheme" disabled={busy} onClick={() => { setNaming("new"); setName(`Scheme ${String.fromCharCode(65 + (index.proposals.length % 26))}`); }}><Icon name="plus" size={15} /></button>
        <button className="icon-button" title="Duplicate scheme" aria-label="Duplicate scheme" disabled={busy || !proposal} onClick={() => proposal && props.onCreate(`${proposal.name} copy`, proposal.id)}><Icon name="file" size={15} /></button>
        <button className="icon-button" title="Rename scheme" aria-label="Rename scheme" disabled={busy || !proposal} onClick={() => { setNaming("rename"); setName(proposal?.name ?? ""); }}><Icon name="settings" size={15} /></button>
        <button className={`icon-button${confirmDelete ? " danger-armed" : ""}`} title={confirmDelete ? "Click again to delete" : "Delete scheme"} aria-label="Delete scheme" disabled={busy || !proposal} onClick={() => { if (confirmDelete) { props.onDeleteProposal(); setConfirmDelete(false); } else setConfirmDelete(true); }} onBlur={() => setConfirmDelete(false)}><Icon name="trash" size={15} /></button>
      </div>}
      {(naming || (index && !index.proposals.length)) && <form className="plan-name" onSubmit={(e) => { e.preventDefault(); if (!naming) setNaming("new"); submitName(); }}>
        {!naming && <p className="inspector-copy">Start a scheme to draw proposed roads, buildings and plots over this scan. The scan itself is never edited.</p>}
        <input aria-label="Scheme name" placeholder="Scheme A" value={naming ? name : name || ""} onChange={(e) => { if (!naming) setNaming("new"); setName(e.target.value); }} maxLength={80} />
        <button className="button primary small" type="submit" disabled={busy || !(naming ? name : name).trim()}>{naming === "rename" ? "Rename" : "Create scheme"}</button>
        {naming && <button className="text-button" type="button" onClick={() => { setNaming(null); setName(""); }}>Cancel</button>}
      </form>}
      {state && <div className={`plan-frame plan-frame-${state.frame.status}`}>
        <Icon name={state.frame.status === "georeferenced" ? "globe" : "info"} size={13} />
        <span>{state.frame.status === "georeferenced" ? "Georeferenced — exports in WGS84 and UTM" : scale === "metric" ? "Metric, local coordinates — no GPS fit" : scale === "estimated" ? "Estimated scale — sizes are approximate" : "Relative scale — sizes are not metres"}</span>
      </div>}
      {state && <div className="plan-hypothesis">
        <label className="plan-switch"><input type="checkbox" checked={!!state.proposal.inferred} disabled={busy}
          onChange={(e) => props.onInferred(e.target.checked, state.proposal.inferred_basis ?? "")} />Hypothesis (inferred reconstruction)</label>
        {state.proposal.inferred && <>
          <p className="plan-tag plan-tag-inferred">INFERRED — a reconstruction of what is not there, not a measurement</p>
          <input className="plan-title" aria-label="Basis of the hypothesis" key={state.proposal.id + (state.proposal.inferred_basis ?? "")} defaultValue={state.proposal.inferred_basis ?? ""}
            placeholder="Basis: 1910 photograph, comparison with the sister temple…" maxLength={1000} disabled={busy}
            onBlur={(e) => { if (e.target.value !== (state.proposal.inferred_basis ?? "")) props.onInferred(true, e.target.value); }} />
        </>}
      </div>}
    </section>

    {proposal && <>
      <section className="inspector-section">
        <div className="section-label"><span>DRAW</span><span>{proposal.features.length} FEATURES</span></div>
        <div className="plan-tools">{TOOLS.map(({ tool: t, icon }) => <button key={t} className={`plan-tool${tool === t ? " active" : ""}`} aria-pressed={tool === t} disabled={locked} onClick={() => props.onTool(tool === t ? null : t)}><Icon name={icon} size={16} /><span>{TOOL_COPY[t].label}</span></button>)}</div>
        {(tool === "object" || tool === "array") && <label className="plan-field plan-item"><span>Item</span>
          <select value={props.objectItem} onChange={(e) => props.onObjectItem(e.target.value)}>
            {Object.keys(catalogue).map((item) => <option key={item} value={item}>{item.replace(/_/g, " ")} · {catalogue[item].size[1]} m tall</option>)}
          </select></label>}
        {tool === "array" && <div className="plan-array">
          <NumberField label="Spacing" unit="m" value={spacing} min={1} max={500} onCommit={(v) => { setSpacing(v ?? 30); props.onArray(v ?? 30, offset); }} />
          <NumberField label="Side offset" unit="m" value={offset} min={-50} max={50} step={0.5} onCommit={(v) => { setOffset(v ?? 0); props.onArray(spacing, v ?? 0); }} />
        </div>}
        {tool && <div className="plan-drawing" role="status">
          <p>{TOOL_COPY[tool].hint}</p>
          {tool !== "object" && <div className="plan-drawing-actions">
            <span>{points} point{points === 1 ? "" : "s"}</span>
            <button className="text-button" disabled={!points} onClick={props.onUndoPoint}>Undo point</button>
            <button className="button primary small" disabled={points < TOOL_COPY[tool].min || busy} onClick={props.onFinish}><Icon name="check" size={13} />Finish</button>
            <button className="button secondary small" onClick={() => props.onTool(null)}>Cancel</button>
          </div>}
        </div>}
        {message && <p className="plan-message" role="status">{message}</p>}
      </section>

      <section className="inspector-section">
        <div className="section-label"><span>COMPARE</span></div>
        <div className="plan-view" role="group" aria-label="Compare existing and proposed">
          {VIEWS.map(([v, label, title]) => <button key={v} title={title} className={view === v ? "active" : ""} aria-pressed={view === v} onClick={() => props.onView(v)}>{label}</button>)}
        </div>
        {(view === "swipe" || view === "side") && <p className="plan-note">A second viewer shows the scan as it is; both cameras move together. Orbit in either half.</p>}
      </section>

      {feature && <section className="inspector-section plan-selected">
        <div className="section-label"><span>{feature.type.toUpperCase()} · SELECTED</span><button className="icon-button" aria-label="Deselect" onClick={() => props.onSelect(null)}><Icon name="close" size={13} /></button></div>
        <input className="plan-title" aria-label="Feature name" defaultValue={feature.name} key={feature.id + feature.name} maxLength={120} disabled={locked}
          onBlur={(e) => { const v = e.target.value.trim(); if (v && v !== feature.name) props.onUpdate({ ...feature, name: v }); }}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
        {violating.has(feature.id) && <div className="plan-violation-card">{violations.filter((v) => v.feature === feature.id).map((v, i) => <p key={i}><Icon name="info" size={12} />{RULE_LABEL[v.rule] ?? v.rule}: {fmt(v.value, 2)}{v.unit ? " " + v.unit : ""} against {v.rule === "setback_m" ? "at least" : "at most"} {fmt(v.limit, 2)}{v.unit ? " " + v.unit : ""}</p>)}</div>}
        <Metrics feature={feature} state={state!} />
        <Location scene={scene} feature={feature} state={state!} />
        {view !== "flicker" && !feature.locked && <p className="plan-hint-3d">In 3D: drag a corner, drag a <b>+</b> to add one, double-click a corner to remove it; the orange square (or the feature itself) moves it, the blue dot turns it. Hold Shift to snap.</p>}
        <FeatureForm feature={feature} disabled={locked || feature.locked} onUpdate={props.onUpdate} metrics={state?.evaluation.features[feature.id]?.metrics as Record<string, unknown> | undefined} />
        <div className="plan-feature-actions">
          <button className="button secondary small" disabled={locked} onClick={() => props.onUpdate({ ...feature, hidden: !feature.hidden })}><Icon name="eye" size={13} />{feature.hidden ? "Show" : "Hide"}</button>
          <button className="button danger small" disabled={locked} onClick={() => props.onDeleteFeature(feature.id)}><Icon name="trash" size={13} />Delete</button>
        </div>
      </section>}

      {violations.length > 0 && <section className="inspector-section">
        <div className="section-label"><span>RULE VIOLATIONS</span><span className="plan-count bad">{violations.length}</span></div>
        <ul className="plan-violations">{violations.map((v, i) => { const f = proposal.features.find((x) => x.id === v.feature); return <li key={i}><button onClick={() => props.onSelect(v.feature)}><strong>{RULE_LABEL[v.rule] ?? v.rule}</strong><small>{f?.name ?? "Plot"} · {fmt(v.value, 2)} / {v.rule === "setback_m" ? "min" : "max"} {fmt(v.limit, 2)}{v.unit ? " " + v.unit : ""}</small></button></li>; })}</ul>
      </section>}

      {proposal.features.length > 0 && <section className="inspector-section">
        <div className="section-label"><span>FEATURES</span></div>
        {(["building", "road", "zone", "clip", "object"] as const).map((type) => {
          const list = proposal.features.filter((f) => f.type === type);
          if (!list.length) return null;
          return <div key={type} className="plan-group"><h3>{TYPE_LABEL[type]} <small>{list.length}</small></h3>
            <ul className="plan-list">{list.map((f) => <li key={f.id}><button className={selected === f.id ? "active" : ""} aria-pressed={selected === f.id} onClick={() => props.onSelect(selected === f.id ? null : f.id)}>
              <span className={`plan-dot ${violating.has(f.id) ? "bad" : f.hidden ? "off" : "ok"}`} /><span>{f.name}</span>{f.hidden && <small>hidden</small>}
            </button></li>)}</ul></div>;
        })}
      </section>}

      <section className="inspector-section">
        <div className="section-label"><span>BEFORE / AFTER</span></div>
        <table className="plan-table"><thead><tr><th>Metric</th><th>Existing</th><th>Proposed</th><th>Change</th></tr></thead>
          <tbody>{state!.evaluation.metrics.map((row) => <tr key={row.metric} title={row.note}><td>{row.metric}<small>{row.unit}</small></td><td>{fmt(row.existing)}</td><td>{fmt(row.proposal)}</td><td className={row.change > 0 ? "up" : row.change < 0 ? "down" : ""}>{row.change > 0 ? "+" : ""}{fmt(row.change)}</td></tr>)}</tbody></table>
        {state!.evaluation.notes.map((note) => <p className="plan-note" key={note}>{note}</p>)}
      </section>

      {state!.existing.buildings.length > 0 && <section className="inspector-section">
        <div className="section-label"><span>EXISTING BUILDINGS</span><span>{state!.existing.buildings.length}</span></div>
        <ul className="plan-existing">{state!.existing.buildings.map((b, i) => <li key={b.id}><span><strong>Building {i + 1}</strong><small>{fmt(b.area_m2, 0)} m² · {fmt(b.height_m)} m · ~{b.floors_estimate} floors{state!.evaluation.buildings_hit.includes(b.id) ? " · in the way" : ""}</small></span>
          {demolished.has(b.id) ? <span className="plan-tag">demolished</span> : <button className="button secondary small" disabled={locked} onClick={() => props.onDemolish(b)}>Demolish</button>}</li>)}</ul>
        <p className="plan-note">{state!.existing.basis}</p>
        <div className="plan-shadow-head"><button className="button secondary small" disabled={props.facadesBusy} onClick={props.onFacades}><Icon name="eye" size={13} />{props.facadesBusy ? "Checking walls…" : "Facade completeness"}</button>
          {props.facades && props.facades.overlay.length > 0 && <label className="plan-switch"><input type="checkbox" checked={props.showFacades} onChange={(e) => props.onShowFacades(e.target.checked)} />Hatch unseen walls</label>}</div>
        {props.facades && <>
          <ul className="plan-existing">{props.facades.buildings.map((b, i) => <li key={b.id}><span><strong>Building {i + 1} · {fmt(b.observed_pct, 0)}% of walls seen</strong>
            <small>{b.facades.map((f) => `${f.facade} ${fmt(f.observed_pct, 0)}%`).join(" · ")}{b.unobserved_pct > 0 ? ` — ${fmt(b.unobserved_pct, 0)}% never observed` : ""}</small></span></li>)}</ul>
          {props.showFacades && <div className="plan-legend"><span><i style={{ background: "rgb(220,60,60)" }} />Not seen</span><span><i style={{ background: "rgb(240,170,50)" }} />Weak</span></div>}
          <p className="plan-note">{props.facades.basis}. {props.facades.notes.join(" ")}</p>
        </>}
      </section>}

      <ShadowStudy shadow={props.shadow} georeferenced={state!.frame.status === "georeferenced"} disabled={!state}
        onToggle={props.onShadowToggle} onSettings={props.onShadowSettings} onRun={props.onShadowRun} />

      <ExchangeSection busy={busy} locked={locked} georeferenced={state!.frame.status === "georeferenced"} onExport={props.onExport} onImport={props.onImport} />
    </>}
  </div>;
}
