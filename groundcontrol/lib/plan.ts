/**
 * Planning editor client: proposals (schemes) of roads, buildings, plots, objects and
 * demolitions drawn over the reconstructed scene.
 *
 * The Python backend (scripts/workspace_proposals.py) owns all geometry, rule checks
 * and metrics; every write returns the proposal together with its fresh evaluation, so
 * this client only ever shows what the backend computed. Coordinates are the viewer
 * frame (Y-up metres); exports convert to WGS84 only when the scene is georeferenced.
 */
import type { Point } from "./workspace";

export type PlanFeatureType = "road" | "building" | "zone" | "object" | "clip" | "symbol" | "route" | "phase_line";
export type PlanTool = PlanFeatureType | "array" | "los";
export type ProposalKind = "plan" | "mission";
export type XZ = [number, number];

export type RoadParams = {
  centerline: XZ[]; width_m: number; lanes: number; footpath_m: number; median_m: number;
  mode: "drape" | "graded"; max_grade_pct: number; surface: "asphalt" | "concrete" | "gravel";
};
export type BuildingParams = {
  footprint: XZ[]; floors: number; floor_height_m: number; roof: "flat" | "gable" | "hip";
  roof_pitch_deg: number; use: "residential" | "commercial" | "mixed" | "institutional" | "industrial";
};
export type ZoneRules = { max_height_m?: number; max_fsi?: number; max_coverage_pct?: number; setback_m?: number; max_floors?: number };
export type ZoneParams = { polygon: XZ[]; rules: ZoneRules; label: string };
export type ObjectParams = { item: string; position: XZ; yaw_deg: number; scale: number; jib_radius_m?: number; clearance_m?: number };
export type ClipParams = { polygon: XZ[]; reason: string };
export type Affiliation = "hostile" | "friendly" | "unknown" | "neutral";
export type SymbolRole = "infantry" | "sniper" | "machine_gun" | "vehicle" | "observation_post" | "objective" | "rally_point"
  | "hlz" | "support_by_fire" | "obstacle" | "checkpoint";
export type Behaviour = "sentry" | "patrol" | "overwatch" | "reaction";
export type SymbolParams = {
  position: XZ; affiliation: Affiliation; role: SymbolRole; elevation_m: number; bearing_deg: number; sector_deg: number;
  range_m: number; behaviour: Behaviour; alert_radius_m: number; count: number; diameter_m: number; patrol_route: string | null;
  source: "planner" | "labels" | "detections"; notes: string;
};
export type RouteParams = { waypoints: XZ[]; names: string[]; kind: "approach" | "withdrawal" | "patrol"; pace: "walk" | "patrol" | "run"; loop: boolean };
export type PhaseLineParams = { line: XZ[]; label: string };

export type PlanFeature = {
  id: string; type: PlanFeatureType; name: string; hidden: boolean; locked: boolean;
  params: RoadParams | BuildingParams | ZoneParams | ObjectParams | ClipParams | SymbolParams | RouteParams | PhaseLineParams;
  created_at?: string; updated_at?: string;
};
export type Proposal = {
  id: string; name: string; kind: string; revision: number; features: PlanFeature[];
  settings: { persons_per_m2: number }; created_at: string; updated_at: string;
  /** A hypothesis reconstruction (ARC-05): shown and exported as inferred, never as measured. */
  inferred?: boolean; inferred_basis?: string;
};
export type PlanMesh = {
  kind: string; part: string; color: [number, number, number]; opacity: number;
  positions: number[]; indices: number[]; violation?: boolean;
};
export type ClipObb = { cx: number; cz: number; hx: number; hz: number; angle: number; y_min: number; y_max: number };
export type ZoneCheck = { rule: string; value: number; limit: number; unit: string; subject: string; ok: boolean };
export type DerivedFeature = {
  meshes: PlanMesh[]; footprint: XZ[];
  metrics: Record<string, number | null | ZoneCheck[] | undefined> & { checks?: ZoneCheck[] };
  clip?: { polygon: XZ[]; y_min: number; y_max: number; obb: ClipObb };
  facing_xz?: [number, number];
};
export type Violation = { zone: string; feature: string; rule: string; value: number; limit: number; unit: string };
export type MetricRow = { metric: string; unit: string; existing: number; proposal: number; change: number; note?: string };
export type Evaluation = {
  features: Record<string, DerivedFeature>; violations: Violation[];
  impacts: Record<string, { existing_buildings_hit: string[]; trees_removed: string[] }>;
  demolished: string[]; buildings_hit: string[]; metrics: MetricRow[]; notes: string[]; scale_status: string;
};
export type ExistingBuilding = {
  id: string; centre: XZ; size: XZ; angle_rad: number; footprint: XZ[]; base_y: number;
  height_m: number; area_m2: number; floors_estimate: number;
};
export type PlanState = {
  proposal: Proposal; evaluation: Evaluation;
  frame: { status: string; scale_status: string; scale_source: string | null };
  existing: { status: string; basis?: string; buildings: ExistingBuilding[]; trees: number };
};
export type ProposalIndex = { schema_version: number; proposals: { id: string; name: string; kind: string; created_at: string }[]; active: string | null };
export type Catalogue = Record<string, { size: [number, number, number]; color: [number, number, number]; detailed?: boolean }>;
export type FacadeResult = {
  buildings: { id: string; observed_pct: number; weak_pct: number; unobserved_pct: number; wall_area_m2: number;
    facades: { facade: string; bearing_deg: number; observed_pct: number; weak_pct: number; unobserved_pct: number; area_m2: number }[] }[];
  overlay: PlanMesh[]; basis: string; notes: string[];
};
export type ExportFormat = "geojson" | "cityjson" | "dxf" | "3dtiles";
export type ExportFile = { filename: string; media_type: string; encoding: "utf-8" | "base64"; content: string };
export type ShadowMetric = { metric: string; unit: string; existing: number; proposal: number; change: number };
export type ShadowResult = {
  mode: "instant" | "day"; overlay: PlanMesh[]; metrics: ShadowMetric[]; notes: string[];
  sun: { azimuth_deg?: number; elevation_deg?: number; local_time?: string; samples?: { local_time: string; azimuth_deg: number; elevation_deg: number }[] };
  location?: { lat_deg: number; lon_deg: number; source: string };
};
export type ShadowSettings = {
  mode: "instant" | "day"; date: string; time: string; utc_offset: number; start: number; end: number; step: number;
  lat?: number; lon?: number; north_deg: number;
};
export type ImportReport = { imported: number; skipped: { parcel: string; reason: string }[]; skipped_count: number; basis: string };
export type Coordinates = {
  status: string; mgrs?: string; dms?: string; lat_deg?: number; lon_deg?: number; height_m?: number;
  utm?: { zone: string; epsg: number; easting_m: number; northing_m: number }; reason?: string;
};

const ROOT = "/api/backend/api/workspace";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${ROOT}/${path}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store",
    ...(body === undefined ? {} : { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  });
  const result = await response.json().catch(() => ({ error: "The service returned an unreadable response." }));
  if (!response.ok) throw new Error(result.error ?? `Request failed (${response.status}).`);
  return result as T;
}
const q = (scene: string, extra = "") => `scene=${encodeURIComponent(scene)}${extra}`;

export const plan = {
  list: (scene: string, kind: ProposalKind = "plan") => request<{ index: ProposalIndex; catalogue: Catalogue }>(`plan/proposals?${q(scene, `&kind=${kind}`)}`),
  load: (scene: string, id: string) => request<PlanState>(`plan/proposal?${q(scene, `&id=${encodeURIComponent(id)}`)}`),
  create: (scene: string, name: string, source?: string, kind: ProposalKind = "plan") => request<PlanState>("plan/proposals", { scene, name, kind, ...(source ? { source } : {}) }),
  rename: (scene: string, id: string, name: string) => request<PlanState>("plan/proposals/rename", { scene, id, name }),
  setInferred: (scene: string, id: string, inferred: boolean, basis: string) => request<PlanState>("plan/proposals/inferred", { scene, id, inferred, basis }),
  remove: (scene: string, id: string) => request<{ index: ProposalIndex }>("plan/proposals/delete", { scene, id }),
  upsert: (scene: string, proposal: Proposal, feature: Partial<PlanFeature> & { type: PlanFeatureType; params: PlanFeature["params"] }) =>
    request<PlanState>("plan/feature", { scene, id: proposal.id, revision: proposal.revision, feature }),
  removeFeature: (scene: string, proposal: Proposal, featureId: string) =>
    request<PlanState>("plan/feature/delete", { scene, id: proposal.id, revision: proposal.revision, feature_id: featureId }),
  replace: (scene: string, proposal: Proposal, features: PlanFeature[]) =>
    request<PlanState>("plan/features", { scene, id: proposal.id, revision: proposal.revision, features }),
  array: (scene: string, proposal: Proposal, item: string, line: XZ[], spacing_m: number, offset_m: number) =>
    request<PlanState>("plan/array", { scene, id: proposal.id, revision: proposal.revision, item, line, spacing_m, offset_m }),
  exportGeo: (scene: string, id: string) => request<Record<string, unknown>>(`plan/export?${q(scene, `&id=${encodeURIComponent(id)}`)}`),
  exportAs: (scene: string, id: string, format: ExportFormat) =>
    request<ExportFile>(`plan/export?${q(scene, `&id=${encodeURIComponent(id)}&format=${format}`)}`),
  shadow: (scene: string, id: string, s: ShadowSettings) => {
    const extra = new URLSearchParams({ id, mode: s.mode, date: s.date, time: s.time, utc_offset: String(s.utc_offset),
      start: String(s.start), end: String(s.end), step: String(s.step), north_deg: String(s.north_deg) });
    if (s.lat !== undefined && s.lon !== undefined) { extra.set("lat", String(s.lat)); extra.set("lon", String(s.lon)); }
    return request<ShadowResult>(`plan/shadow?${q(scene)}&${extra.toString()}`);
  },
  importParcels: (scene: string, proposal: Proposal, filename: string, content: string, crs?: number) =>
    request<PlanState & { import: ImportReport }>("plan/import", { scene, id: proposal.id, revision: proposal.revision, filename, content, ...(crs ? { crs } : {}) }),
  facades: (scene: string) => request<FacadeResult>(`plan/facades?${q(scene)}`),
  coords: (scene: string, point: Point) => request<Coordinates>(`coords?${q(scene, `&x=${point[0]}&y=${point[1]}&z=${point[2]}`)}`),
};

/** Two clicks make an axis-aligned rectangle from its diagonal; three or more, a polygon. */
export function outline(points: Point[]): XZ[] | null {
  const xz = points.map((p) => [p[0], p[2]] as XZ);
  if (xz.length === 2) {
    const [[x0, z0], [x1, z1]] = xz;
    if (Math.abs(x1 - x0) < 1 || Math.abs(z1 - z0) < 1) return null;
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]];
  }
  return xz.length >= 3 ? xz : null;
}

/** The first draft of a feature from what was clicked; defaults match the backend's. */
export function draftFeature(tool: PlanFeatureType, points: Point[], extra: { item?: string } = {}):
  (Partial<PlanFeature> & { type: PlanFeatureType; params: PlanFeature["params"] }) | null {
  if (tool === "road") {
    if (points.length < 2) return null;
    return { type: "road", name: "New road", params: { centerline: points.map((p) => [p[0], p[2]] as XZ), width_m: 7, lanes: 2, footpath_m: 1.5, median_m: 0, mode: "drape", max_grade_pct: 6, surface: "asphalt" } };
  }
  if (tool === "object") {
    if (points.length < 1 || !extra.item) return null;
    return { type: "object", name: extra.item.replace(/_/g, " "), params: { item: extra.item, position: [points[0][0], points[0][2]], yaw_deg: 0, scale: 1 } };
  }
  const shape = outline(points);
  if (!shape) return null;
  if (tool === "building") return { type: "building", name: "New building", params: { footprint: shape, floors: 4, floor_height_m: 3.2, roof: "flat", roof_pitch_deg: 30, use: "residential" } };
  if (tool === "zone") return { type: "zone", name: "Plot", params: { polygon: shape, rules: {}, label: "Plot" } };
  return { type: "clip", name: "Demolish", params: { polygon: shape, reason: "demolish" } };
}

export type EditShape = { id: string; points: XZ[]; y: number; closed: boolean; vertices: boolean; rotate: boolean; heading_deg?: number };

/** What the viewer may drag for the selected feature: its outline, on its base plane. */
export function editShape(state: PlanState | null, id: string | null): EditShape | null {
  const feature = state?.proposal.features.find((f) => f.id === id);
  const derived = id ? state?.evaluation.features[id] : undefined;
  if (!feature || !derived || feature.locked || feature.hidden) return null;
  const y = typeof derived.metrics.base_y === "number" ? derived.metrics.base_y
    : Math.min(...derived.meshes.flatMap((m) => m.positions.filter((_, i) => i % 3 === 1)));
  if (!Number.isFinite(y)) return null;
  if (feature.type === "road") return { id: feature.id, points: (feature.params as RoadParams).centerline, y, closed: false, vertices: true, rotate: false };
  if (feature.type === "route") return { id: feature.id, points: (feature.params as RouteParams).waypoints, y, closed: false, vertices: true, rotate: false };
  if (feature.type === "phase_line") return { id: feature.id, points: (feature.params as PhaseLineParams).line, y, closed: false, vertices: true, rotate: false };
  if (feature.type === "symbol") {
    const f = derived.facing_xz ?? [0, -1];
    return { id: feature.id, points: [(feature.params as SymbolParams).position], y, closed: false, vertices: false, rotate: true,
      heading_deg: (Math.atan2(f[1], f[0]) * 180) / Math.PI };
  }
  if (feature.type === "object") {
    const p = feature.params as ObjectParams;
    return { id: feature.id, points: [p.position], y, closed: false, vertices: false, rotate: true, heading_deg: p.yaw_deg };
  }
  const outline = feature.type === "building" ? (feature.params as BuildingParams).footprint : (feature.params as ZoneParams | ClipParams).polygon;
  return { id: feature.id, points: outline, y, closed: true, vertices: true, rotate: true };
}

const round = (v: number) => Math.round(v * 1e4) / 1e4;
/** A feature with the outline the viewer dragged (objects turn by the dragged angle). */
export function applyEdit(feature: PlanFeature, points: XZ[], rotationDeg: number): PlanFeature {
  const pts = points.map(([x, z]) => [round(x), round(z)] as XZ);
  const params = { ...feature.params } as Record<string, unknown>;
  if (feature.type === "road") params.centerline = pts;
  else if (feature.type === "route") {
    const names = (feature.params as RouteParams).names ?? [];
    params.waypoints = pts;
    // A corner added or removed shifts every name after it: keep only the start and end names.
    if (pts.length !== (feature.params as RouteParams).waypoints.length) params.names = pts.map((_, i) => (i === 0 ? names[0] ?? "" : i === pts.length - 1 ? names.at(-1) ?? "" : ""));
  } else if (feature.type === "phase_line") params.line = pts;
  else if (feature.type === "symbol") {
    params.position = pts[0];
    // The 3D rotate grip turns +x toward +z, i.e. clockwise seen from above: bearings grow.
    params.bearing_deg = round((((feature.params as SymbolParams).bearing_deg + rotationDeg) % 360 + 360) % 360);
  } else if (feature.type === "building") params.footprint = pts;
  else if (feature.type === "object") {
    params.position = pts[0];
    const yaw = (feature.params as ObjectParams).yaw_deg + rotationDeg;
    params.yaw_deg = round(((yaw + 180) % 360 + 360) % 360 - 180);
  } else params.polygon = pts;
  return { ...feature, params: params as PlanFeature["params"] };
}

/** The viewer payload: every derived mesh, the demolished boxes to hide, overlays and the edit outline. */
export function viewerPlan(state: PlanState | null, selected: string | null, view: "proposal" | "existing",
  { overlay = [], edit = null }: { overlay?: PlanMesh[]; edit?: EditShape | null } = {}) {
  if (!state) return { features: [], clips: [], view: "existing" as const, selected: null, edit: null };
  // A hypothesis scheme is drawn translucent and tinted violet, so inferred geometry can
  // never be mistaken for the scan (ARC-05).
  const tint = (m: PlanMesh): PlanMesh => ({ ...m, opacity: Math.min(m.opacity, 0.6),
    color: [Math.round(m.color[0] * 0.55 + 150 * 0.45), Math.round(m.color[1] * 0.55 + 90 * 0.45), Math.round(m.color[2] * 0.55 + 210 * 0.45)] });
  const features = Object.entries(state.evaluation.features).map(([id, derived]) => ({
    id, kind: state.proposal.features.find((f) => f.id === id)?.type ?? "feature",
    meshes: state.proposal.inferred ? derived.meshes.map(tint) : derived.meshes,
  }));
  // Shadow overlays ride along as one unpickable feature per mesh (16-mesh cap per feature).
  overlay.forEach((mesh, i) => features.push({ id: `overlay-${i}`, kind: "overlay", meshes: [mesh] }));
  const clips = Object.values(state.evaluation.features).flatMap((d) => (d.clip ? [d.clip.obb] : [])).slice(0, 8);
  return { features, clips, view, selected, edit: view === "proposal" ? edit : null };
}

export const TOOL_COPY: Record<PlanTool, { label: string; hint: string; min: number }> = {
  road: { label: "Road", hint: "Click along the centreline, then Finish. The road drapes on the measured ground; switch to graded for a design profile.", min: 2 },
  building: { label: "Building", hint: "Click two opposite corners for a rectangle, or three or more corners for any footprint, then Finish.", min: 2 },
  zone: { label: "Plot / zone", hint: "Outline the plot (two corners or a polygon). Its rules — height, FSI, coverage, setback — are checked against the buildings inside.", min: 2 },
  clip: { label: "Demolish", hint: "Outline what goes (two corners or a polygon). The scanned model inside is hidden in the proposal view.", min: 2 },
  object: { label: "Object", hint: "Pick an item, then click the ground to place it.", min: 1 },
  array: { label: "Array", hint: "Click a line, then Finish: the item repeats along it at the spacing you set.", min: 2 },
  symbol: { label: "Symbol", hint: "Pick a role, then click where it stands (a roof counts). Set its facing and sector after.", min: 1 },
  route: { label: "Route", hint: "Click the waypoints in order from the start point to the objective, then Finish.", min: 2 },
  phase_line: { label: "Phase line", hint: "Click along the line, then Finish. The route report gives the time each is crossed.", min: 2 },
  los: { label: "Line of sight", hint: "Click the observer, then the target: eye 1.6 m, chest 1.2 m above the surface.", min: 2 },
};

export const SYMBOL_ROLES: { role: SymbolRole; label: string; affiliation: Affiliation }[] = [
  { role: "infantry", label: "Enemy post", affiliation: "hostile" }, { role: "sniper", label: "Sniper", affiliation: "hostile" },
  { role: "machine_gun", label: "Machine gun", affiliation: "hostile" }, { role: "observation_post", label: "Observation post", affiliation: "hostile" },
  { role: "vehicle", label: "Vehicle", affiliation: "hostile" }, { role: "objective", label: "Objective", affiliation: "friendly" },
  { role: "rally_point", label: "Rally point", affiliation: "friendly" }, { role: "support_by_fire", label: "Support by fire", affiliation: "friendly" },
  { role: "hlz", label: "HLZ", affiliation: "friendly" }, { role: "checkpoint", label: "Checkpoint", affiliation: "unknown" },
  { role: "obstacle", label: "Obstacle", affiliation: "unknown" },
];

/** First draft of a mission feature from clicks; the backend fills every default. */
export function draftMissionFeature(tool: "symbol" | "route" | "phase_line", points: Point[], role: SymbolRole = "infantry"):
  (Partial<PlanFeature> & { type: PlanFeatureType; params: PlanFeature["params"] }) | null {
  if (tool === "symbol") {
    if (!points.length) return null;
    const entry = SYMBOL_ROLES.find((r) => r.role === role) ?? SYMBOL_ROLES[0];
    return { type: "symbol", name: entry.label, params: { position: [points[0][0], points[0][2]], role, affiliation: entry.affiliation } as unknown as SymbolParams };
  }
  if (points.length < 2) return null;
  const line = points.map((p) => [p[0], p[2]] as XZ);
  if (tool === "route") return { type: "route", name: "Route", params: { waypoints: line, names: line.map((_, i) => (i === 0 ? "SP" : i === line.length - 1 ? "OBJ" : "")) } as unknown as RouteParams };
  return { type: "phase_line", name: "Phase line", params: { line, label: "PL" } as PhaseLineParams };
}
