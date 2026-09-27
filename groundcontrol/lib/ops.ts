/**
 * Operations client (Phase 4): disaster, construction and border analyses on a scan.
 * Every result is measured from the reconstruction (or heuristic where the report says so);
 * overlays are drawn over the scan and never written into it.
 */
import type { ExportFile, PlanMesh } from "./plan";

export type OpsApplication = "disaster" | "construction" | "border";
export type OpsTool = "polygon" | "access" | "seed" | "line" | "post";
export type Geo = { lat?: number; lon?: number; mgrs?: string; viewer?: [number, number, number] };

export type OpsSummary = {
  scene: string; name: string; captured_at: string | null;
  frame: { status: string; scale_status: string };
  grid: { cell_m: number; size: [number, number]; observed_pct: number; extent_m: [number, number] };
  epochs: { scene: string; name: string; georeferenced: boolean; captured_at: string | null; compatible: boolean; same_site: boolean; reason: string | null }[];
  designs: string[]; road_labels: number; detections: { raw: number; source: string };
  firstmap: { run: string; window: number; georeferenced: boolean; points: number; registered_images: number; elapsed_s: number | null; units: string; crs?: string } | null;
  last: Record<string, string>;
};
export type ChangeRegion = Geo & { id: string; kind: "gain" | "loss"; area_m2: number; mean_dh_m: number; extreme_dh_m: number; volume_m3: number; volume_sigma_m3: number; centre_enu: [number, number] };
export type ChangeReport = {
  registration: { dx_m: number; dy_m: number; dz_m: number; sigma_reg_m: number; stable_cells: number };
  cells: { total: number; observed_both: number; unobserved: number; gain: number; loss: number };
  lod95_median_m: number | null; volume: { gain_m3: number; loss_m3: number }; regions: ChangeRegion[];
  epochs: { before: string; after: string; basis: string }; notes: string[];
};
export type VolumeReport = { net_m3: number; gain_m3: number; loss_m3: number; sigma_m3: number | null; polygon_area_m2: number; observed_fraction: number; filled_fraction: number; valid: boolean; reason: string | null; basis: string };
export type Building = Geo & { id: string; grade: "intact" | "partial" | "collapsed" | "unknown"; reason: string; footprint_m2?: number; height_m?: number; height_ratio?: number | null; roof_cover?: number; rough?: number };
export type DamageReport = { buildings: Building[]; counts: Record<string, number>; footprints: string; notes: string[] };
export type Detection = Geo & { id: string; class: string; position: [number, number, number]; sightings: number; first_t: number | null; last_t: number | null; moving: boolean; spread_m: number };
export type DetectionReport = { objects: Detection[]; hits: number; unplaced: number; source: string; notes: string[] };
export type AccessReport = {
  blocked: (Geo & { id: string; area_m2: number; max_height_m: number; centre: [number, number]; new_since_before: boolean })[];
  road_m2: number; blocked_m2: number; road_source: string;
  route?: { snapped_m: Record<string, number>; length_m: number; road_m: number; offroad_m: number; eta_s: number; vehicle_width_m: number; notes: string[] };
  route_error?: string;
};
export type FloodReport = {
  level_m: number; flooded_m2: number; volume_m3: number; max_depth_m: number; depth_bands_m2: Record<string, number>;
  isolated_low_ground_m2: number; buildings: (Geo & { id: string; max_depth_m: number })[]; notes: string[];
};
export type CutFillReport = {
  design: string; design_frame: string; cut_m3: number; cut_sigma_m3: number; fill_m3: number; fill_sigma_m3: number; net_m3: number;
  tolerance_m: number; point_noise_m: number; design_observed_fraction: number;
  area_m2: { design: number; compared: number; cut: number; fill: number; on_grade: number };
  zones?: { zone: string; remaining_m3: number; start_m3?: number; progress_pct?: number | null }[]; notes: string[];
};
export type CorridorReport = {
  line_length_m: number; covered_fraction: number; step_m: number;
  posts: { id: string; covered_m?: number; eye_z?: number; error?: string }[];
  blind_stretches: (Geo & { from_m: number; to_m: number; length_m: number; unobserved_ground: boolean })[];
  profile: { chainage_m: number[]; z: (number | null)[]; covered: boolean[]; length_m: number; min_z: number | null; max_z: number | null; climb_m: number; observed_fraction: number };
  notes: string[];
};
export type DeviationReport = { design: string; model_frame: string; model_area_m2: number; scan_points_compared: number; scan_points_unrelated: number;
  tolerance_m: number; within_tolerance_pct: number | null; mean_m: number | null; rms_m: number | null; p95_abs_m: number | null;
  proud_pct: number | null; short_pct: number | null; sampling_error_m: number; basis: string };
export type FirstMap = { available: false; reason: string } | {
  available: true; run: string; window: number; georeferenced: boolean; reason: string | null; points: number; registered_images: number;
  elapsed_s: number | null; cell_m: number; units: string; crs?: string; label: string; png_base64: string | null; observed_fraction: number;
};
export type Result<R> = { report: R; overlay: PlanMesh[] };
export type Post = { id: string; at: [number, number]; height_m: number; range_m: number; bearing_deg: number | null; fov_deg: number };
export type PackKind = "change" | "cutfill" | "damage" | "corridor" | "access" | "flood" | "detections";

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
const q = (scene: string) => `scene=${encodeURIComponent(scene)}`;
type XZ = [number, number];

export const opsApi = {
  summary: (scene: string) => request<OpsSummary>(`ops/summary?${q(scene)}`),
  firstmap: (scene: string) => request<FirstMap>(`ops/firstmap?${q(scene)}`),
  detections: (scene: string) => request<Result<DetectionReport>>(`ops/detections?${q(scene)}`),
  change: (scene: string, before: string) => request<Result<ChangeReport>>("ops/change", { scene, before }),
  volume: (scene: string, before: string, polygon: XZ[]) => request<Result<VolumeReport>>("ops/volume", { scene, before, polygon }),
  damage: (scene: string, before: string | null) => request<Result<DamageReport>>("ops/damage", { scene, ...(before ? { before } : {}) }),
  access: (scene: string, body: { before?: string | null; start?: XZ; end?: XZ; vehicle_width_m: number }) =>
    request<Result<AccessReport>>("ops/access", { scene, ...body, before: body.before || undefined }),
  flood: (scene: string, seed: XZ, rise_m: number) => request<Result<FloodReport>>("ops/flood", { scene, seed, rise_m }),
  design: (scene: string, filename: string, content: string) => request<{ design: string; kind: string; faces: number | null; designs: string[] }>("ops/design", { scene, filename, content }),
  cutfill: (scene: string, design: string, design_frame: "scene" | "utm", baseline: string | null) =>
    request<Result<CutFillReport>>("ops/cutfill", { scene, design, design_frame, ...(baseline ? { baseline } : {}) }),
  deviation: (scene: string, design: string, model_frame: "scene" | "enu", tolerance_m: number) =>
    request<Result<DeviationReport>>("ops/deviation", { scene, design, model_frame, tolerance_m }),
  corridor: (scene: string, line: XZ[], posts: Post[]) => request<Result<CorridorReport>>("ops/corridor", { scene, line, posts }),
  tiles: (scene: string, tile_m: number) => request<ExportFile & { tiles: number; crs: boolean }>("ops/tiles", { scene, tile_m }),
  pack: (scene: string, kind: PackKind) => request<ExportFile>("ops/pack", { scene, kind }),
};

/** What the viewer draws in the Operations tab: overlays only, nothing pickable or editable. */
export function opsViewerPlan(overlay: PlanMesh[]) {
  return { features: overlay.map((mesh, i) => ({ id: `ops-${i}`, kind: "overlay", meshes: [mesh] })), clips: [], view: "proposal" as const, selected: null, edit: null };
}

export const TOOL_HINT: Record<OpsTool, { hint: string; limit: number; min: number }> = {
  polygon: { hint: "Click around the pile or debris, then Finish (three or more points).", limit: 64, min: 3 },
  access: { hint: "Click the staging point, then the site the vehicle must reach.", limit: 2, min: 2 },
  seed: { hint: "Click a point in the river or the lowest street where water enters.", limit: 1, min: 1 },
  line: { hint: "Click along the border or fence line, then Finish.", limit: 128, min: 2 },
  post: { hint: "Click where each observation post stands. Set its height and range below.", limit: 1, min: 1 },
};

export const fmtM3 = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k m³` : `${Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)} m³`);
export const fmtArea = (v: number) => (v >= 10000 ? `${(v / 10000).toFixed(2)} ha` : `${v.toFixed(0)} m²`);
