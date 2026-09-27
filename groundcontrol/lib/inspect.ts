/**
 * Phase 5 client: inspection (defect register, frames that saw a point, tilt, wire, cracks),
 * archaeology (sections, terrain rasters, M3C2 change, archive record) and the digital twin
 * (assets with attribute cards, epochs, engine package). Positions are viewer-frame metres.
 */
import type { ExportFile, PlanMesh } from "./plan";

export type Vec3 = [number, number, number];
export type TracedFrame = { camera_index: number; name: string; t_sec: number | null; u: number; v: number; width: number; height: number;
  distance_m: number; gsd_mm: number; centrality: number; score: number; url: string | null };
export type FrameTrace = { frames: TracedFrame[]; seen_by: number; cameras: number; basis: string; where: { lat: number; lon: number; mgrs: string } | null };
export const DEFECT_TYPES = ["crack", "spalling", "corrosion", "exposed_rebar", "deformation", "vegetation", "water_ingress",
  "missing_element", "inscription", "carving", "erosion", "graffiti", "other"] as const;
export type DefectType = (typeof DEFECT_TYPES)[number];
export const STATUSES = ["open", "monitor", "repair_planned", "closed"] as const;
export type Annotation = {
  id: string; position: Vec3; type: DefectType; severity: number; status: (typeof STATUSES)[number]; title: string; note: string;
  element: string; measurements: Record<string, number>; created_at: string; seen_by: number; frames: TracedFrame[];
  photo: { file: string; url?: string; frame: string; centre_px: [number, number]; box: number[] } | null;
};
export type Register = { items: Annotation[]; revision: number };
export type CrackResult = { candidates: { length_px: number; width_px: number; length_mm?: number; width_mm?: number; elongation: number }[];
  overlay_jpeg_base64: string; gsd_mm: number; basis: string };
export type TiltReport = { tilt_deg: number; lean_bearing_deg: number; lean_mm_per_m: number; top_offset_m: number; height_m: number; points: number; radius_rms_m: number; basis: string };
export type WireReport = { span_m: number; sag_m: number; sag_pct_of_span: number; min_clearance_m: number | null; clearance_at_m: number | null;
  clearance_ok?: boolean; clearance_limit_m?: number; fit_rms_m: number; points: number; basis: string };
export type SectionReport = { length_m: number; points: number; azimuth_deg: number; y_range: [number, number]; observed_pct: number | null };
export type TerrainLayer = "ortho" | "hillshade_dtm" | "hillshade_dsm" | "lrm" | "slope_deg";
export type TerrainResult = { layer: TerrainLayer; png_base64: string; bounds: { x0: number; z0: number; x1: number; z1: number }; cell_m: number;
  lrm_range_m: [number, number]; overlay: PlanMesh[]; basis: string };
export type M3c2Report = { core_points: number; observed: number; significant: number; moved_towards_viewer: number; moved_away: number;
  median_lod95_m: number | null; max_abs_significant_m: number | null; point_spacing_m: number; basis: string; note: string;
  parameters: { normal_scale_m: number; projection_diameter_m: number } };
export type ArchiveRecord = { dublin_core: Record<string, unknown>; capture: Record<string, unknown>; spatial_reference: Record<string, unknown>;
  processing: Record<string, unknown>; fixity: { file: string; bytes: number; sha256: string }[]; notes: string[]; generated: string };
export type Asset = { id: string; kind: "building" | "tree" | "pole"; position: [number, number]; viewer: Vec3; measured: Record<string, number | number[] | null>;
  attributes: Record<string, string>; defects_open: number; worst_severity: number | null; mgrs?: string; basis: string };
export type Inventory = { assets: Asset[]; counts: Record<string, number>; orphaned_attributes: string[]; notes: string[] };
export type Epochs = { site: string | null; epochs: { scene: string; name: string; captured_at: string | null; current: boolean;
  change: { against: string; gain_m3: number; loss_m3: number; regions: number } | null }[]; note: string | null };
export const ATTRIBUTE_KEYS = ["name", "asset_tag", "owner", "use", "material", "condition", "built_year", "last_inspected", "notes"] as const;
type WithOverlay<R> = { report: R; overlay: PlanMesh[] };

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

export const inspectApi = {
  frames: (scene: string, p: Vec3) => request<FrameTrace>(`inspect/frames?${q(scene)}&x=${p[0]}&y=${p[1]}&z=${p[2]}`),
  register: (scene: string) => request<Register>(`inspect/annotations?${q(scene)}`),
  add: (scene: string, revision: number, item: Partial<Annotation>) => request<{ register: Register; created: string }>("inspect/annotation", { scene, revision, item }),
  update: (scene: string, revision: number, id: string, item: Partial<Annotation>) => request<{ register: Register }>("inspect/annotation/update", { scene, revision, id, item }),
  remove: (scene: string, revision: number, id: string) => request<{ register: Register }>("inspect/annotation/delete", { scene, revision, id }),
  crack: (scene: string, id: string) => request<CrackResult>(`inspect/crack?${q(scene)}&id=${id}`),
  report: (scene: string) => request<ExportFile>(`inspect/report?${q(scene)}`),
  tilt: (scene: string, base: Vec3, radius_m: number) => request<WithOverlay<TiltReport>>("inspect/tilt", { scene, base, radius_m }),
  wire: (scene: string, a: Vec3, b: Vec3, clearance_limit_m: number | null) => request<WithOverlay<WireReport>>("inspect/wire", { scene, a, b, clearance_limit_m }),
  section: (scene: string, a: Vec3, b: Vec3, half_width_m: number, title: string) =>
    request<WithOverlay<SectionReport> & { svg: ExportFile; dxf: ExportFile }>("inspect/section", { scene, a, b, half_width_m, title }),
  terrain: (scene: string, layer: TerrainLayer) => request<TerrainResult>(`inspect/terrain?${q(scene)}&layer=${layer}`),
  m3c2: (scene: string, before: string, region: [number, number][] | null) => request<WithOverlay<M3c2Report>>("inspect/m3c2", { scene, before, ...(region ? { region } : {}) }),
  provenance: (scene: string) => request<ArchiveRecord>(`inspect/provenance?${q(scene)}`),
  verify: (scene: string, record: ArchiveRecord) => request<{ changed: { file: string; problem: string }[] }>("inspect/provenance/verify", { scene, record }),
};

export const twinApi = {
  inventory: (scene: string) => request<Inventory>(`twin/inventory?${q(scene)}`),
  attributes: (scene: string, id: string, values: Record<string, string>) => request<{ attributes: Record<string, Record<string, string>> }>("twin/attributes", { scene, id, values }),
  epochs: (scene: string) => request<Epochs>(`twin/epochs?${q(scene)}`),
  package: (scene: string, splats: boolean) => request<{ manifest: { files: string[]; triangles: number; frame: string }; bytes: number; url: string }>("twin/package", { scene, splats }),
};

/** Vertical marker meshes (mast + head) at viewer points, one mesh per colour. */
export function pinMesh(points: Vec3[], color: [number, number, number], part: string, height = 3): PlanMesh | null {
  if (!points.length) return null;
  const positions: number[] = [], indices: number[] = [];
  const box = (x0: number, x1: number, y0: number, y1: number, z0: number, z1: number) => {
    const b = positions.length / 3;
    positions.push(x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1, x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1);
    for (const [a, c, d, e] of [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]) indices.push(b + a, b + c, b + d, b + a, b + d, b + e);
  };
  for (const [x, y, z] of points) { box(x - 0.06, x + 0.06, y, y + height, z - 0.06, z + 0.06); box(x - 0.35, x + 0.35, y + height, y + height + 0.7, z - 0.35, z + 0.35); }
  return { kind: "overlay", part, color, opacity: 0.95, positions, indices };
}

export const SEVERITY_COLOR: Record<number, [number, number, number]> = { 1: [120, 170, 230], 2: [90, 190, 120], 3: [240, 200, 60], 4: [240, 140, 50], 5: [220, 50, 50] };
export const fmtDeg = (v: number) => `${v.toFixed(1)}°`;
export const bearingName = (b: number) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(((b % 360) + 360) % 360 / 45) % 8];
export function downloadFile(file: ExportFile, prefix: string) {
  const body = file.encoding === "base64" ? Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0)) : file.content;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([body], { type: file.media_type }));
  link.download = `${prefix}-${file.filename}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 4000);
}
