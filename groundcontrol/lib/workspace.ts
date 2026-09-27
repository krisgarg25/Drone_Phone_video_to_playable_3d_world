import type { ActiveJob, GnssReport, Preset, Quality, StepRow } from "./api";

export type Workflow = "general" | "inspection" | "survey" | "response" | "heritage";
export type Capture = "unknown" | "drone" | "handheld" | "phone";
export type Layer = "splats" | "cameras" | "points" | "coverage" | "collider" | "semantics";
export type Mode = "orbit" | "fly" | "walk";
export type Point = [number, number, number];
export type MeasurementKind = "point" | "distance" | "height" | "area" | "volume";
export type Measurement = {
  id: string; kind: MeasurementKind; label: string; points: Point[];
  value: number | null; unit: string; created_at: string; geometry: string;
  model_revision: string; stale: boolean;
  engine?: boolean; valid?: boolean; support?: boolean; reason?: string | null;
  uncertainty?: { m: number; unit: string; value?: number | null } | null; snapped?: Point[];
};
export type FurnitureItem = { item: string; label: string; size: [number, number, number]; footprint_m2: number; imported?: boolean; model?: ImportedModel };
export type ImportedModel = {
  id: string; file: string; source: string; size: [number, number, number];
  scale_status: "relative" | "estimated" | "metric"; scale_source: string; true_size_note: string;
};
export type PlacementFit = {
  supported: boolean; valid: boolean; fits_height: boolean | null; tight: boolean;
  support_fraction?: number; floor_clearance_m: number | null; ceiling_height_m: number | null; reason: string | null;
};
export type Placement = {
  id: string; item: string; label: string; size: [number, number, number];
  center_xz: [number, number]; center_y: number; yaw_deg: number; scale: number;
  footprint_m2: number; created_at: string; model_revision: string; stale: boolean; fit: PlacementFit;
  model?: ImportedModel;
};
export type Project = {
  id: string; name: string; workflow: Workflow; capture: Capture;
  /** Mission profile (playbook §4): decides which workspace tabs lead. */
  application: { id: ApplicationId; label: string; workspace_tabs: string[]; analyses: string[] } | null; site: string | null;
  status: "ready" | "processing" | "failed" | "uploaded" | "empty";
  updated: string; thumbnail_url: string | null; video_count: number;
  frame_count: number; registered_count: number; viewable: boolean; trained: boolean;
  scale: { status: "relative" | "estimated" | "metric"; source: string; unit: string };
  // Second, learned ruler. Advisory by construction: it reports a disagreement and
  // never alters the measurements above it. Null until the depth step has run.
  scale_check: {
    ruler_m_per_unit: number; existing_m_per_unit: number | null; gap_percent: number | null;
    agreement: string | null; confidence: string | null; model: string | null;
    changes_measurements: boolean;
  } | null;
  georeference: { status: "local" | "georeferenced"; crs: string | null; vertical_datum?: string | null; position_reference?: string | null; viewer_frame?: string };
  accuracy: { status: "unverified" | "verified"; rmse_m: number | null; horizontal_rmse_m?: number | null; vertical_rmse_m?: number | null; checkpoints?: number | null; basis?: string };
  survey?: { status: string; blockers: string[]; fit_rmse_m: number | null; gnss?: GnssReport | null } | null;
  // The world gate's verdict, compact form, so a card can say "blocked" instead of
  // looking identical to a clean pass. The full rows are on ProjectDetail.
  quality?: { status: "pass" | "warnings" | "failed"; hard_failures: string[]; warnings: string[]; checks: number } | null;
  // On the list only the name of the scenario is carried; ProjectDetail has the record.
  scenario?: { preset: string; label: string | null; decided_by: string | null } | null;
};
// ------------------------------------------------------------------------------------
// E: the scenario a scene was built as, and what the finished world says about it.
//
// Before this the preset existed only in a console line during the run: nothing on disk
// recorded which capture style produced a model, so nothing could be audited against it,
// and the gate's per-check numbers were thrown away into prose. Both are data now.
// ------------------------------------------------------------------------------------
export type ScenarioEvidence = {
  signal: string; value: string | number | boolean | null; because: string;
  param?: string; from?: unknown; to?: unknown; set_by?: string;
};
export type AuditFinding = { kind: string; severity: "info" | "warn" | "fail"; message: string; fix?: string };
export type Scenario = {
  preset: string; label: string | null; advice: string | null; decided_by: string | null;
  quality: string | null; cull: string | null;
  evidence: ScenarioEvidence[]; capture: Record<string, unknown>;
  applied: { param: string; value: string | number | boolean | null; set_by: string }[];
  recorded_at?: number | null;
  audit: {
    findings: AuditFinding[]; camera_agl_m?: number | null; footprint_m?: number | null;
    cell_m?: number | null; registration_pct?: number | null; gate_status?: string | null;
  } | null;
};
export type QualityCheckRow = {
  name: string; status: "pass" | "fail" | "na"; severity: "hard" | "soft";
  value?: number | null; unit?: string | null; threshold?: string | null;
  better?: string | null; basis?: string | null;
};
export type QualityReport = {
  status: "pass" | "warnings" | "failed"; checks: QualityCheckRow[];
  hard_failures: string[]; warnings: string[];
  thresholds: {
    min_coverage?: number; min_perimeter_m?: number; character_height_m?: number;
    min_headroom_m?: number; cell_m?: number; footprint_m?: number; grid?: number[];
    character_height_source?: string; pinned_by_operator?: string[];
  };
};
export type MediaFile = { name: string; url: string; bytes: number };
export type Frame = { name: string; url: string; t_sec: number | null; pos?: Point; camera_index: number | null };
export type ProjectDetail = Project & {
  videos: MediaFile[]; artifacts: (MediaFile & { kind: string })[];
  frames: Frame[]; steps: StepRow[]; measurements: Measurement[];
  placements: Placement[]; furniture: FurnitureItem[];
  model_revision: string; diagnostics: Record<string, unknown> | null;
  notes: string; viewer_url: string | null; warnings: string[]; job: ActiveJob;
  semantics?: { classes: string[]; counts: Record<string, number>; summary?: Record<string, { count: number; area_m2: number; mean_height_m: number }> | null } | null;
  // In detail the same key carries every row, not just the counts.
  quality?: QualityReport | null;
  scenario?: Scenario | null;
};
export type ProjectList = {
  projects: Project[]; job: ActiveJob; presets: Record<string, Preset>; qualities: Record<string, Quality>;
};
export type RunOptions = {
  scene: string; preset: string; quality: string; action?: "run" | "scan";
  engine?: "pipeline" | "survey"; dense_profile?: string;
  vertical_datum?: "ellipsoidal" | "egm96";
  anchor?: { kind: "height" | "speed"; value: number };
};
export const WORKFLOWS: Record<Workflow, { label: string; description: string }> = {
  general: { label: "Explore", description: "Reconstruct and explore any scene" },
  inspection: { label: "Inspect", description: "Examine structures and record findings" },
  survey: { label: "Measure", description: "Work with scale, geometry and exports" },
  response: { label: "Assess", description: "Review conditions and observed coverage" },
  heritage: { label: "Present", description: "Explore a place and save viewpoints" },
};
export const CAPTURES: Record<Capture, string> = { unknown: "Other / not specified", drone: "Drone", handheld: "Handheld camera", phone: "Phone" };
const ROOT = "/api/backend/api/workspace";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${ROOT}/${path}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store",
    ...(body === undefined ? {} : { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  });
  const result = await response.json().catch(() => ({ error: "The service returned an unreadable response." }));
  if (!response.ok) throw new Error(result.error ?? `Request failed (${response.status}).`);
  return result;
}

export const workspace = {
  list: () => request<ProjectList>("projects"),
  project: (scene: string) => request<ProjectDetail>(`project?scene=${encodeURIComponent(scene)}`),
  save: (body: { scene: string; name?: string; workflow?: Workflow; capture?: Capture; notes?: string; application?: ApplicationId | null; site?: string }) => request<ProjectDetail>("project", body),
  run: (body: RunOptions) => request<{ status: string }>("run", body),
  cancel: (scene: string) => request<{ status: string }>("cancel", { scene }),
  measure: (scene: string, measurement: { label: string; kind: MeasurementKind; points: Point[]; model_revision: string }) => request<ProjectDetail>("measurements", { scene, ...measurement }),
  removeMeasurement: (scene: string, id: string) => request<ProjectDetail>("measurements/delete", { scene, id }),
  place: (scene: string, body: { item: string; point: Point; yaw_deg?: number; scale?: number; label?: string; model_revision: string }) => request<ProjectDetail>("placements", { scene, ...body }),
  // `size` is the edited footprint: a resize cannot be expressed by point/yaw/scale alone.
  movePlacement: (scene: string, body: { id: string; item: string; point: Point; yaw_deg?: number; scale?: number; size?: Point; label?: string; model_revision: string }) => request<ProjectDetail>("placements/update", { scene, ...body }),
  removePlacement: (scene: string, id: string) => request<ProjectDetail>("placements/delete", { scene, id }),
  upload: (scene: string, files: File[], progress: (value: number) => void) => new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${ROOT}/upload`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) progress(Math.round(event.loaded / event.total * 100)); };
    xhr.onerror = () => reject(new Error("Upload interrupted. Check the local service and retry."));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else {
        let message = `Upload failed (${xhr.status}).`;
        try { message = JSON.parse(xhr.responseText).error ?? message; } catch { /* Non-JSON transport errors have no safe detail. */ }
        reject(new Error(message));
      }
    };
    const form = new FormData();
    form.append("scene", scene);
    files.forEach((file) => form.append("file", file, file.name));
    xhr.send(form);
  }),
  // A single self-contained glTF/GLB becomes a placeable item with its real size.
  importModel: (scene: string, file: File, progress: (value: number) => void) => new Promise<ProjectDetail>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${ROOT}/model/import`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) progress(Math.round(event.loaded / event.total * 100)); };
    xhr.onerror = () => reject(new Error("Model import interrupted. Check the local service and retry."));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { reject(new Error("The service returned an unreadable import response.")); }
      } else {
        let message = `Model import failed (${xhr.status}).`;
        try { message = JSON.parse(xhr.responseText).error ?? message; } catch { /* Non-JSON transport errors have no safe detail. */ }
        reject(new Error(message));
      }
    };
    const form = new FormData();
    form.append("scene", scene);
    form.append("model", file, file.name);
    xhr.send(form);
  }),
};

export function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1048576) return `${(value / 1024).toFixed(0)} KB`;
  if (value < 1073741824) return `${(value / 1048576).toFixed(1)} MB`;
  return `${(value / 1073741824).toFixed(2)} GB`;
}
export function duration(value: number | null | undefined) {
  if (value == null) return "—";
  return value < 60 ? `${Math.round(value)}s` : `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}
export function date(value: string) {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime()) ? "Not run yet" : timestamp.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
export function displayName(project: Project) { return project.name && project.name !== project.id ? project.name : project.id.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }

export function downloadBlob(name: string, mime: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  URL.revokeObjectURL(url);
}

// ---- placement editing math -------------------------------------------------
// The editor mutates a local draft of the layout so the 3D view reacts on the
// same frame as the input; these pure helpers keep every clamp in one place.
export const MOVE_STEP_M = 0.1;
export const LIFT_STEP_M = 0.1;
export const RESIZE_STEP_M = 0.1;
export const SIZE_MIN_M = 0.2;
export const SIZE_MAX_M = 8;

export function isPoint(value: unknown): value is Point {
  return Array.isArray(value) && value.length === 3 && value.every((n) => typeof n === "number" && Number.isFinite(n));
}
const round = (value: number) => Math.round(value * 1000) / 1000;
export function clampSize(size: Point): Point {
  return size.map((value) => round(Math.min(SIZE_MAX_M, Math.max(SIZE_MIN_M, value)))) as Point;
}
/** Slide on the scanned floor plane: arrows follow the plan axes (up = -z). */
export function slidePlacement(p: Placement, dx: number, dz: number): Placement {
  return { ...p, center_xz: [round(p.center_xz[0] + dx), round(p.center_xz[1] + dz)] };
}
export function liftPlacement(p: Placement, dy: number): Placement {
  return { ...p, center_y: round(p.center_y + dy) };
}
export function yawPlacement(p: Placement, delta: number): Placement {
  return { ...p, yaw_deg: Math.round(((p.yaw_deg + delta) % 360 + 360) % 360) };
}
export function resizePlacement(p: Placement, axis: 0 | 1 | 2, delta: number): Placement {
  const next = [...p.size] as Point;
  next[axis] = Math.min(SIZE_MAX_M, Math.max(SIZE_MIN_M, round(next[axis] + delta)));
  const size = clampSize(next);
  return { ...p, size, footprint_m2: round(size[0] * size[2]) };
}
/** A copy dropped beside its source along the piece's own width axis. */
export function duplicatePlacement(p: Placement, identity: string): Placement {
  const a = (p.yaw_deg * Math.PI) / 180, gap = p.size[0] + 0.15;
  return {
    ...p, id: identity, label: `${p.label} copy`, model_revision: p.model_revision,
    center_xz: [round(p.center_xz[0] + gap * Math.cos(a)), round(p.center_xz[1] + gap * Math.sin(a))],
  };
}
/** Dragged onto the scanned surface: x/z from the ray hit, y only if the viewer sent one. */
export function moveToPlacement(p: Placement, point: Point): Placement {
  return { ...p, center_xz: [round(point[0]), round(point[2])], center_y: round(Number.isFinite(point[1]) ? point[1] : p.center_y) };
}
export function placementPoint(p: Placement): Point {
  return [p.center_xz[0], p.center_y, p.center_xz[1]];
}
/** A layout row the server has not confirmed yet: real shape, pending fit verdict. */
export function provisionalPlacement(source: FurnitureItem, point: Point, yaw_deg: number, model_revision: string, identity: string): Placement {
  const size = clampSize([...source.size] as Point);
  return {
    id: identity, item: source.item, label: source.label, size, center_xz: [round(point[0]), round(point[2])],
    center_y: round(point[1]), yaw_deg: Math.round(yaw_deg), scale: 1, footprint_m2: round(size[0] * size[2]),
    created_at: new Date().toISOString(), model_revision, stale: false,
    model: source.model,   // an imported item renders its real geometry immediately, before the ack
    fit: { supported: true, valid: true, fits_height: null, tight: false, floor_clearance_m: null, ceiling_height_m: null, reason: "Saved copy pending — the server fit check runs next." },
  };
}
function geometry(points: Point[], kind: MeasurementKind) {
  if (kind === "point" || points.length === 1) return { type: "Point", coordinates: points[0] };
  if (kind === "area" || kind === "volume") { const ring = [...points]; if (JSON.stringify(ring[0]) !== JSON.stringify(ring[ring.length - 1])) ring.push(ring[0]); return { type: "Polygon", coordinates: [ring] }; }
  return { type: "LineString", coordinates: points };
}
// Local viewer-frame export only — never faked as WGS84. Positions are Y-up metres
// with no surveyed CRS, and each feature carries its uncertainty + validity.
export function measurementsGeoJSON(measurements: Measurement[], scene: string) {
  return {
    type: "FeatureCollection", name: `${scene} measurements`,
    coordinate_reference_system: "local viewer Y-up (x, y, z) — not georeferenced; no surveyed CRS is claimed",
    features: measurements.map((m) => ({
      type: "Feature", id: `measure:${m.id}`,
      properties: { name: m.label, kind: m.kind, value: m.value, unit: m.unit, uncertainty_m: m.uncertainty?.m ?? null, valid: m.valid ?? true, reason: m.reason ?? null, stale: m.stale },
      geometry: geometry(m.points, m.kind),
    })),
  };
}
export function measurementsCSV(measurements: Measurement[]) {
  const esc = (v: unknown) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const rows = [["name", "kind", "value", "unit", "uncertainty_m", "valid", "reason", "stale"].join(",")];
  for (const m of measurements) rows.push([esc(m.label), esc(m.kind), esc(m.value), esc(m.unit), esc(m.uncertainty?.m ?? ""), esc(m.valid ?? true), esc(m.reason ?? ""), esc(m.stale)].join(","));
  return rows.join("\n");
}

// A placed-furniture layout as GeoJSON in the local viewer frame (never faked as
// WGS84). Each feature is the footprint ring with its fit verdict attached.
export function placementsGeoJSON(placements: Placement[], scene: string) {
  const ring = (p: Placement) => {
    const a = (p.yaw_deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    const [cx, cz] = p.center_xz, hu = p.size[0] / 2, hw = p.size[2] / 2;
    const corners = [[-hu, -hw], [hu, -hw], [hu, hw], [-hu, hw], [-hu, -hw]]
      .map(([u, v]) => [cx + u * ca - v * sa, cz + u * sa + v * ca]);
    return corners;
  };
  return {
    type: "FeatureCollection", name: `${scene} furniture layout`,
    coordinate_reference_system: "local viewer Y-up (x, z plan) — not georeferenced; nominal catalogue dimensions or file-read model size, not surveyed",
    features: placements.map((p) => ({
      type: "Feature", id: `place:${p.id}`,
      properties: { name: p.label, item: p.item, source: p.model ? "gltf-import" : "catalogue", height_m: p.size[1], footprint_m2: p.footprint_m2, yaw_deg: p.yaw_deg, true_size_m: p.model?.size ?? null, model_scale_status: p.model?.scale_status ?? null, fits: p.fit?.valid ?? false, reason: p.fit?.reason ?? null, stale: p.stale },
      geometry: { type: "Polygon", coordinates: [ring(p)] },
    })),
  };
}

export type ApplicationId = "military" | "urban" | "disaster" | "construction" | "border" | "inspection" | "archaeology" | "twin";
export const APPLICATIONS: Record<ApplicationId, string> = {
  military: "Military reconnaissance and mission planning", urban: "Urban planning and smart cities",
  disaster: "Disaster damage assessment", construction: "Construction progress monitoring",
  border: "Border and strategic area mapping", inspection: "Infrastructure inspection",
  archaeology: "Archaeological documentation", twin: "Digital twin generation",
};
