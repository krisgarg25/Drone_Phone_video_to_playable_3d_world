import type { ActiveJob, Preset, Quality, StepRow } from "./api";

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
export type FurnitureItem = { item: string; label: string; size: [number, number, number]; footprint_m2: number };
export type PlacementFit = {
  supported: boolean; valid: boolean; fits_height: boolean; tight: boolean;
  floor_clearance_m: number | null; ceiling_height_m: number | null; reason: string | null;
};
export type Placement = {
  id: string; item: string; label: string; size: [number, number, number];
  center_xz: [number, number]; center_y: number; yaw_deg: number; scale: number;
  footprint_m2: number; created_at: string; model_revision: string; stale: boolean; fit: PlacementFit;
};
export type Project = {
  id: string; name: string; workflow: Workflow; capture: Capture;
  status: "ready" | "processing" | "failed" | "uploaded" | "empty";
  updated: string; thumbnail_url: string | null; video_count: number;
  frame_count: number; registered_count: number; viewable: boolean; trained: boolean;
  scale: { status: "relative" | "estimated" | "metric"; source: string; unit: string };
  georeference: { status: "local" | "georeferenced"; crs: string | null; vertical_datum?: string | null; position_reference?: string | null; viewer_frame?: string };
  accuracy: { status: "unverified" | "verified"; rmse_m: number | null; horizontal_rmse_m?: number | null; vertical_rmse_m?: number | null; checkpoints?: number | null; basis?: string };
  survey?: { status: string; blockers: string[]; fit_rmse_m: number | null } | null;
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
};
export type ProjectList = {
  projects: Project[]; job: ActiveJob; presets: Record<string, Preset>; qualities: Record<string, Quality>;
};
export type RunOptions = {
  scene: string; preset: string; quality: string; action?: "run" | "scan";
  engine?: "pipeline" | "survey"; dense_profile?: string;
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
  save: (body: { scene: string; name?: string; workflow?: Workflow; capture?: Capture; notes?: string }) => request<ProjectDetail>("project", body),
  run: (body: RunOptions) => request<{ status: string }>("run", body),
  cancel: (scene: string) => request<{ status: string }>("cancel", { scene }),
  measure: (scene: string, measurement: { label: string; kind: MeasurementKind; points: Point[]; model_revision: string }) => request<ProjectDetail>("measurements", { scene, ...measurement }),
  removeMeasurement: (scene: string, id: string) => request<ProjectDetail>("measurements/delete", { scene, id }),
  place: (scene: string, body: { item: string; point: Point; yaw_deg?: number; scale?: number; label?: string; model_revision: string }) => request<ProjectDetail>("placements", { scene, ...body }),
  movePlacement: (scene: string, body: { id: string; item: string; point: Point; yaw_deg?: number; scale?: number; model_revision: string }) => request<ProjectDetail>("placements/update", { scene, ...body }),
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
    coordinate_reference_system: "local viewer Y-up (x, z plan) — not georeferenced; nominal catalogue dimensions, not surveyed",
    features: placements.map((p) => ({
      type: "Feature", id: `place:${p.id}`,
      properties: { name: p.label, item: p.item, height_m: p.size[1], footprint_m2: p.footprint_m2, yaw_deg: p.yaw_deg, fits: p.fit?.valid ?? false, reason: p.fit?.reason ?? null, stale: p.stale },
      geometry: { type: "Polygon", coordinates: [ring(p)] },
    })),
  };
}
