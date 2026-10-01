import type { IconName } from "@/components/studio-icons";
import { inspectApi, twinApi } from "./inspect";
import { missionApi } from "./mission";
import { plan, type ExportFile } from "./plan";
import { measurementsCSV, measurementsGeoJSON, placementsGeoJSON, type ProjectDetail } from "./workspace";

/**
 * Everything a project can hand over, as one list: files the pipeline already wrote,
 * and products made on request (measurements, plans, reports, packs). Each item knows
 * its formats and how to produce a Blob, so the export page can rename, bundle or
 * download any selection without caring where the bytes come from.
 */
export type ExportGroup = "models" | "survey" | "analysis" | "reports" | "technical";
export type ExportFormat = { id: string; label: string; ext: string };
export type ExportItem = {
  id: string; group: ExportGroup; title: string; detail: string; icon: IconName;
  source: "file" | "made";
  formats: ExportFormat[]; defaultName: string; bytes: number | null;
  build: (format: string) => Promise<Blob>;
};

export const GROUPS: { id: ExportGroup; label: string; what: string; icon: IconName }[] = [
  { id: "models", label: "3D models", what: "The reconstruction itself: photo-real splats, meshes and point clouds.", icon: "cube" },
  { id: "survey", label: "Measurements & layouts", what: "What you measured and placed, ready for GIS or a spreadsheet.", icon: "ruler" },
  { id: "analysis", label: "Plans, missions & packages", what: "Drawn schemes, mission packs and engine-ready packages, built when you download.", icon: "building" },
  { id: "reports", label: "Reports & records", what: "Inspection reports, quality checks and processing records.", icon: "file" },
  { id: "technical", label: "Technical files", what: "Collision and navigation geometry the viewer uses. Most people do not need these.", icon: "settings" },
];

const ext = (name: string) => (name.match(/\.([a-z0-9]+)$/i)?.[1] ?? "bin").toLowerCase();
const stem = (name: string) => name.split("/").pop()!.replace(/\.[a-z0-9]+$/i, "");
const safe = (text: string) => text.replace(/[^A-Za-z0-9._ -]+/g, "-").replace(/\s+/g, "_").replace(/^[-_.]+|[-_.]+$/g, "").slice(0, 120) || "file";
export const cleanName = safe;

async function fetchBlob(url: string) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not read ${url.split("?")[0].split("/").pop()} (${response.status}).`);
  return response.blob();
}
function fileBlob(file: ExportFile) {
  const body = file.encoding === "base64" ? Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0)) : file.content;
  return new Blob([body], { type: file.media_type });
}
const text = (value: string, type: string) => new Blob([value], { type });

const MODEL_FORMAT: Record<string, { title: string; detail: string }> = {
  "splat.ply": { title: "Photo-real model", detail: "Gaussian splats, the model you see in the viewer" },
  "scene.ply": { title: "Point cloud", detail: "Coloured points for CloudCompare or QGIS" },
  "semantics.ply": { title: "Labelled point cloud", detail: "Points tagged ground, road, building, tree, obstacle" },
  "textured.obj": { title: "Textured mesh (OBJ)", detail: "Surface with photo texture; bring the .mtl and .jpg" },
  "textured.gltf": { title: "Textured mesh (glTF)", detail: "Surface with photo texture; bring the .bin and .jpg" },
  "nav.obj": { title: "Walkable surface", detail: "Simplified ground used for walking and navigation" },
};

function describe(name: string, kind: string) {
  const base = name.split("/").pop()!;
  const known = MODEL_FORMAT[base];
  if (known) return known;
  if (kind.startsWith("textured")) return { title: base, detail: "Part of the textured mesh" };
  if (/collision|surface_|regress_|collider|voxel/.test(name)) return { title: base, detail: "Collision geometry used for measuring and walking" };
  if (kind === "report") return { title: base, detail: "Processing record" };
  if (kind === "rooms") return { title: base, detail: "Detected rooms and floors" };
  return { title: base, detail: kind };
}

export async function exportItems(project: ProjectDetail): Promise<ExportItem[]> {
  const scene = project.id;
  const items: ExportItem[] = [];

  // Files the pipeline wrote. The ones people ask for lead; plumbing trails.
  const known = (name: string) => Object.keys(MODEL_FORMAT).indexOf(name.split("/").pop()!);
  const rank = (name: string) => (known(name) < 0 ? 99 : known(name));
  const ranked = [...project.artifacts].sort((a, b) => rank(a.name) - rank(b.name));
  for (const file of ranked) {
    const e = ext(file.name);
    const { title, detail } = describe(file.name, file.kind);
    const group: ExportGroup = file.kind === "report" || e === "json" || e === "jsonl" ? "reports"
      : known(file.name) >= 0 || file.kind.startsWith("textured") ? "models" : "technical";
    items.push({
      id: `file:${file.name}`, group, title, detail, icon: group === "reports" ? "file" : e === "ply" ? "grid" : "cube", source: "file",
      formats: [{ id: e, label: e.toUpperCase(), ext: e }], defaultName: safe(`${scene}_${stem(file.name)}`), bytes: file.bytes,
      build: () => fetchBlob(file.url.includes("download=1") ? file.url : `${file.url}${file.url.includes("?") ? "&" : "?"}download=1`),
    });
  }

  if (project.measurements.length) items.push({
    id: "made:measurements", group: "survey", title: "Measurements", detail: `${project.measurements.length} saved distance, height, area and volume readings`, icon: "ruler", source: "made",
    formats: [{ id: "geojson", label: "GeoJSON", ext: "geojson" }, { id: "csv", label: "CSV", ext: "csv" }, { id: "json", label: "JSON", ext: "json" }],
    defaultName: safe(`${scene}_measurements`), bytes: null,
    build: async (format) => format === "csv" ? text(measurementsCSV(project.measurements), "text/csv")
      : format === "json" ? text(JSON.stringify({ scene, coordinate_system: "viewer Y-up", scale: project.scale, measurements: project.measurements }, null, 2), "application/json")
        : text(JSON.stringify(measurementsGeoJSON(project.measurements, scene), null, 2), "application/geo+json"),
  });
  if (project.placements.length) items.push({
    id: "made:layout", group: "survey", title: "Furniture layout", detail: `${project.placements.length} placed items with footprints`, icon: "cube", source: "made",
    formats: [{ id: "geojson", label: "GeoJSON", ext: "geojson" }], defaultName: safe(`${scene}_layout`), bytes: null,
    build: async () => text(JSON.stringify(placementsGeoJSON(project.placements, scene), null, 2), "application/geo+json"),
  });

  const [plans, missions, register] = await Promise.all([
    plan.list(scene, "plan").then((r) => r.index.proposals).catch(() => []),
    plan.list(scene, "mission").then((r) => r.index.proposals).catch(() => []),
    inspectApi.register(scene).then((r) => r.items.length).catch(() => 0),
  ]);
  for (const scheme of plans) items.push({
    id: `made:plan:${scheme.id}`, group: "analysis", title: `Plan · ${scheme.name}`, detail: "Drawn buildings, roads and zones", icon: "building", source: "made",
    formats: [{ id: "geojson", label: "GeoJSON", ext: "geojson" }, { id: "cityjson", label: "CityJSON", ext: "json" }, { id: "dxf", label: "DXF (CAD)", ext: "dxf" }, { id: "3dtiles", label: "3D Tiles", ext: "zip" }],
    defaultName: safe(`${scene}_${scheme.name}`), bytes: null,
    build: async (format) => fileBlob(await plan.exportAs(scene, scheme.id, format as "geojson" | "cityjson" | "dxf" | "3dtiles")),
  });
  for (const mission of missions) items.push({
    id: `made:mission:${mission.id}`, group: "analysis", title: `Mission pack · ${mission.name}`, detail: "Symbols, routes and phase lines for Google Earth and handheld GPS", icon: "flag", source: "made",
    formats: [{ id: "kmz", label: "KMZ + GPX", ext: "zip" }], defaultName: safe(`${scene}_${mission.name}_pack`), bytes: null,
    build: async () => fileBlob(await missionApi.pack(scene, mission.id)),
  });
  if (project.viewable) items.push({
    id: "made:twin", group: "analysis", title: "Engine package", detail: "GLB, FBX and 3D Tiles for Unity, Unreal or Cesium (built on download)", icon: "twin", source: "made",
    formats: [{ id: "zip", label: "ZIP", ext: "zip" }], defaultName: safe(`${scene}_twin_package`), bytes: null,
    build: async () => fetchBlob((await twinApi.package(scene, false)).url),
  });
  if (register) items.push({
    id: "made:inspection", group: "reports", title: "Inspection report", detail: `${register} logged defect${register === 1 ? "" : "s"} with photos and positions`, icon: "search", source: "made",
    formats: [{ id: "pdf", label: "PDF", ext: "pdf" }], defaultName: safe(`${scene}_inspection_report`), bytes: null,
    build: async () => fileBlob(await inspectApi.report(scene)),
  });
  return items;
}

export function bytes(value: number | null) {
  if (value == null) return "on download";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}
