import type { IconName } from "@/components/studio-icons";
import type { ApplicationId } from "./workspace";

/**
 * The eight SIH26158 applications as the studio presents them. The registry fields
 * (analyses, workspace tabs, products, acceptance checks, capture patterns) mirror
 * `scripts/applications.py`, which is what the backend records against a run; keep
 * the two in step. `summary`, `icon` and `hue` are presentation only.
 */
export type ApplicationProfile = {
  id: ApplicationId; label: string; summary: string; icon: IconName; hue: number;
  analyses: string[]; tabs: string[]; products: string[]; checks: string[]; patterns: string[];
};

export const WORKSPACE_TABS: Record<string, { label: string; icon: IconName }> = {
  layers: { label: "Explore", icon: "compass" }, measure: { label: "Measure", icon: "ruler" },
  inspect: { label: "Inspect", icon: "search" }, ops: { label: "Operations", icon: "radar" },
  twin: { label: "Twin", icon: "twin" }, plan: { label: "Plan", icon: "building" },
  mission: { label: "Mission", icon: "flag" }, place: { label: "Place", icon: "cube" }, walk: { label: "Walk", icon: "play" },
};

export const OFFICIAL_FORMATS = ["obj", "ply", "las", "geotiff", "glb/gltf", "fbx"];

export const APPLICATION_PROFILES: ApplicationProfile[] = [
  {
    id: "military", label: "Military reconnaissance & mission planning", icon: "shield", hue: 140,
    summary: "Target coordinates, line of sight and route exposure, then a first-person rehearsal of the plan with an after-action review.",
    analyses: ["MGRS coordinates", "line of sight", "viewsheds", "route exposure", "HLZ", "threat heatmap", "obstacles", "trafficability", "detections", "mission pack"],
    tabs: ["layers", "measure", "mission", "ops", "walk"], products: ["las", "geotiff", "glb/gltf", "ply"],
    checks: ["time to first target coordinate", "CE90 vs checkpoints", "planted person detected", "route exposure computed", "rehearsal + AAR"],
    patterns: ["corridor"],
  },
  {
    id: "urban", label: "Urban planning & smart cities", icon: "building", hue: 212,
    summary: "Buildings, roads and vegetation as objects. Draw a proposal on the scan and test it against the rules and the sun.",
    analyses: ["building inventory", "facade completeness", "roads", "buildings", "rule checks", "shadow study", "before/after", "CityJSON / 3D Tiles"],
    tabs: ["layers", "measure", "plan", "twin", "walk"], products: [...OFFICIAL_FORMATS],
    checks: ["5 building heights within 0.5 m", "road impact table", "rule violation fixed by edit", "swipe before/after", "completeness per class"],
    patterns: ["orbit"],
  },
  {
    id: "disaster", label: "Disaster damage assessment", icon: "alert", hue: 4,
    summary: "A first map while the drone is still in the air, then change, damage grade, debris volume and blocked roads.",
    analyses: ["first map", "change", "damage grade", "debris volume", "detections", "blocked roads", "vehicle route", "flood", "field pack"],
    tabs: ["layers", "measure", "ops", "plan"], products: ["geotiff", "las", "ply"],
    checks: ["time to first map", "debris volume within 10%", "people absent from mesh, present as points"],
    patterns: ["grid"],
  },
  {
    id: "construction", label: "Construction progress monitoring", icon: "crane", hue: 38,
    summary: "Stockpile volumes, cut and fill against the design surface, and what changed between two flights.",
    analyses: ["stockpile volume", "cut/fill vs design", "epoch change", "zone progress", "design deviation", "crane clearance", "report"],
    tabs: ["layers", "measure", "ops", "plan", "twin"], products: ["las", "geotiff", "ply", "obj"],
    checks: ["change above LoD found, unchanged ground clean", "pile volume within 5%"],
    patterns: ["grid"],
  },
  {
    id: "border", label: "Border & strategic area mapping", icon: "route", hue: 268,
    summary: "Long corridors aligned from a single pass, change flagged with MGRS, and blind spots between surveillance posts.",
    analyses: ["straight-track alignment", "tiles", "change with MGRS", "profile", "blind spots", "surveillance posts", "KLV ingest"],
    tabs: ["layers", "measure", "ops", "mission"], products: ["geotiff", "las"],
    checks: [">= 1 km corridor aligned", "CE90 vs checkpoints", "planted change found", "blind-spot map from two posts"],
    patterns: ["corridor"],
  },
  {
    id: "inspection", label: "Infrastructure inspection", icon: "wrench", hue: 190,
    summary: "Trace every defect back to the frames that saw it. Tilt, wire sag, crack candidates and deformation, then a report.",
    analyses: ["frames that saw a point", "defect register", "crack candidates", "tilt", "wire sag + clearance", "M3C2 deformation", "inspection report"],
    tabs: ["layers", "measure", "inspect", "plan", "twin"], products: ["obj", "ply", "las", "glb/gltf"],
    checks: ["5 tape lengths within 2%", "defect traced to its frames", "report generated"],
    patterns: ["orbit", "corridor"],
  },
  {
    id: "archaeology", label: "Archaeological documentation", icon: "columns", hue: 24,
    summary: "A textured record with hillshade and relief models, sections and annotations, and honest marks where nothing was observed.",
    analyses: ["textured mesh", "ortho", "hillshade", "local relief model", "sections", "annotations", "observed/unobserved layers", "hypothesis reconstruction", "archive record", "seasonal M3C2"],
    tabs: ["layers", "measure", "inspect", "plan", "twin", "walk"], products: [...OFFICIAL_FORMATS],
    checks: ["textured mesh, hillshade, a section, unobserved regions marked", "hold-out texture check"],
    patterns: ["orbit", "grid"],
  },
  {
    id: "twin", label: "Digital twin generation", icon: "twin", hue: 256,
    summary: "Visual, measured and evidence views of one model, with an asset inventory, epochs and an engine-ready package.",
    analyses: ["visual / measured / evidence views", "asset inventory + attributes", "epochs", "3D Tiles LOD", "engine package", "proposals"],
    tabs: ["layers", "measure", "twin", "plan", "place", "walk"], products: [...OFFICIAL_FORMATS],
    checks: ["splat and mesh in one frame", "FBX / glTF / 3D Tiles open in an engine"],
    patterns: ["orbit"],
  },
];

export function workspaceHref(scene: string, tab: string) {
  const base = `/projects/${encodeURIComponent(scene)}`;
  return tab === "layers" ? base : `${base}/${tab}`;
}
