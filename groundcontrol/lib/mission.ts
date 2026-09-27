/**
 * Mission planning and rehearsal client (Phase 3). Missions are proposals of kind
 * "mission" written through the plan routes (lib/plan.ts); this adds the analysis, the
 * mission pack, the recorded rehearsal runs and the basemap the after-action review draws on.
 * Everything here is planned or simulated, never observed; the UI says so.
 */
import type { ExportFile, PlanMesh, PlanState, Proposal } from "./plan";

export type HostileExposure = { id: string; name: string; exposure_s: number; longest_exposure_s: number; first_seen_m: number | null; first_seen_s: number | null; closest_m: number };
export type RouteReport = {
  length_m: number; eta_s: number; climb_m: number; descent_m: number; max_slope_pct: number; exposure_s: number; exposed_m: number;
  dead_ground_pct: number; unscanned_m: number; per_hostile: HostileExposure[];
  waypoints: { name: string; time_s: number; station_m: number }[]; phase_lines: { label: string; station_m: number; time_s: number }[];
  profile: { station_m: number[]; height_m: number[]; seen_by: number[] }; basis: string;
};
export type MissionAnalysis = {
  hostiles: number; routes: { id: string; name: string; report: RouteReport }[]; overlay: PlanMesh[];
  ground: { watched_m2: number; dead_ground_m2: number }; notes: string[];
};
export type LosResult = { visible: boolean; blocked_at: [number, number, number] | null; distance_m: number };
export type HlzCandidate = { centre: [number, number]; usable_diameter_m: number; max_slope_deg: number; max_obstacle_m: number };
export type LabelCandidate = { role: "infantry" | "vehicle"; position: [number, number]; points: number; basis: string; source: "labels" | "detections" };
export type RunSummary = {
  duration_s: number; distance_m: number; exposure_s: number; exposed_pct: number; waypoints_reached: number;
  waypoint_times: { t: number; text: string }[]; hits_taken: number; casualty: number; neutralised: number; completed: boolean;
  per_bot: { id: number; label: string; exposure_s: number; first_seen_s: number | null; neutralised: boolean }[];
};
export type RunListing = { id: string; created_at: string; summary: RunSummary; conditions: Record<string, string> };
export type RunEvent = { t: number; type: string; text: string; bot?: number; at?: [number, number, number] };
export type Run = {
  id: string; mission: string; created_at: string; hz: number; player: number[][];
  bots: { id: number; label: string; behaviour: string; frames: number[][] }[]; events: RunEvent[]; route: [number, number][];
  conditions: Record<string, string>; summary: RunSummary; note: string;
};
export type ThreatCandidate = { position: [number, number]; score: number; overwatch_pct: number; height_above_route_m: number; concealment_pct: number; elevated: boolean; mgrs?: string };
export type ThreatMap = { candidates: ThreatCandidate[]; overlay: PlanMesh[]; route_samples: number; range_m: number; basis: string };
export type Obstacle = { position: [number, number]; height_m: number; top_y: number; area_m2: number; kind: string; mgrs?: string };
export type ObstacleList = { obstacles: Obstacle[]; count: number; min_height_m: number; basis: string };
export type Trafficability = { mobility: "foot" | "wheeled" | "tracked"; area_m2: { go: number; slow_go: number; no_go: number; unobserved: number };
  overlay: PlanMesh[]; basis: string };
export type SessionSnapshot = {
  session: string; name: string; seq: number; note: string; bots: number[][];
  players: { id: string; name: string; role: string; pose: number[] | null; alive: boolean; health: number | null; lost: boolean; seconds_since: number }[];
  commands: { seq: number; type: string }[];
};
export type Basemap = { png_base64: string; bounds: { x0: number; z0: number; x1: number; z1: number }; cell: number };
export type Conditions = { light: "day" | "dusk" | "night"; nvg: boolean; fog_m: number | null };

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

export const missionApi = {
  analysis: (scene: string, id: string) => request<MissionAnalysis>(`mission/analysis?${q(scene, `&id=${id}`)}`),
  los: (scene: string, a: number[], b: number[]) =>
    request<LosResult>(`mission/los?${q(scene, `&x1=${a[0]}&y1=${a[1]}&z1=${a[2]}&x2=${b[0]}&y2=${b[1]}&z2=${b[2]}`)}`),
  hlz: (scene: string, diameter: number) => request<{ candidates: HlzCandidate[]; note: string }>(`mission/hlz?${q(scene, `&diameter=${diameter}`)}`),
  candidates: (scene: string) => request<{ candidates: LabelCandidate[] }>(`mission/candidates?${q(scene)}`),
  coveredRoute: (scene: string, proposal: Proposal, route: string) =>
    request<PlanState & { suggested: { id: string; planned: RouteReport; covered: RouteReport } }>("mission/covered-route",
      { scene, id: proposal.id, revision: proposal.revision, route }),
  pack: (scene: string, id: string) => request<ExportFile>(`mission/pack?${q(scene, `&id=${id}`)}`),
  runs: (scene: string, id: string) => request<{ runs: RunListing[] }>(`mission/runs?${q(scene, `&id=${id}`)}`),
  run: (scene: string, id: string, run: string) => request<Run>(`mission/run?${q(scene, `&id=${id}&run=${run}`)}`),
  deleteRun: (scene: string, id: string, run: string) => request<{ runs: RunListing[] }>("mission/run/delete", { scene, id, run }),
  basemap: (scene: string) => request<Basemap>(`mission/basemap?${q(scene)}`),
  threat: (scene: string, id: string, route: string, range: number) => request<ThreatMap>(`mission/threat?${q(scene, `&id=${id}&route=${route}&range=${range}`)}`),
  obstacles: (scene: string, minHeight: number) => request<ObstacleList>(`mission/obstacles?${q(scene, `&min_height=${minHeight}`)}`),
  trafficability: (scene: string, mobility: string) => request<Trafficability>(`mission/trafficability?${q(scene, `&mobility=${mobility}`)}`),
  openSession: (scene: string, id: string, name: string) =>
    request<{ session: string; token: string; links: { path: string; lan: string[]; note: string } }>("mission/session/open", { scene, id, name }),
  sessionSnapshot: (scene: string, id: string, session: string) => request<SessionSnapshot>(`mission/session?${q(scene, `&id=${id}&session=${session}`)}`),
  sessionCommand: (scene: string, id: string, session: string, command: Record<string, unknown>) =>
    request<{ command: { seq: number } }>("mission/session/command", { scene, id, session, command }),
  aarPdf: (scene: string, id: string, run: string) => request<ExportFile>(`mission/aar-pdf?${q(scene, `&id=${id}&run=${run}`)}`),
};

/** Query string the arena viewer reads to rehearse a mission (see viewer/pc.js). */
export function rehearsalQuery(scene: string, missionId: string, c: Conditions) {
  const p = new URLSearchParams({ mission: missionId, scene, light: c.light });
  if (c.nvg) p.set("nvg", "1");
  if (c.fog_m) p.set("fog", String(c.fog_m));
  return p.toString();
}

export const fmtClock = (seconds: number) => { const s = Math.max(0, Math.round(seconds)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
