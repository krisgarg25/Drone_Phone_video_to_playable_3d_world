/**
 * Typed client for the reconstruction backend.
 *
 * Everything goes through a same-origin Next route handler (`/api/backend/*`)
 * which proxies to the Python server. That keeps the console free of CORS
 * special-casing and keeps the write guard on one process boundary the operator
 * can see: the Python side refuses GPU execution over the network by design, and
 * this client never pretends otherwise.
 *
 * When the backend is not running, `load()` returns `{ live: false }` and every
 * screen renders that state instead of inventing numbers.
 */

const BACKEND = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8137";
/**
 * Server components have no request origin, so a relative fetch throws
 * "Failed to parse URL". In the browser a relative path is the only correct
 * answer: it follows whatever origin actually served the page instead of a
 * port baked in at build time.
 */
const APP = process.env.NEXT_PUBLIC_APP_URL ?? "http://127.0.0.1:3000";
const via = (path: string) =>
  `${typeof window === "undefined" ? APP : ""}/api/backend${path}`;

export type Readiness = { label: string; status: string; detail: string };
export type Criterion = {
  id: string; label: string; weight: number; status: string; reason: string;
  metrics?: Record<string, unknown>;
};
export type Artifact = { name: string; url: string; kind: string };
export type FormatRow = { format: string; status: string; path?: string; reason?: string; geometry?: string };

/** Which parameters this flight's geometry can separate at all. `false` is not a
 *  failure to calibrate — it means the trajectory cannot support the claim. */
export type Identifiability = {
  clock_offset: boolean; lever_arm_along_track: boolean; lever_arm_cross_track: boolean;
  rotation_about_trajectory_axis: boolean; metric_scale: boolean;
};

/**
 * The GPS diagnostics `survey_gnss` already wrote into the preparation manifest.
 *
 * Every figure here is a *reported* per-axis standard deviation or a *bound*, never a
 * measured error: there is no covariance matrix in `survey_gnss` (uncertainties are
 * scalars per axis), the clock offset is bounded but never estimated
 * (`offset_estimate_s` is always null by design), and `accuracy_validated` is false in
 * every sub-report because validating accuracy needs an independent surveyed
 * reference. A screen that renders one of these numbers as a measurement has
 * misread it.
 */
export type GnssReport = {
  accuracy_validated: boolean;
  std_basis: string;
  quality: {
    count: number; duration_s: number; path_length_m: number; median_step_m: number | null;
    median_horizontal_std_m: number; median_vertical_std_m: number;
    displacement_below_noise: boolean; suspicious_count: number;
    gaps_over_threshold_count: number; gap_threshold_s: number; max_speed_m_s: number;
    warnings: string[]; speeds_are_implied_not_measured: boolean; accuracy_validated: boolean;
  } | null;
  clock: {
    status: string; convention: string;
    offset_bounds_s: [number, number] | null; offset_width_s: number | null;
    worst_case_along_track_error_m: number | null; infeasibility_s: number | null;
    max_speed_m_s: number; offset_estimate_s: number | null;
    estimates_offset: boolean; warnings: string[];
  } | null;
  fix_quality: {
    status: string; quality_known: boolean; hdop_supplied: boolean; fallback: string | null;
    all_rows_usable: boolean; dop_scaling: string;
    measured_on_this_hardware: boolean; precision_invented: boolean; warnings: string[];
  } | null;
  observability: {
    status: string; geometry: string | null; identifiability: Identifiability | null;
    georef_rejects_this_geometry: boolean;
    never_claim: string[]; warnings: string[];
    estimates_anything: boolean; accuracy_validated: boolean;
  } | null;
};

export type SurveyState = {
  scene: string;
  status: string;
  readiness: Readiness[];
  evaluation: { criteria: Criterion[] };
  alignment: Record<string, unknown> | null;
  gnss?: GnssReport | null;
  artifacts: Artifact[];
  blockers: string[];
  commands?: { label: string; argv: string[]; requires_gpu: boolean }[];
  crs?: string;
  vertical_datum?: string;
  position_reference?: string;
  latest_run?: { id: string; status: string; secs?: number; error?: string };
  measurements?: unknown[];
  /** The survey path as six rows, derived server-side from the validated status. */
  steps?: SurveyStep[];
  geoid?: { model: string; available: boolean };
  [key: string]: unknown;
};

export type SurveyStepStatus =
  "done" | "ready" | "needs_you" | "running" | "failed" | "optional" | "waiting";
export type SurveyStep = {
  id: "inputs" | "prepare" | "reconstruct" | "align" | "checkpoints" | "evaluate";
  label: string; status: SurveyStepStatus; detail: string;
};
export type AdvanceResult = {
  trace: { stage: string; status: string; detail?: string }[];
  state: SurveyState;
};

export type Loaded =
  | { live: false; error: string }
  | { live: true; state: SurveyState; scenes: string[] };

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(via(path), {
    cache: "no-store",
    ...init,
    headers: { accept: "application/json", ...(init?.headers ?? {}) },
  });
  const text = await response.text();
  if (!response.ok) {
    let detail = text.slice(0, 400);
    try { detail = (JSON.parse(text) as { error?: string }).error ?? detail; } catch { /* keep raw */ }
    throw new Error(`${response.status} ${detail}`);
  }
  return JSON.parse(text) as T;
}

export async function load(scene: string): Promise<Loaded> {
  try {
    const [state, scenes] = await Promise.all([
      get<SurveyState>(`/api/survey?scene=${encodeURIComponent(scene)}`),
      get<{ scenes?: { name: string }[] | string[] }>("/api/scenes").catch(() => ({ scenes: [] })),
    ]);
    const list = (scenes.scenes ?? []).map((entry) =>
      typeof entry === "string" ? entry : entry.name);
    return { live: true, state, scenes: list.length ? list : [scene] };
  } catch (error) {
    return { live: false, error: error instanceof Error ? error.message : String(error) };
  }
}

const post = <T,>(path: string, body: Record<string, unknown>) => get<T>(path, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

/** CPU-only actions. GPU reconstruction stays behind the run dialog's explicit approval. */
export async function act(action: "inputs" | "prepare" | "align" | "evaluate" | "checkpoints",
                          body: Record<string, unknown>) {
  return post<SurveyState>(`/api/survey/${action}`, body);
}

/** Runs every CPU step that can run now, in order; never starts GPU work. */
export async function advance(scene: string) {
  return post<AdvanceResult>("/api/survey/advance", { scene });
}

export async function surveyStatus(scene: string) {
  return get<SurveyState>(`/api/survey?scene=${encodeURIComponent(scene)}`);
}

/* --------------------------------------------------------------------------
   The pipeline.py lane.

   A different backend surface from the survey one above, and a different
   authority: these routes DO start the reconstruction, including its GPU
   stages, because `pipeline.py run` is what the operator's own machine was
   always going to execute. The server serialises it behind one run lock, so
   "a job is already running" is a real answer and not a race.
   -------------------------------------------------------------------------- */

export type StepRow = {
  i: number; name: string;
  status: "done" | "failed" | "running" | "recovered" | "interrupted" | string;
  secs: number | null; exit: number | null; attempts: number | null; kind: string | null;
};

export type SceneRow = {
  name: string; has_work: boolean; has_video: boolean;
  viewable: boolean; trained: boolean;
  registered: [number, number | null] | null;
  running: boolean; updated: string; steps: StepRow[];
};

export type ActiveJob = {
  status: string; scene: string; step: string;
  preset?: string; quality?: string; cmd?: string; logs?: string[];
};

export type ServerInfo = {
  local_ip: string; port: number; https_port: number;
  dashboard: string; scenes: string[]; active_job: ActiveJob;
};

export type Preset = { label?: string; advice?: string; [key: string]: unknown };
export type Quality = { target: number | null; width: number; steps: number; cap: number; voxel: string };

export type Tail = { cursor: string; current: string; lines: string[]; running: boolean };

export type UploadResult = { status: string; scene: string; saved_files: string[] };

const json = async <T,>(response: Response): Promise<T> => {
  const text = await response.text();
  if (!response.ok) {
    let detail = text.slice(0, 400);
    try { detail = (JSON.parse(text) as { error?: string }).error ?? detail; } catch { /* keep raw */ }
    throw new Error(`${response.status} ${detail}`);
  }
  return JSON.parse(text) as T;
};

export const pipeline = {
  info: () => get<ServerInfo>("/api/info"),
  scenes: () => get<{ scenes: SceneRow[] }>("/api/scenes"),
  status: () => get<ActiveJob>("/api/status"),
  presets: () => get<{ presets: Record<string, Preset>; qualities: Record<string, Quality> }>("/api/presets"),
  tail: (scene: string, cursor: string) =>
    get<Tail>(`/api/tail?scene=${encodeURIComponent(scene)}&cursor=${encodeURIComponent(cursor)}`),

  /** Starts a real reconstruction on the operator's own machine, GPU stages included. */
  start: (scene: string, preset: string, quality: string, extra: string[] = []) =>
    get<{ status: string; scene: string; preset: string }>("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ scene, preset, quality, extra_args: extra }),
    }),

  kill: () => get<{ status: string }>("/api/kill", { method: "POST" }),

  /** The server's multipart parser wants the browser FormData shape: a `scene` field plus files. */
  upload: async (scene: string, files: File[]): Promise<UploadResult> => {
    const form = new FormData();
    form.append("scene", scene);
    for (const file of files) form.append("file", file, file.name);
    return json<UploadResult>(await fetch(via("/api/upload"), { method: "POST", body: form, cache: "no-store" }));
  },
};

export const backendOrigin = BACKEND;

/** Reads the delivery ledger out of the artifacts a run published. */
export function formatsFromArtifacts(artifacts: Artifact[]): FormatRow[] {
  const has = (match: (name: string) => boolean) => artifacts.some((a) => match(a.name));
  const row = (format: string, match: (name: string) => boolean): FormatRow =>
    has(match) ? { format, status: "delivered", path: format } : { format, status: "not_delivered" };
  return [
    row("obj", (n) => n.endsWith(".obj")),
    row("ply", (n) => n.endsWith(".ply")),
    row("las", (n) => n.endsWith(".las")),
    row("geotiff", (n) => n.endsWith(".tif") || n.endsWith(".tiff")),
    row("glb/gltf", (n) => n.endsWith(".gltf") || n.endsWith(".glb")),
    row("fbx", (n) => n.endsWith(".fbx")),
  ];
}
