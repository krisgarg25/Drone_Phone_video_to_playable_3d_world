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

export type SurveyState = {
  scene: string;
  status: string;
  readiness: Readiness[];
  evaluation: { criteria: Criterion[] };
  alignment: Record<string, unknown> | null;
  artifacts: Artifact[];
  blockers: string[];
  commands?: { label: string; argv: string[]; requires_gpu: boolean }[];
  crs?: string;
  vertical_datum?: string;
  position_reference?: string;
  latest_run?: { id: string; status: string; secs?: number; error?: string };
  measurements?: unknown[];
  [key: string]: unknown;
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

/** CPU-only actions. GPU reconstruction stays on the CLI behind --allow-gpu. */
export async function act(action: "inputs" | "prepare" | "align" | "evaluate",
                          body: Record<string, unknown>) {
  return get<SurveyState>(`/api/survey/${action}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
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
