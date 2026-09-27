"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { pipeline } from "@/lib/api";
import {
  duplicatePlacement, duration, isPoint, liftPlacement, moveToPlacement, placementPoint, placementsGeoJSON,
  provisionalPlacement, resizePlacement, slidePlacement, WORKFLOWS, workspace, downloadBlob, yawPlacement,
  type Layer, type MeasurementKind, type Mode, type Placement, type Point, type ProjectDetail, type ProjectList,
} from "@/lib/workspace";
import { Studio, Status, Offline } from "./studio";
import { Icon, type IconName } from "./studio-icons";
import { RunDialog } from "./run-dialog";
import { ViewerPanel } from "./viewer-panel";
import { MeasurementPanel } from "./measurement-panel";
import { PlacementPanel } from "./placement-panel";
import { WalkthroughPanel } from "./walkthrough-panel";
import { DetailsPanel, ExportPanel, GnssReference } from "./project-details";
import { QualityPanel } from "./quality-panel";
import { PlanPanel, type PlanView } from "./plan-panel";
import { MissionPanel } from "./mission-panel";
import { AarReplay } from "./aar-replay";
import { SandTable } from "./sand-table";
import { InstructorView } from "./instructor-view";
import { OpsPanel } from "./ops-panel";
import { InspectPanel, INSPECT_TOOLS, type InspectTool } from "./inspect-panel";
import { TwinPanel, type TwinView } from "./twin-panel";
import { opsViewerPlan, TOOL_HINT, type OpsTool } from "@/lib/ops";
import {
  missionApi, rehearsalQuery,
  type Basemap, type Conditions, type HlzCandidate, type LabelCandidate, type LosResult, type MissionAnalysis, type ObstacleList, type Run, type RunListing,
  type ThreatMap, type Trafficability,
} from "@/lib/mission";
import {
  applyEdit, draftFeature, draftMissionFeature, editShape, plan, viewerPlan, type SymbolRole,
  type Catalogue, type ExistingBuilding, type ExportFormat, type ImportReport, type PlanFeature, type PlanState, type PlanTool,
  type FacadeResult, type PlanMesh, type ProposalIndex, type ShadowResult, type ShadowSettings, type XZ,
} from "@/lib/plan";
import "./workspace-modes.css";

const LAYERS: { key: Layer; label: string; detail: string; icon: IconName }[] = [
  { key: "splats", label: "Photorealistic model", detail: "Gaussian splats", icon: "cube" },
  { key: "cameras", label: "Camera positions", detail: "Recovered capture trajectory", icon: "camera" },
  { key: "points", label: "Sparse point cloud", detail: "Triangulated feature points", icon: "grid" },
  { key: "coverage", label: "Observed coverage", detail: "View-support diagnostic", icon: "layers" },
  { key: "collider", label: "Collision surface", detail: "Navigation geometry · estimated", icon: "compass" },
  { key: "semantics", label: "Semantic classes", detail: "Ground · road · building · vegetation · obstacle", icon: "layers" },
];
const INITIAL_LAYERS: Record<Layer, boolean> = { splats: true, cameras: false, points: false, coverage: false, collider: false, semantics: false };
const MODES = ["measure", "place", "plan", "mission", "ops", "inspect", "twin", "walk"];
/** Tabs whose panel draws its own overlays and owns picking (no measurements, no frame strip). */
const PANEL_TABS = ["plan", "mission", "ops", "inspect", "twin"];
const EMPTY_PLAN = { features: [], clips: [], view: "existing", selected: null };
type Orbit = { target: [number, number, number]; distance: number; yaw: number; pitch: number };
/** Fly/walk compare: an eye position and forward vector relayed between the two viewers. */
const isPose = (p: unknown): p is { eye: number[]; forward: number[] } => !!p && typeof p === "object"
  && [(p as { eye: unknown }).eye, (p as { forward: unknown }).forward].every((v) => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n)));
const isOrbit = (o: unknown): o is Orbit => !!o && typeof o === "object" && Array.isArray((o as Orbit).target) && (o as Orbit).target.length === 3
  && [...(o as Orbit).target, (o as Orbit).distance, (o as Orbit).yaw, (o as Orbit).pitch].every((v) => typeof v === "number" && Number.isFinite(v));
const DEFAULT_SHADOW: ShadowSettings = { mode: "instant", date: `${new Date().getFullYear()}-12-21`, time: "10:00",
  utc_offset: -new Date().getTimezoneOffset() / 60, start: 9, end: 15, step: 30, north_deg: 0 };

/** One local intent waiting for the backend. Latest write wins per piece. */
type PlacementOp = { kind: "add" | "update"; id: string; draft: Placement } | { kind: "delete"; id: string };
type SaveState = "clean" | "queued" | "saving" | "error";
const SAVE_COPY: Record<SaveState, string> = { clean: "Layout saved.", queued: "Unsaved changes.", saving: "Saving layout…", error: "Last save failed — changes are held." };

export function ProjectWorkspace({ scene, initialTab = "layers" }: { scene: string; initialTab?: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [options, setOptions] = useState<ProjectList | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState(initialTab);
  const [runOpen, setRunOpen] = useState(false);
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<Mode>("orbit");
  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [capabilities, setCapabilities] = useState<Record<string, boolean>>({});
  const [selectedFrame, setSelectedFrame] = useState(-1);
  const [framesOpen, setFramesOpen] = useState(true);
  const [kind, setKind] = useState<MeasurementKind | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [place, setPlace] = useState<{ item: string; yaw: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hover, setHover] = useState<Point | null>(null);
  const [game, setGame] = useState(false);
  const [bots, setBots] = useState(3);
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [draft, setDraftState] = useState<Placement[]>([]);
  const [saveState, setSaveState] = useState<SaveState>("clean");
  const busyRef = useRef(false);
  const dataVersion = useRef(0);
  const draftRef = useRef<Placement[]>([]);
  const queue = useRef<PlacementOp[]>([]);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flushing = useRef(false);
  const addSeq = useRef(0);
  const [message, setMessage] = useState("");
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const iframe = useRef<HTMLIFrameElement>(null);
  const revision = useRef<string | null>(null);
  // Planning editor: proposals are written through the backend, which returns the
  // evaluated state; undo/redo replays whole feature lists through the same path.
  const [planIndex, setPlanIndex] = useState<ProposalIndex | null>(null);
  const [catalogue, setCatalogue] = useState<Catalogue>({});
  const [planState, setPlanState] = useState<PlanState | null>(null);
  const [planTool, setPlanTool] = useState<PlanTool | null>(null);
  const [planPoints, setPlanPoints] = useState<Point[]>([]);
  const [planSelected, setPlanSelected] = useState<string | null>(null);
  const [planView, setPlanView] = useState<PlanView>("proposal");
  const [flickerPhase, setFlickerPhase] = useState(true);
  const [planBusy, setPlanBusy] = useState(false);
  const [planMessage, setPlanMessage] = useState("");
  const [objectItem, setObjectItem] = useState("street_light");
  const [arraySpacing, setArraySpacing] = useState(30);
  const [arrayOffset, setArrayOffset] = useState(0);
  const planStateRef = useRef<PlanState | null>(null);
  const undoStack = useRef<PlanFeature[][]>([]);
  const redoStack = useRef<PlanFeature[][]>([]);
  const [, setHistoryTick] = useState(0);
  const [planNonce, setPlanNonce] = useState(0);
  // Mission tab (Phase 3): same proposal machinery, kind "mission".
  const [symbolRole, setSymbolRole] = useState<SymbolRole>("infantry");
  const [analysis, setAnalysis] = useState<MissionAnalysis | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [showViewsheds, setShowViewsheds] = useState(true);
  const [los, setLos] = useState<LosResult | null>(null);
  const [hlz, setHlz] = useState<HlzCandidate[] | null>(null);
  const [candidates, setCandidates] = useState<LabelCandidate[] | null>(null);
  const [conditions, setConditions] = useState<Conditions>({ light: "day", nvg: false, fog_m: null });
  const [runs, setRuns] = useState<RunListing[]>([]);
  const [openRun, setOpenRun] = useState<Run | null>(null);
  const [basemap, setBasemap] = useState<Basemap | null>(null);
  const [gameQuery, setGameQuery] = useState<{ mission: string; query: string } | null>(null);
  const [shadowOn, setShadowOn] = useState(false);
  const [shadowSettings, setShadowSettings] = useState<ShadowSettings>(DEFAULT_SHADOW);
  const [shadow, setShadow] = useState<ShadowResult | null>(null);
  const [shadowFor, setShadowFor] = useState("");
  const [shadowBusy, setShadowBusy] = useState(false);
  const [shadowError, setShadowError] = useState("");
  const [swipe, setSwipe] = useState(50);
  const [compareReady, setCompareReady] = useState(false);
  const compareFrame = useRef<HTMLIFrameElement>(null);
  const lastOrbit = useRef<Orbit | null>(null);
  // Operations tab (Phase 4): the panel owns its analyses; the host owns picks and overlays.
  const [opsTool, setOpsTool] = useState<OpsTool | null>(null);
  const [opsPoints, setOpsPoints] = useState<Point[]>([]);
  const [opsOverlay, setOpsOverlay] = useState<PlanMesh[]>([]);
  const [inspTool, setInspTool] = useState<InspectTool | null>(null);
  const [inspPoints, setInspPoints] = useState<Point[]>([]);
  const [inspOverlay, setInspOverlay] = useState<PlanMesh[]>([]);
  const [twinOverlay, setTwinOverlay] = useState<PlanMesh[]>([]);
  const [twinView, setTwinView] = useState<TwinView>("visual");
  const [allTools, setAllTools] = useState(false);
  const [sandTable, setSandTable] = useState(false);
  const [instructor, setInstructor] = useState(false);
  const [threat, setThreat] = useState<ThreatMap | null>(null);
  const [threatBusy, setThreatBusy] = useState(false);
  const [showThreat, setShowThreat] = useState(true);
  const [obstacles, setObstacles] = useState<ObstacleList | null>(null);
  const [traffic, setTraffic] = useState<Trafficability | null>(null);
  const [showTraffic, setShowTraffic] = useState(true);
  const [facades, setFacades] = useState<FacadeResult | null>(null);
  const [facadesBusy, setFacadesBusy] = useState(false);
  const [showFacades, setShowFacades] = useState(true);
  const cursor = useRef("0");

  const command = useCallback((name: string, values: Record<string, unknown> = {}) => {
    iframe.current?.contentWindow?.postMessage({ namespace: "groundcontrol", type: "command", command: name, ...values }, window.location.origin);
  }, []);
  const compareCommand = useCallback((name: string, values: Record<string, unknown> = {}) => {
    compareFrame.current?.contentWindow?.postMessage({ namespace: "groundcontrol", type: "command", command: name, ...values }, window.location.origin);
  }, []);

  const setDraft = useCallback((next: Placement[]) => { draftRef.current = next; setDraftState(next); }, []);

  /** Server rows plus whatever local intent has not been written yet, in order. */
  const mergeQueue = useCallback((server: Placement[]) => {
    let list = [...server];
    for (const op of queue.current) if (op.kind === "delete") list = list.filter((p) => p.id !== op.id);
    for (const op of queue.current) if (op.kind === "add" && !list.some((p) => p.id === op.id)) list = [...list, op.draft];
    for (const op of queue.current) if (op.kind === "update") list = list.map((p) => (p.id === op.id ? op.draft : p));
    return list;
  }, []);

  const adopt = useCallback((detail: ProjectDetail) => {
    revision.current = detail.model_revision;
    dataVersion.current++;
    setProject(detail);
    setDraft(mergeQueue(detail.placements));
  }, [mergeQueue, setDraft]);

  const scheduleRef = useRef<(delay?: number) => void>(() => {});
  const runQueue = useCallback(async () => {
    if (flushing.current) { scheduleRef.current(300); return; }
    const batch = queue.current; queue.current = [];
    if (!batch.length) { setSaveState("clean"); return; }
    flushing.current = true; busyRef.current = true; setBusy(true); setSaveState("saving");
    let index = 0;
    try {
      for (; index < batch.length; index++) {
        const op = batch[index];
        if (op.kind === "delete") { adopt(await workspace.removePlacement(scene, op.id)); continue; }
        const row = draftRef.current.find((p) => p.id === op.id) ?? op.draft;
        const body = { point: placementPoint(row), yaw_deg: row.yaw_deg, scale: row.scale, size: row.size, label: row.label, model_revision: revision.current ?? "" };
        if (op.kind === "add") {
          const detail = await workspace.place(scene, { item: row.item, ...body });
          const created = detail.placements.at(-1);
          // The provisional row only exists locally until the POST names it.
          if (created && created.id !== op.id) for (const pending of queue.current) if (pending.id === op.id && pending.kind !== "delete") { pending.id = created.id; pending.draft = { ...pending.draft, id: created.id }; }
          // `placements` create has no size field, so a resized duplicate needs a follow-up write.
          if (created && created.size.some((value, axis) => Math.abs(value - row.size[axis]) > 0.001)) {
            const sized = draftRef.current.find((p) => p.id === created.id) ?? { ...created, size: row.size };
            queue.current.push({ kind: "update", id: created.id, draft: sized });
          }
          adopt(detail);
        } else {
          adopt(await workspace.movePlacement(scene, { id: op.id, item: row.item, ...body }));
        }
      }
      setSaveState(queue.current.length ? "queued" : "clean"); setMessage("");
      // Anything typed while the batch was in flight still owes a write.
      if (queue.current.length) scheduleRef.current();
    } catch (cause) {
      // Keep only the writes that never reached the server, so a retry cannot place a
      // duplicate item, and say out loud that the layout is not saved. No auto-retry:
      // a rejected revision or fit must not hammer the backend every second.
      queue.current = [...batch.slice(index), ...queue.current];
      setSaveState("error");
      setMessage(cause instanceof Error ? cause.message : String(cause));
    } finally {
      flushing.current = false; busyRef.current = false; setBusy(false);
    }
  }, [adopt, scene]);

  const scheduleFlush = useCallback((delay = 600) => {
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = setTimeout(() => { flushTimer.current = null; void runQueue(); }, delay);
  }, [runQueue]);
  scheduleRef.current = scheduleFlush;
  const flushNow = useCallback(() => {
    if (flushTimer.current) { clearTimeout(flushTimer.current); flushTimer.current = null; }
    void runQueue();
  }, [runQueue]);

  const queueOp = useCallback((op: PlacementOp) => {
    const existing = queue.current.find((pending) => pending.id === op.id);
    if (!existing) queue.current.push(op);
    else if (existing.kind === "add" && op.kind === "update") existing.draft = op.draft;
    else queue.current[queue.current.indexOf(existing)] = op;
    setSaveState("queued");
    scheduleFlush();
  }, [scheduleFlush]);

  /** Apply a pure edit to one piece: instant on screen, persisted on the debounce. */
  const mutate = useCallback((source: Placement, next: Placement) => {
    setDraft(draftRef.current.map((p) => (p.id === source.id ? next : p)));
    queueOp({ kind: "update", id: next.id, draft: next });
  }, [queueOp, setDraft]);

  // Unmount or leaving the screen must not strand an unwritten layout.
  const latestQueue = useRef(runQueue);
  useEffect(() => { latestQueue.current = runQueue; });
  useEffect(() => () => {
    if (flushTimer.current) clearTimeout(flushTimer.current);
    if (queue.current.length) void latestQueue.current();
  }, []);

  const refresh = useCallback(async () => {
    const version = dataVersion.current;
    try {
      const next = await workspace.project(scene);
      if (version !== dataVersion.current || busyRef.current) return;
      if (revision.current !== next.model_revision) {
        revision.current = next.model_revision;
        setReady(false); setPoints([]); setKind(null); setPlace(null); setHover(null); setSelected(null); setMessage("");
        if (queue.current.length) { queue.current = []; setSaveState("clean"); setMessage("The model was rebuilt, so unsaved layout edits were discarded."); }
        setProject(next); setDraft(next.placements);
      } else if (queue.current.length) {
        // A dirty draft outranks the poll: never let it yank the editor mid-drag.
        setProject({ ...next, placements: draftRef.current });
      } else { setProject(next); setDraft(next.placements); }
      setError("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, [scene, setDraft]);
  useEffect(() => { let active = true; let timer: ReturnType<typeof setTimeout>; const poll = async () => { if (!active) return; await refresh(); if (active) timer = setTimeout(poll, 3500); }; void poll(); workspace.list().then(setOptions).catch(() => {}); return () => { active = false; clearTimeout(timer); }; }, [refresh]);

  useEffect(() => {
    if (ready && !game && project) command("measurements", { value: tab === "place" || tab === "walk" || PANEL_TABS.includes(tab) ? [] : project.measurements });
  }, [ready, game, project, tab, command]);
  useEffect(() => { if (ready && !game) command("placements", { value: draft }); }, [ready, game, draft, command]);
  useEffect(() => {
    if (ready && !game) command("measurement-tool", { kind, unit: project?.scale.unit === "m" ? "m" : "units" });
  }, [kind, ready, game, project?.scale.unit, command]);
  useEffect(() => { if (ready && !game) command("select", { id: selected }); }, [selected, ready, game, command]);
  // Placement picking/dragging lives only on the Place screen, and never at the same
  // time as an armed drop or a measurement pick: exactly one of them owns the pointer.
  useEffect(() => {
    if (!ready || game) return;
    command("edit", { value: tab === "place" && !place });
    if (tab === "place") command("pick", { value: !!place, limit: 1 });
  }, [ready, game, tab, place, command]);

  const adoptPlan = useCallback((next: PlanState) => {
    planStateRef.current = next;
    setPlanState(next);
    setPlanIndex((index) => index ? { ...index, active: next.proposal.id } : index);
    setPlanSelected((id) => (id && !next.proposal.features.some((f) => f.id === id) ? null : id));
  }, []);
  const planTab = tab === "plan" || tab === "mission";
  const planKind = tab === "mission" ? "mission" : "plan";
  const loadPlans = useCallback(async (preferred?: string | null) => {
    try {
      const listing = await plan.list(scene, planKind);
      setPlanIndex(listing.index); setCatalogue(listing.catalogue);
      const id = preferred ?? listing.index.active ?? listing.index.proposals[0]?.id ?? null;
      if (id) adoptPlan(await plan.load(scene, id)); else { planStateRef.current = null; setPlanState(null); }
    } catch (cause) { setPlanMessage(cause instanceof Error ? cause.message : String(cause)); }
  }, [adoptPlan, scene, planKind]);
  useEffect(() => {
    if (!planTab) return;
    let live = true;
    // Plans and missions share one store: switching tabs switches the list and forgets history.
    const run = async () => {
      await Promise.resolve();
      if (!live) return;
      undoStack.current = []; redoStack.current = []; setPlanSelected(null); setAnalysis(null); setLos(null);
      await loadPlans();
    };
    void run();
    return () => { live = false; };
  }, [planTab, loadPlans]);

  /** One backend write; `record` pushes the pre-edit features for undo. */
  const planWrite = useCallback(async (write: () => Promise<PlanState>, { record = true, done = "" }: { record?: boolean; done?: string } = {}) => {
    const before = planStateRef.current?.proposal.features ?? null;
    setPlanBusy(true);
    try {
      const next = await write();
      if (record && before) { undoStack.current = [...undoStack.current.slice(-49), before]; redoStack.current = []; setHistoryTick((n) => n + 1); }
      adoptPlan(next); setPlanMessage(done);
      return next;
    } catch (cause) {
      const text = cause instanceof Error ? cause.message : String(cause);
      setPlanMessage(text);
      // Another window moved the scheme on: take its state rather than fight it.
      if (/changed in another window/.test(text) && planStateRef.current) void loadPlans(planStateRef.current.proposal.id);
      // Re-send the unchanged scheme so a refused 3D drag snaps back on screen.
      setPlanNonce((n) => n + 1);
      return null;
    } finally { setPlanBusy(false); }
  }, [adoptPlan, loadPlans]);

  const comparing = tab === "plan" && (planView === "swipe" || planView === "side");
  const effectiveView = planView === "flicker" ? (flickerPhase ? "proposal" : "existing") : planView === "existing" ? "existing" : "proposal";
  useEffect(() => {
    if (planView !== "flicker") return;
    const timer = setInterval(() => setFlickerPhase((phase) => !phase), 1400);
    return () => clearInterval(timer);
  }, [planView]);
  const shadowKey = planState ? `${planState.proposal.id}@${planState.proposal.revision}` : "";
  const shadowOverlay = useMemo(() => (shadowOn && shadow ? shadow.overlay : []), [shadowOn, shadow]);
  useEffect(() => {
    if (!ready || game) return;
    const edit = planTool || planView === "flicker" ? null : editShape(planState, planSelected);
    const overlay = tab === "mission" ? [...(showViewsheds && analysis ? analysis.overlay : []), ...(showThreat && threat ? threat.overlay : []), ...(showTraffic && traffic ? traffic.overlay : [])]
      : [...shadowOverlay, ...(showFacades && facades ? facades.overlay : [])];
    command("plan", { value: planTab ? viewerPlan(planState, planSelected, tab === "mission" ? "proposal" : effectiveView, { overlay, edit }) : tab === "ops" ? opsViewerPlan(opsOverlay) : tab === "inspect" ? opsViewerPlan(inspOverlay) : tab === "twin" ? opsViewerPlan(twinOverlay) : EMPTY_PLAN });
    // planNonce: a refused edit re-sends the same scheme so the viewer drops its preview.
  }, [ready, game, tab, planTab, planState, planSelected, effectiveView, command, shadowOverlay, planTool, planView, planNonce, showViewsheds, analysis, opsOverlay, showFacades, facades, showThreat, threat, inspOverlay, twinOverlay, showTraffic, traffic]);
  const chooseInspTool = useCallback((next: InspectTool | null) => {
    setInspTool(next); setInspPoints([]);
    if (!ready) return;
    command("clear-picks");
    command("pick", { value: !!next, limit: next ? INSPECT_TOOLS[next].limit : 1 });
  }, [command, ready]);
  const chooseTwinView = useCallback((view: TwinView) => {
    setTwinView(view);
    const want: Partial<Record<Layer, boolean>> = view === "visual" ? { splats: true, collider: false, coverage: false }
      : view === "measured" ? { splats: false, collider: true, coverage: false } : { splats: true, collider: false, coverage: true };
    for (const [layer, value] of Object.entries(want)) command("layer", { layer, value });
    setLayers((previous) => ({ ...previous, ...want }));
  }, [command]);
  const chooseOpsTool = useCallback((next: OpsTool | null) => {
    setOpsTool(next); setOpsPoints([]);
    if (!ready) return;
    command("clear-picks");
    command("pick", { value: !!next, limit: next ? TOOL_HINT[next].limit : 1 });
  }, [command, ready]);

  // Side-by-side and swipe: a second viewer shows the scan as it is, its camera locked
  // to the main (proposed) one through the host.
  useEffect(() => { if (ready && !game) command("camera-follow", { value: comparing }); }, [ready, game, comparing, command]);
  useEffect(() => {
    if (!comparing || !compareReady) return;
    compareCommand("plan", { value: viewerPlan(planState, null, "existing") });
  }, [comparing, compareReady, planState, compareCommand]);
  useEffect(() => {
    if (!comparing || !compareReady) return;
    // The main camera first: turning on the second viewer's broadcast before it has the
    // main orbit would push its own default view onto the main viewer.
    if (lastOrbit.current) compareCommand("camera-set", { value: lastOrbit.current });
    compareCommand("camera-follow", { value: true });
  }, [comparing, compareReady, compareCommand]);
  useEffect(() => {
    if (!comparing) return;
    const receive = (event: MessageEvent) => {
      if (event.source !== compareFrame.current?.contentWindow || event.origin !== window.location.origin || event.data?.namespace !== "groundcontrol") return;
      if (event.data.type === "ready") setCompareReady(true);
      else if (event.data.type === "camera" && isOrbit(event.data.orbit)) { lastOrbit.current = event.data.orbit; command("camera-set", { value: event.data.orbit }); }
      else if (event.data.type === "camera" && isPose(event.data.pose)) command("camera-set", { value: { pose: event.data.pose } });
    };
    window.addEventListener("message", receive);
    return () => window.removeEventListener("message", receive);
  }, [comparing, command]);

  // Shadow study: an instant view follows every edit (debounced); a day study runs on demand.
  const runShadow = useCallback(async (settings: ShadowSettings) => {
    const current = planStateRef.current;
    if (!current) return;
    setShadowBusy(true); setShadowError("");
    try {
      setShadow(await plan.shadow(scene, current.proposal.id, settings));
      setShadowFor(`${current.proposal.id}@${current.proposal.revision}`);
    } catch (cause) { setShadow(null); setShadowError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setShadowBusy(false); }
  }, [scene]);
  useEffect(() => {
    if (tab !== "plan" || !shadowOn || !shadowKey || shadowSettings.mode !== "instant") return;
    const timer = setTimeout(() => void runShadow(shadowSettings), 350);
    return () => clearTimeout(timer);
  }, [tab, shadowOn, shadowKey, shadowSettings, runShadow]);
  useEffect(() => { if (ready && !game) command("plan-select", { value: planTab && !planTool }); }, [ready, game, planTab, planTool, command]);

  const choosePlanTool = useCallback((next: PlanTool | null) => {
    setPlanTool(next); setPlanPoints([]); setPlanMessage("");
    if (!ready) return;
    command("clear-picks");
    command("pick", { value: !!next, limit: next === "object" || next === "symbol" ? 1 : next === "los" ? 2 : 128 });
    if (next === "los") setLos(null);
    if (next) setPlanSelected(null);
  }, [command, ready]);

  const finishPlan = useCallback(async (points: Point[] = planPoints) => {
    const current = planStateRef.current;
    if (!current || !planTool) return;
    if (planTool === "array") {
      const line = points.map((p) => [p[0], p[2]] as XZ);
      if (line.length < 2) return;
      await planWrite(() => plan.array(scene, current.proposal, objectItem, line, arraySpacing, arrayOffset), { done: "Array placed." });
    } else if (planTool === "los") {
      if (points.length < 2) return;
      try { setLos(await missionApi.los(scene, points[0], points[1])); } catch (cause) { setPlanMessage(cause instanceof Error ? cause.message : String(cause)); }
      command("clear-picks"); command("pick", { value: true, limit: 2 }); setPlanPoints([]);
      return;
    } else if (planTool === "symbol" || planTool === "route" || planTool === "phase_line") {
      const feature = draftMissionFeature(planTool, points, symbolRole);
      if (!feature) { setPlanMessage("Not enough points yet."); return; }
      const next = await planWrite(() => plan.upsert(scene, current.proposal, feature));
      if (next) setPlanSelected(next.proposal.features.at(-1)?.id ?? null);
    } else {
      const feature = draftFeature(planTool, points, { item: objectItem });
      if (!feature) { setPlanMessage("Not enough distinct points yet — a rectangle needs two corners at least 1 m apart."); return; }
      const next = await planWrite(() => plan.upsert(scene, current.proposal, feature));
      if (next) setPlanSelected(next.proposal.features.at(-1)?.id ?? null);
    }
    if (planTool === "object" || planTool === "symbol") { setPlanPoints([]); command("clear-picks"); command("pick", { value: true, limit: 1 }); return; }
    choosePlanTool(null);
  }, [arrayOffset, arraySpacing, choosePlanTool, command, objectItem, planPoints, planTool, planWrite, scene, symbolRole]);

  const undoPlan = useCallback(async (redo = false) => {
    const current = planStateRef.current;
    const stack = redo ? redoStack : undoStack;
    const target = stack.current.at(-1);
    if (!current || !target) return;
    const next = await planWrite(() => plan.replace(scene, current.proposal, target), { record: false, done: redo ? "Redone." : "Undone." });
    if (next) {
      stack.current = stack.current.slice(0, -1);
      (redo ? undoStack : redoStack).current.push(current.proposal.features);
      setHistoryTick((n) => n + 1);
    }
  }, [planWrite, scene]);

  const updateFeature = useCallback((feature: PlanFeature) => {
    const current = planStateRef.current;
    if (current) void planWrite(() => plan.upsert(scene, current.proposal, feature));
  }, [planWrite, scene]);
  const deleteFeature = useCallback((id: string) => {
    const current = planStateRef.current;
    if (current) void planWrite(() => plan.removeFeature(scene, current.proposal, id), { done: "Deleted. Undo brings it back." });
  }, [planWrite, scene]);
  const demolish = useCallback((building: ExistingBuilding) => {
    const current = planStateRef.current;
    if (!current) return;
    const [cx, cz] = building.centre;
    // Half a metre of margin so the whole scanned facade falls inside the hidden volume.
    const polygon = building.footprint.map(([x, z]) => {
      const dx = x - cx, dz = z - cz, d = Math.hypot(dx, dz) || 1;
      return [x + (dx / d) * 0.5, z + (dz / d) * 0.5] as XZ;
    });
    void planWrite(() => plan.upsert(scene, current.proposal, { type: "clip", name: "Demolish existing building", params: { polygon, reason: "demolish" } }), { done: "Marked for demolition." });
  }, [planWrite, scene]);
  const exportPlan = useCallback(async (format: ExportFormat) => {
    const current = planStateRef.current;
    if (!current) return;
    try {
      const file = await plan.exportAs(scene, current.proposal.id, format);
      const body = file.encoding === "base64" ? Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0)) : file.content;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([body], { type: file.media_type }));
      link.download = `${scene}-${file.filename}`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 4000);
      setPlanMessage(`Exported ${file.filename}.`);
    } catch (cause) { setPlanMessage(cause instanceof Error ? cause.message : String(cause)); }
  }, [scene]);
  const importParcels = useCallback(async (file: File, epsg?: number) => {
    const current = planStateRef.current;
    if (!current) return;
    const zipped = file.name.toLowerCase().endsWith(".zip");
    if (file.size > (zipped ? 1_400_000 : 2_000_000)) { setPlanMessage("Cadastral files are limited to about 2 MB here (1.4 MB zipped): clip a district-wide file to the site first."); return; }
    let text: string;
    if (zipped) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      text = btoa(binary);
    } else text = await file.text();
    const next = await planWrite(() => plan.importParcels(scene, current.proposal, file.name, text, epsg));
    const report = (next as (PlanState & { import?: ImportReport }) | null)?.import;
    if (report) setPlanMessage(`Imported ${report.imported} parcel${report.imported === 1 ? "" : "s"} as plots from ${report.basis}`
      + (report.skipped_count ? `; skipped ${report.skipped_count} (${[...new Set(report.skipped.map((s) => s.reason))].join("; ")})` : "") + ". Undo removes them.");
  }, [planWrite, scene]);

  // ---- Mission: analysis (re-run after each edit once shown), runs, review, rehearsal.
  const missionKey = tab === "mission" && planState ? `${planState.proposal.id}@${planState.proposal.revision}` : "";
  const analyse = useCallback(async () => {
    const current = planStateRef.current;
    if (!current || current.proposal.kind !== "mission") return;
    setAnalysisBusy(true); setAnalysisError("");
    try { setAnalysis(await missionApi.analysis(scene, current.proposal.id)); }
    catch (cause) { setAnalysisError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setAnalysisBusy(false); }
  }, [scene]);
  const analysed = analysis !== null;
  useEffect(() => {
    if (!missionKey || !analysed) return;
    const timer = setTimeout(() => void analyse(), 500);
    return () => clearTimeout(timer);
  }, [missionKey, analysed, analyse]);
  const missionId = tab === "mission" ? planState?.proposal.id ?? null : null;
  const reviewAfterGame = useRef<string | null>(null);
  const openReview = useCallback(async (id: string) => {
    const current = planStateRef.current;
    if (!current) return;
    try {
      const [run, base] = await Promise.all([missionApi.run(scene, current.proposal.id, id), basemap ? Promise.resolve(basemap) : missionApi.basemap(scene)]);
      setBasemap(base); setOpenRun(run);
    } catch (cause) { setPlanMessage(cause instanceof Error ? cause.message : String(cause)); }
  }, [basemap, scene]);
  useEffect(() => {
    if (!missionId) return;
    let live = true;
    missionApi.runs(scene, missionId).then((r) => {
      if (!live) return;
      setRuns(r.runs);
      const pending = reviewAfterGame.current;
      if (pending && r.runs.some((x) => x.id === pending)) { reviewAfterGame.current = null; void openReview(pending); }
    }).catch(() => {});
    return () => { live = false; };
  }, [missionId, scene, openReview]);
  const coveredRoute = useCallback(async (routeId: string) => {
    const current = planStateRef.current;
    if (!current) return;
    const next = await planWrite(() => missionApi.coveredRoute(scene, current.proposal, routeId));
    const s = (next as (PlanState & { suggested?: { id: string; planned: { exposure_s: number; length_m: number }; covered: { exposure_s: number; length_m: number } } }) | null)?.suggested;
    if (s) {
      setPlanSelected(s.id);
      setPlanMessage(`Suggested route: exposed ${s.covered.exposure_s.toFixed(0)} s instead of ${s.planned.exposure_s.toFixed(0)} s, ${s.covered.length_m.toFixed(0)} m instead of ${s.planned.length_m.toFixed(0)} m. It is a new route; the planned one is kept.`);
    }
  }, [planWrite, scene]);
  const downloadFile = useCallback((file: { filename: string; media_type: string; encoding: string; content: string }) => {
    const body = file.encoding === "base64" ? Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0)) : file.content;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([body], { type: file.media_type }));
    link.download = `${scene}-${file.filename}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  }, [scene]);
  const switchTabRef = useRef<(value: string) => void>(() => {});

  useEffect(() => {
    if (!planTab) return;
    const keys = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      const mod = event.ctrlKey || event.metaKey;
      if (mod && event.key.toLowerCase() === "z") { event.preventDefault(); void undoPlan(event.shiftKey); }
      else if (mod && event.key.toLowerCase() === "y") { event.preventDefault(); void undoPlan(true); }
      else if (event.key === "Escape") { if (planTool) choosePlanTool(null); else setPlanSelected(null); }
      else if (event.key === "Enter" && planTool && planTool !== "object" && planTool !== "symbol") void finishPlan();
      else if ((event.key === "Delete" || event.key === "Backspace") && planSelected && !planTool) deleteFeature(planSelected);
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  }, [planTab, planTool, planSelected, undoPlan, choosePlanTool, finishPlan, deleteFeature]);

  /** Drop a locally-built row into the layout and ask the backend to name it. */
  // Stable on purpose: addAt is a dependency of the viewer message listener, so an
  // addRow that changed identity every render would resubscribe it constantly.
  const addRow = useCallback((row: Placement) => {
    setDraft([...draftRef.current, row]);
    queueOp({ kind: "add", id: row.id, draft: row });
    setSelected(row.id);
  }, [queueOp, setDraft]);

  const addAt = useCallback((point: Point) => {
    const source = project?.furniture.find((f) => f.item === place?.item);
    if (!source || !place) return;
    addRow(provisionalPlacement(source, point, place.yaw, revision.current ?? "", `local-${++addSeq.current}`));
    setMessage(`${source.label} added. Click again for another, or stop placing to edit.`);
  }, [addRow, place, project, setMessage]);

  const dragTo = useCallback((id: string, point: Point, final: boolean) => {
    const current = draftRef.current.find((p) => p.id === id);
    if (!current) return;
    mutate(current, moveToPlacement(current, point));
    setSelected(id);
    if (final) flushNow();
  }, [flushNow, mutate]);

  useEffect(() => {
    const receive = (event: MessageEvent) => {
      if (event.source !== iframe.current?.contentWindow || event.origin !== window.location.origin || !event.data || event.data.namespace !== "groundcontrol") return;
      const data = event.data;
      if (data.type === "ready" || data.type === "state") {
        if (["orbit", "fly", "walk"].includes(data.mode)) setMode(data.mode);
        if (data.layers && typeof data.layers === "object") setLayers((previous) => Object.fromEntries(Object.keys(previous).map((key) => [key, typeof data.layers[key] === "boolean" ? data.layers[key] : previous[key as Layer]])) as Record<Layer, boolean>);
        if (data.type === "ready") { setReady(true); setMessage(""); if (data.capabilities && typeof data.capabilities === "object") setCapabilities(data.capabilities); }
      } else if (data.type === "pick" && isPoint(data.point) && (tab === "plan" || tab === "mission") && planTool) {
        const point = data.point as Point;
        setMessage("");
        if (planTool === "object" || planTool === "symbol") void finishPlan([point]);
        else if (planTool === "los") setPlanPoints((previous) => { const next = [...previous, point].slice(0, 2); if (next.length === 2) void finishPlan(next); return next; });
        else setPlanPoints((previous) => [...previous, point].slice(0, 128));
      } else if (data.type === "pick" && isPoint(data.point) && tab === "inspect" && inspTool) {
        const point = data.point as Point;
        setMessage("");
        setInspPoints((previous) => [...previous, point].slice(0, INSPECT_TOOLS[inspTool].limit));
      } else if (data.type === "pick" && isPoint(data.point) && tab === "ops" && opsTool) {
        const point = data.point as Point;
        setMessage("");
        setOpsPoints((previous) => [...previous, point].slice(0, TOOL_HINT[opsTool].limit));
      } else if (data.type === "plan-pick") {
        setPlanSelected(typeof data.id === "string" ? data.id : null);
      } else if (data.type === "plan-edit" && typeof data.id === "string" && Array.isArray(data.points)) {
        const feature = planStateRef.current?.proposal.features.find((f) => f.id === data.id);
        const points = (data.points as unknown[]).filter((p): p is XZ => Array.isArray(p) && p.length === 2 && p.every((v) => typeof v === "number" && Number.isFinite(v)));
        if (feature && points.length === data.points.length) {
          updateFeature(applyEdit(feature, points, typeof data.rotation_deg === "number" && Number.isFinite(data.rotation_deg) ? data.rotation_deg : 0));
          setPlanMessage(data.handle === "remove" ? "Corner removed." : "");
        }
      } else if (data.type === "rehearsal-ended") {
        // The arena saved the run: come back to the mission and open its review.
        const saved = data.saved && typeof data.saved.id === "string" ? data.saved.id : null;
        setMessage(saved ? "Rehearsal saved. Opening the after-action review…" : `Rehearsal ended; the run was not saved (${String(data.error ?? "unknown error")}).`);
        if (saved) setTimeout(() => { reviewAfterGame.current = saved; setGame(false); setReady(false); setGameQuery(null); switchTabRef.current("mission"); }, 1800);
      } else if (data.type === "camera" && isOrbit(data.orbit)) {
        lastOrbit.current = data.orbit;
        compareCommand("camera-set", { value: data.orbit });
      } else if (data.type === "camera" && isPose(data.pose)) {
        compareCommand("camera-set", { value: { pose: data.pose } });
      } else if (data.type === "pick" && isPoint(data.point)) {
        if (place) addAt(data.point);
        else if (kind) {
          setPoints((previous) => [...previous, data.point as Point].slice(0, kind === "point" ? 1 : kind === "height" ? 2 : 128));
          setHover(null); setMessage("Point added. The live reading is a surface estimate.");
        }
      } else if (data.type === "placement-pick" && typeof data.id === "string") {
        setSelected(data.id); setMessage("Selected. Drag it, or use the dock.");
      } else if (data.type === "placement-move" && typeof data.id === "string" && isPoint(data.point)) {
        dragTo(data.id, data.point, data.final === true);
      } else if (data.type === "preview") {
        setHover(isPoint(data.point) ? data.point : null);
      } else if ((data.type === "error" || data.type === "pick-miss") && typeof data.message === "string") setMessage(data.message.slice(0, 500));
      else if (data.type === "snapshot" && typeof data.dataUrl === "string" && data.dataUrl.startsWith("data:image/png;base64,") && data.dataUrl.length < 40000000) {
        const link = document.createElement("a"); link.href = data.dataUrl; link.download = `${scene}-view.png`; link.click(); setMessage("Viewport snapshot downloaded.");
      }
    };
    window.addEventListener("message", receive); return () => window.removeEventListener("message", receive);
  }, [addAt, dragTo, kind, place, scene, tab, planTool, finishPlan, updateFeature, compareCommand, opsTool, inspTool]);

  useEffect(() => {
    if (!logsOpen) return;
    let active = true; let timer: ReturnType<typeof setTimeout>;
    const poll = async () => { try { const result = await pipeline.tail(scene, cursor.current); if (active) { cursor.current = result.cursor; setLogs((previous) => [...previous, ...result.lines].slice(-400)); } } catch { if (active) setLogs(["Log service unavailable. Retry when the backend reconnects."]); } if (active) timer = setTimeout(poll, 2500); };
    void poll(); return () => { active = false; clearTimeout(timer); };
  }, [logsOpen, scene]);
  function clear() { setKind(null); setPoints([]); setHover(null); setPlace(null); setMessage(""); setPlanTool(null); setPlanPoints([]); setOpsTool(null); setOpsPoints([]); setOpsOverlay([]); setInspTool(null); setInspPoints([]); setInspOverlay([]); setTwinOverlay([]); if (ready && !game) { command("pick", { value: false }); command("clear-picks"); } }
  function chooseTool(value: MeasurementKind) {
    clear(); command("edit", { value: false }); setKind(value);
    command("pick", { value: true, limit: value === "point" ? 1 : value === "height" ? 2 : 128 });
    setMessage("Click to measure. Drag to orbit; right-drag to pan.");
  }
  function choosePlacement(item: string) {
    clear(); setSelected(null); setPlace({ item, yaw: 0 });
    if (ready) { command("edit", { value: false }); command("select", { id: null }); command("pick", { value: true, limit: 1 }); }
    setMessage("Click the scanned floor to drop it. Click empty space, then another spot, for more.");
  }
  function yawDelta(delta: number) { setPlace((previous) => previous ? { ...previous, yaw: ((previous.yaw + delta) % 360 + 360) % 360 } : previous); }
  /** Upload one glTF/GLB; the backend reads its real size, then it joins the catalogue. */
  async function importModel(file: File) {
    if (busyRef.current) { setMessage("Finish the current save before importing a model."); return; }
    setImporting(true); setMessage("");
    try {
      const detail = await workspace.importModel(scene, file, () => {});
      adopt(detail);
      const stem = file.name.replace(/\.(glb|gltf)$/i, "");
      const created = detail.furniture.find((f) => f.imported && f.label === stem) ?? detail.furniture.filter((f) => f.imported).at(-1);
      setMessage(created ? `Imported “${created.label}” — real size ${created.size.map((v) => v.toFixed(2)).join(" × ")} m, read from the file. Pick it, then click the floor to place.` : "Model imported.");
    } catch (cause) {
      // The exact server reason (unsupported file, unreadable bounds, traversal…) is surfaced verbatim.
      setMessage(cause instanceof Error ? cause.message : String(cause));
    } finally { setImporting(false); }
  }
  function undoPoint() { const next = points.slice(0, -1); setPoints(next); setHover(null); command("set-picks", { value: next }); command("pick", { value: true, limit: kind === "point" ? 1 : kind === "height" ? 2 : 128 }); }
  function removePlacement(id: string) {
    setDraft(draftRef.current.filter((p) => p.id !== id));
    setSelected((previous) => (previous === id ? null : previous));
    if (queue.current.find((op) => op.id === id)?.kind === "add") {
      // It never reached the server: dropping the local intent is the whole delete.
      queue.current = queue.current.filter((op) => op.id !== id);
      setSaveState(queue.current.length ? "queued" : "clean");
      return;
    }
    queueOp({ kind: "delete", id });
    flushNow();
  }
  function exportLayout() { downloadBlob(`${scene}-layout.geojson`, "application/geo+json", JSON.stringify(placementsGeoJSON(draft, scene), null, 2)); }
  function chooseFrame(index: number) { clear(); setSelectedFrame(index); const camera = project?.frames[index]?.camera_index; if (typeof camera === "number" && ready) command("frame", { index: camera }); else { setTab("layers"); setMessage("Showing the original image. A recovered camera pose and ready viewer are needed to position the 3D view."); } }
  function rehearse() {
    const current = planStateRef.current;
    if (!current) return;
    const query = rehearsalQuery(scene, current.proposal.id, conditions);
    switchTab("walk");
    setGameQuery({ mission: current.proposal.id, query });
    setBots(1); setReady(false); setGame(true);
  }
  switchTabRef.current = (value: string) => switchTab(value);
  function switchTab(value: string) {
    clear(); if (queue.current.length) flushNow();
    // The arena mounts its own iframe, so the reconstruction viewer is gone either way.
    if (value === "walk" || tab === "walk") setReady(false);
    if (game) { setGame(false); setReady(false); }
    setTab(value); setFramesOpen(value === "layers");
    const path = `/projects/${encodeURIComponent(scene)}`;
    window.history.pushState(null, "", MODES.includes(value) ? `${path}/${value}` : `${path}?tab=${value}`);
  }
  useEffect(() => {
    const back = () => {
      if (queue.current.length) flushNow();
      setKind(null); setPoints([]); setHover(null); setPlace(null); setGame(false); setReady(false);
      command("pick", { value: false }); command("edit", { value: false }); command("clear-picks");
      const route = window.location.pathname.split("/").at(-1);
      setTab(route && MODES.includes(route) ? route : new URLSearchParams(window.location.search).get("tab") || "layers");
    };
    window.addEventListener("popstate", back); return () => window.removeEventListener("popstate", back);
  }, [command, flushNow]);
  async function cancel() { try { await workspace.cancel(scene); setConfirmCancel(false); await refresh(); } catch (cause) { setMessage(cause instanceof Error ? cause.message : String(cause)); } }
  const running = project?.job.scene === scene && ["running", "starting"].includes(project.job.status);
  if (!project) return <Studio connected={!error}><main className="library-main">{error ? <Offline message={error} retry={() => void refresh()} /> : <div className="loading-state"><span className="spinner" />Opening project…</div>}</main></Studio>;
  return <Studio project={project.name} connected={!error}>
    <main className={`workspace-main workspace-${tab}`}>
      <header className="workspace-header"><div className="workspace-title"><span className="brand-symbol"><Icon name="cube" size={25} /></span><div><h1>{project.name}</h1><Status status={project.status} /></div></div><nav className="workspace-modes" aria-label="Choose workspace">{[["layers", "Explore", "compass"], ["measure", "Measure", "ruler"], ["plan", "Plan", "grid"], ["mission", "Mission", "pin"], ["ops", "Operations", "activity"], ["inspect", "Inspect", "search"], ["twin", "Twin", "layers"], ["place", "Place", "cube"], ["walk", "Walkthrough", "play"]]
            // The application's tabs lead; the rest wait behind "All tools" (and the open tab always shows).
            .filter(([value]) => allTools || !project.application || project.application.workspace_tabs.includes(value) || value === "layers" || value === tab)
            .map(([value, label, icon]) => <button key={value} aria-pressed={tab === value} className={tab === value ? "active" : ""} onClick={() => switchTab(value)}><Icon name={icon as IconName} size={16} /><span>{label}</span></button>)}
            {project.application && <button className="workspace-more" aria-pressed={allTools} title={allTools ? `Show only the ${project.application.label} tools` : "Show every tool"} onClick={() => setAllTools(!allTools)}><Icon name={allTools ? "close" : "plus"} size={14} /><span>{allTools ? "Fewer" : "All tools"}</span></button>}</nav><div className="workspace-actions"><button className="icon-button" title="Processing" aria-label="Processing" onClick={() => setLogsOpen(!logsOpen)}><Icon name="activity" size={16} /></button><button className="button secondary small" onClick={() => switchTab("exports")}><Icon name="download" size={15} />Exports</button><button className="button primary small" onClick={() => setRunOpen(true)}><Icon name="play" size={14} />Reconstruct</button></div></header>
      {error && <div className="job-banner"><Icon name="info" /><span>Connection lost. Showing the last project state; changes cannot be saved until reconnected.</span><button className="text-button" onClick={() => void refresh()}>Retry</button></div>}
      {running && <div className="job-banner"><span className="spinner" /><span>Processing this capture<small>{project.job.step || "Starting reconstruction"} · you can keep exploring existing outputs</small></span><button className="button danger small" onClick={() => confirmCancel ? void cancel() : setConfirmCancel(true)}>{confirmCancel ? "Confirm stop" : "Stop job"}</button>{confirmCancel && <button className="text-button" onClick={() => setConfirmCancel(false)}>Keep running</button>}</div>}
      <div className="workspace-grid">
        <div className="workspace-stage">
          {tab === "place" && <PlacementPanel
            project={project} placements={draft} selected={selected} place={place} saveState={saveState}
            offline={!!error} message={message} status={busy ? SAVE_COPY.saving : SAVE_COPY[saveState]} importing={importing}
            onPickItem={choosePlacement} onYawArm={yawDelta} onCancelPlace={clear} onRetrySave={flushNow}
            onImportModel={importModel}
            onSelect={setSelected} onSlide={(source, dx, dz) => mutate(source, slidePlacement(source, dx, dz))}
            onLift={(source, dy) => mutate(source, liftPlacement(source, dy))}
            onYaw={(source, delta) => mutate(source, yawPlacement(source, delta))}
            onResize={(source, axis, delta) => mutate(source, resizePlacement(source, axis, delta))}
            onDuplicate={(source) => addRow(duplicatePlacement(source, `local-${++addSeq.current}`))}
            onDelete={removePlacement} onExport={exportLayout} onFit={() => command("fit")}
          />}
          {(tab === "walk" && !game) ? <WalkthroughPanel project={project} bots={bots} onEnter={(count) => { clear(); setBots(count); setReady(false); setGame(true); }} /> : <ViewerPanel compare={comparing ? { mode: planView as "swipe" | "side", iframeRef: compareFrame, position: swipe, onPosition: setSwipe, onLoad: () => { setCompareReady(false); compareCommand("get-state"); } } : undefined} project={project} iframeRef={iframe} ready={ready} mode={mode} selectedFrame={selectedFrame} message={message} framesOpen={framesOpen} showFrames={tab !== "place" && tab !== "walk" && !PANEL_TABS.includes(tab)} showMessage={tab !== "place"} arena={tab === "walk"} game={game} bots={bots} onCommand={command} onFrame={chooseFrame} onFramesToggle={() => setFramesOpen(!framesOpen)} onRun={() => setRunOpen(true)} gameQuery={gameQuery?.query} onExitGame={() => {
            if (gameQuery) { iframe.current?.contentWindow?.postMessage({ namespace: "groundcontrol", type: "command", command: "end-rehearsal" }, window.location.origin); return; }
            setGame(false); setReady(false);
          }} />}
          {tab === "mission" && instructor && planState && <div className="aar-overlay"><InstructorView scene={scene} missionId={planState.proposal.id} basemap={basemap} onClose={() => setInstructor(false)} /></div>}
          {tab === "mission" && sandTable && planState && <div className="aar-overlay"><SandTable state={planState} analysis={analysis} basemap={basemap} onClose={() => setSandTable(false)} /></div>}
          {tab === "mission" && openRun && <div className="aar-overlay"><AarReplay run={openRun} basemap={basemap} onClose={() => setOpenRun(null)}
            onPdf={() => { const current = planStateRef.current; if (current) missionApi.aarPdf(scene, current.proposal.id, openRun.id).then(downloadFile).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))); }} /></div>}
          {tab === "measure" && <div className="surface-controls"><span>MEASURE ON</span><button aria-pressed={layers.collider} disabled={!ready || !capabilities.collider} onClick={() => command("layer", { layer: "collider", value: !layers.collider })}>Collision surface</button><button aria-pressed={layers.points} disabled={!ready || !capabilities.points} onClick={() => command("layer", { layer: "points", value: !layers.points })}>Cloud points</button></div>}
        </div>
        {!["place", "walk"].includes(tab) && <aside className="inspector" aria-label="Project tools">{tab !== "measure" && !PANEL_TABS.includes(tab) && <div className="inspector-tabs">{[["layers", "Layers"], ["quality", "Quality"], ["details", "Details"], ["exports", "Files"]].map(([value, label]) => <button key={value} className={tab === value ? "active" : ""} onClick={() => switchTab(value)}>{label}</button>)}</div>}
          {tab === "layers" && <><section className="inspector-section"><div className="section-label"><span>SCENE LAYERS</span><Icon name="layers" size={14} /></div>{LAYERS.map((layer) => <label className="layer-control" key={layer.key}><Icon name={layer.icon} size={17} /><span>{layer.label}<small>{layer.detail}</small></span><input type="checkbox" aria-label={layer.label} checked={layers[layer.key]} disabled={!ready || (layer.key !== "splats" && !capabilities[layer.key])} onChange={(event) => { const value = event.target.checked; setLayers((previous) => ({ ...previous, [layer.key]: value })); command("layer", { layer: layer.key, value }); }} /></label>)}</section>{project.semantics && <section className="inspector-section"><div className="section-label"><span>SCENE CLASSIFICATION</span><Icon name="layers" size={14} /></div><div className="class-list">{Object.entries(project.semantics.counts).filter(([, n]) => n > 0).map(([cls, n]) => { const s = project.semantics?.summary?.[cls as string]; return <div className="datum-row" key={cls}><span><i className={`class-dot class-${cls}`} />{cls}</span><span>{n.toLocaleString()}{s?.area_m2 ? ` · ${s.area_m2.toLocaleString()} m²` : ""}</span></div>; })}</div><p className="inspector-copy">Heuristic geometric + colour labels; area is a grid-occupancy footprint proxy (coarse, not surveyed ground truth).</p></section>}<section className="inspector-section"><div className="section-label">SPATIAL REFERENCE</div><div className="datum-row"><span>Scale</span><span>{project.scale.status === "metric" ? "Metric · pose referenced" : project.scale.status === "estimated" ? "Estimated" : "Relative"}</span></div>{project.scale_check && <div className="datum-row" title={`A learned depth model estimated ${project.scale_check.ruler_m_per_unit} m per scene unit against this scene's ${project.scale_check.existing_m_per_unit ?? "?"} m. It changes no measurement: on the one scene checked with a tape measure this ruler read 18% low where the pose-referenced scale read 4% low, so treat a disagreement as a question to investigate, not a correction to apply.`}><span>Second ruler (learned)</span><span>{project.scale_check.gap_percent === null ? "Not comparable" : `${project.scale_check.gap_percent}% apart`}</span></div>}
<div className="datum-row"><span>Location</span><span>{project.georeference.status === "local" ? "Local coordinates" : project.georeference.crs}</span></div><div className="datum-row"><span>Accuracy</span><span>{project.accuracy.status === "verified" ? `${project.accuracy.rmse_m} m RMSE` : "Not independently verified"}</span></div><GnssReference project={project} /><p className="inspector-copy">{project.scale.source}</p></section><section className="inspector-section"><div className="section-label">{WORKFLOWS[project.workflow]?.label ?? "Explore"} WORKFLOW</div><p className="inspector-copy">{project.workflow === "inspection" ? "Select a source frame to inspect detail. Use annotations to record observations on the model." : project.workflow === "survey" ? "Inspect scale before measuring. Use Details for survey inputs and Exports for generated products." : project.workflow === "response" ? "Enable observed coverage to see view support. Unknown areas are not proof of safe passage." : project.workflow === "heritage" ? "Use orbit or walk to explore. Save a snapshot of the current viewpoint from the viewport toolbar." : "Orbit the reconstruction, inspect original camera views, and switch layers to understand its geometry."}</p></section>{selectedFrame >= 0 && project.frames[selectedFrame] && <div className="frame-inspection"><Image unoptimized width={640} height={400} src={project.frames[selectedFrame].url} alt={`Inspected source frame ${selectedFrame + 1}`} /><p>Frame {selectedFrame + 1} · {project.frames[selectedFrame].name}</p></div>}</>}
          {tab === "measure" && <MeasurementPanel project={project} ready={ready && !!capabilities.collider && !error} points={points} hover={hover} kind={kind} onTool={chooseTool} onClear={clear} onUndo={undoPoint} onSaved={(next) => { dataVersion.current++; setProject(next); }} onSelect={(id) => command("select", { id })} />}
          {tab === "plan" && <PlanPanel
            scene={scene} state={planState} index={planIndex} catalogue={catalogue} selected={planSelected}
            tool={planTool} points={planPoints.length} view={planView} busy={planBusy} message={planMessage}
            ready={ready && !!capabilities.collider && !error} canUndo={undoStack.current.length > 0} canRedo={redoStack.current.length > 0}
            objectItem={objectItem} arraySpacing={arraySpacing} arrayOffset={arrayOffset}
            onProposal={(id) => { setPlanSelected(null); undoStack.current = []; redoStack.current = []; void loadPlans(id); }}
            onCreate={(name, source) => { undoStack.current = []; redoStack.current = []; void planWrite(() => plan.create(scene, name, source), { record: false, done: source ? "Scheme duplicated." : "Scheme created. Pick a tool and draw on the scan." }).then(() => loadPlans(planStateRef.current?.proposal.id)); }}
            onRename={(name) => { const id = planStateRef.current?.proposal.id; if (id) void planWrite(() => plan.rename(scene, id, name), { record: false }).then(() => loadPlans(id)); }}
            onDeleteProposal={() => { const id = planStateRef.current?.proposal.id; if (id) void plan.remove(scene, id).then(() => { undoStack.current = []; redoStack.current = []; setPlanSelected(null); return loadPlans(null); }).catch((cause) => setPlanMessage(String(cause))); }}
            onTool={choosePlanTool} onFinish={() => void finishPlan()}
            onUndoPoint={() => { const next = planPoints.slice(0, -1); setPlanPoints(next); command("set-picks", { value: next }); command("pick", { value: true, limit: 128 }); }}
            onObjectItem={setObjectItem} onArray={(spacing, offset) => { setArraySpacing(spacing); setArrayOffset(offset); }}
            onUpdate={updateFeature} onDeleteFeature={deleteFeature} onSelect={setPlanSelected} onView={setPlanView}
            onUndo={() => void undoPlan(false)} onRedo={() => void undoPlan(true)} onExport={(format) => void exportPlan(format)} onDemolish={demolish}
            onImport={(file, epsg) => void importParcels(file, epsg)}
            onInferred={(inferred, basis) => { const id = planStateRef.current?.proposal.id; if (id) void planWrite(() => plan.setInferred(scene, id, inferred, basis), { record: false, done: inferred ? "Marked as a hypothesis: every view and export now says inferred." : "No longer marked as a hypothesis." }); }}
            shadow={{ on: shadowOn, settings: shadowSettings, result: shadow, busy: shadowBusy, error: shadowError, stale: !!shadow && shadowFor !== shadowKey }}
            onShadowToggle={(on) => { setShadowOn(on); if (!on) { setShadow(null); setShadowError(""); } }}
            onShadowSettings={(next) => { setShadowSettings(next); if (next.mode !== shadowSettings.mode) setShadow(null); }}
            onShadowRun={() => void runShadow(shadowSettings)}
            facades={facades} facadesBusy={facadesBusy} showFacades={showFacades} onShowFacades={setShowFacades}
            onFacades={() => { setFacadesBusy(true); plan.facades(scene).then((r) => { setFacades(r); setShowFacades(true); }).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))).finally(() => setFacadesBusy(false)); }}
          />}
          {tab === "mission" && <MissionPanel
            scene={scene} state={planState} index={planIndex} selected={planSelected} tool={planTool} points={planPoints.length}
            busy={planBusy} message={planMessage} ready={ready && !!capabilities.collider && !error}
            canUndo={undoStack.current.length > 0} canRedo={redoStack.current.length > 0}
            role={symbolRole} onRole={setSymbolRole}
            onProposal={(id) => { setPlanSelected(null); setAnalysis(null); undoStack.current = []; redoStack.current = []; void loadPlans(id); }}
            onCreate={(name, source) => { undoStack.current = []; redoStack.current = []; setAnalysis(null); void planWrite(() => plan.create(scene, name, source, "mission"), { record: false, done: source ? "Mission duplicated." : "Mission created. Place enemy posts, then draw the route." }).then(() => loadPlans(planStateRef.current?.proposal.id)); }}
            onRename={(name) => { const id = planStateRef.current?.proposal.id; if (id) void planWrite(() => plan.rename(scene, id, name), { record: false }).then(() => loadPlans(id)); }}
            onDeleteProposal={() => { const id = planStateRef.current?.proposal.id; if (id) void plan.remove(scene, id).then(() => { undoStack.current = []; redoStack.current = []; setPlanSelected(null); setAnalysis(null); return loadPlans(null); }).catch((cause) => setPlanMessage(String(cause))); }}
            onTool={choosePlanTool} onFinish={() => void finishPlan()}
            onUndoPoint={() => { const next = planPoints.slice(0, -1); setPlanPoints(next); command("set-picks", { value: next }); command("pick", { value: true, limit: 128 }); }}
            onUndo={() => void undoPlan(false)} onRedo={() => void undoPlan(true)}
            onUpdate={updateFeature} onDeleteFeature={deleteFeature} onSelect={setPlanSelected}
            analysis={analysis} analysisBusy={analysisBusy} analysisError={analysisError} showViewsheds={showViewsheds} onViewsheds={setShowViewsheds} onAnalyse={() => void analyse()}
            los={los} onCoveredRoute={(id) => void coveredRoute(id)}
            hlz={hlz} onFindHlz={(d) => { missionApi.hlz(scene, d).then((r) => setHlz(r.candidates)).catch((cause) => setPlanMessage(String(cause))); }}
            onAddHlz={(c, d) => { const current = planStateRef.current; if (current) void planWrite(() => plan.upsert(scene, current.proposal, { type: "symbol", name: "HLZ", params: { position: c.centre, role: "hlz", affiliation: "friendly", diameter_m: d } as never }), { done: "HLZ added." }); }}
            candidates={candidates} onFindCandidates={() => { missionApi.candidates(scene).then((r) => setCandidates(r.candidates)).catch((cause) => setPlanMessage(String(cause))); }}
            onAddCandidate={(c) => { const current = planStateRef.current; if (current) void planWrite(() => plan.upsert(scene, current.proposal, { type: "symbol", name: `${c.role === "vehicle" ? "Vehicle" : "Person"} (${c.source === "detections" ? "seen in video" : "from labels"})`, params: { position: c.position, role: c.role, affiliation: "unknown", source: c.source } as never }), { done: "Added as UNKNOWN: confirm it before treating it as hostile." }); }}
            conditions={conditions} onConditions={setConditions} onRehearse={rehearse}
            runs={runs} onOpenRun={(id) => void openReview(id)}
            onDeleteRun={(id) => { const current = planStateRef.current; if (current) missionApi.deleteRun(scene, current.proposal.id, id).then((r) => setRuns(r.runs)).catch((cause) => setPlanMessage(String(cause))); }}
            threat={threat} threatBusy={threatBusy} showThreat={showThreat} onShowThreat={setShowThreat}
            onThreat={(routeId, range) => { const current = planStateRef.current; if (!current) return; setThreatBusy(true); missionApi.threat(scene, current.proposal.id, routeId, range).then((r) => { setThreat(r); setShowThreat(true); }).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))).finally(() => setThreatBusy(false)); }}
            onAddThreat={(position) => { const current = planStateRef.current; if (current) void planWrite(() => plan.upsert(scene, current.proposal, { type: "symbol", name: "Possible enemy (heatmap)", params: { position, role: "infantry", affiliation: "unknown", source: "planner" } as never }), { done: "Added as UNKNOWN: a likely position, not an observation." }); }}
            obstacles={obstacles} onObstacles={(h) => { missionApi.obstacles(scene, h).then(setObstacles).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))); }}
            traffic={traffic} showTraffic={showTraffic} onShowTraffic={setShowTraffic}
            onTraffic={(m) => { missionApi.trafficability(scene, m).then((r) => { setTraffic(r); setShowTraffic(true); }).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))); }}
            onInstructor={() => { setInstructor(true); if (!basemap) missionApi.basemap(scene).then(setBasemap).catch(() => {}); }}
            onSandTable={() => { setSandTable(true); if (!basemap) missionApi.basemap(scene).then(setBasemap).catch(() => {}); }}
            onPack={() => { const current = planStateRef.current; if (current) missionApi.pack(scene, current.proposal.id).then((f) => { downloadFile(f); setPlanMessage(`Exported ${f.filename}.`); }).catch((cause) => setPlanMessage(cause instanceof Error ? cause.message : String(cause))); }}
          />}
          {tab === "ops" && <OpsPanel scene={scene} ready={ready && !error} tool={opsTool} points={opsPoints}
            onTool={chooseOpsTool} onOverlay={setOpsOverlay} onDownload={(file) => { downloadFile(file); setMessage(`Downloaded ${file.filename}.`); }}
            onPlan={() => switchTab("plan")}
            onUndoPoint={() => { const next = opsPoints.slice(0, -1); setOpsPoints(next); command("set-picks", { value: next }); command("pick", { value: true, limit: opsTool ? TOOL_HINT[opsTool].limit : 1 }); }} />}
          {tab === "inspect" && <InspectPanel scene={scene} ready={ready && !error} tool={inspTool} points={inspPoints}
            onTool={chooseInspTool} onOverlay={setInspOverlay} onPlan={() => switchTab("plan")}
            onUndoPoint={() => { const next = inspPoints.slice(0, -1); setInspPoints(next); command("set-picks", { value: next }); command("pick", { value: true, limit: inspTool ? INSPECT_TOOLS[inspTool].limit : 1 }); }} />}
          {tab === "twin" && <TwinPanel scene={scene} ready={ready && !error} capabilities={capabilities} view={twinView} onView={chooseTwinView} onOverlay={setTwinOverlay} />}
          {tab === "details" && <DetailsPanel key={scene} project={project} onSaved={setProject} refresh={() => void refresh()} />}
          {tab === "exports" && <ExportPanel project={project} />}
          {tab === "quality" && <QualityPanel project={project} />}
        </aside>}
      </div>
      {logsOpen && <section className="job-panel" aria-label="Processing details"><ul className="step-list">{project.steps.map((step) => <li key={step.name} className={step.status}><Icon name={step.status === "done" || step.status === "recovered" ? "check" : "clock"} size={12} /><span>{step.name}</span><small>{duration(step.secs)}</small></li>)}</ul><pre className="job-terminal" role="log" aria-label="Processing output">{logs.length ? logs.join("\n") : "No processing output available yet."}</pre></section>}
      <footer className="workspace-statusbar"><span>{project.registered_count} CAMERAS</span><span>{project.scale.unit === "m" ? "METRES" : "RELATIVE UNITS"} · Y-UP</span><span>{game ? `BOT SESSION · ${bots} BOTS` : ready ? "VIEWER CONNECTED" : "VIEWER NOT READY"}</span><span>{tab === "measure" ? "MEASUREMENT" : tab === "plan" ? "PLANNING EDITOR" : tab === "mission" ? "MISSION PLANNING" : tab === "ops" ? "OPERATIONS" : tab === "inspect" ? "INSPECTION" : tab === "twin" ? "DIGITAL TWIN" : tab === "place" ? "3D PLACEMENT EDITOR" : tab === "walk" ? "ARENA" : WORKFLOWS[project.workflow]?.label.toUpperCase()} WORKSPACE</span></footer>
    </main>
    {runOpen && <RunDialog project={project} options={options} onClose={() => setRunOpen(false)} onStarted={() => { cursor.current = "0"; setLogs([]); setLogsOpen(true); void refresh(); }} />}
  </Studio>;
}
