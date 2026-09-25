"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { pipeline } from "@/lib/api";
import { duration, placementsGeoJSON, WORKFLOWS, workspace, downloadBlob, type Layer, type MeasurementKind, type Mode, type Placement, type Point, type ProjectDetail, type ProjectList } from "@/lib/workspace";
import { Studio, Status, Offline } from "./studio";
import { Icon, type IconName } from "./studio-icons";
import { RunDialog } from "./run-dialog";
import { ViewerPanel } from "./viewer-panel";
import { MeasurementPanel } from "./measurement-panel";
import { PlacementPanel } from "./placement-panel";
import { DetailsPanel, ExportPanel } from "./project-details";

const LAYERS: { key: Layer; label: string; detail: string; icon: IconName }[] = [
  { key: "splats", label: "Photorealistic model", detail: "Gaussian splats", icon: "cube" },
  { key: "cameras", label: "Camera positions", detail: "Recovered capture trajectory", icon: "camera" },
  { key: "points", label: "Sparse point cloud", detail: "Triangulated feature points", icon: "grid" },
  { key: "coverage", label: "Observed coverage", detail: "View-support diagnostic", icon: "layers" },
  { key: "collider", label: "Collision surface", detail: "Navigation geometry · estimated", icon: "compass" },
  { key: "semantics", label: "Semantic classes", detail: "Ground · road · building · vegetation · obstacle", icon: "layers" },
];
const INITIAL_LAYERS: Record<Layer, boolean> = { splats: true, cameras: false, points: false, coverage: false, collider: false, semantics: false };

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
  const placeRef = useRef<{ item: string; yaw: number } | null>(null);
  useEffect(() => { placeRef.current = place; }, [place]);
  const [moving, setMoving] = useState<string | null>(null);
  const moveRef = useRef<Placement | null>(null);
  useEffect(() => { moveRef.current = project?.placements.find((p) => p.id === moving) ?? null; }, [moving, project]);
  const [message, setMessage] = useState("");
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const iframe = useRef<HTMLIFrameElement>(null);
  const revision = useRef<string | null>(null);
  const cursor = useRef("0");
  const refresh = useCallback(async () => {
    try {
      const next = await workspace.project(scene);
      if (revision.current !== next.model_revision) { revision.current = next.model_revision; setReady(false); setPoints([]); setKind(null); setMessage(""); }
      setProject(next); setError("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, [scene]);
  useEffect(() => { let active = true; let timer: ReturnType<typeof setTimeout>; const poll = async () => { if (!active) return; await refresh(); if (active) timer = setTimeout(poll, 3500); }; void poll(); workspace.list().then(setOptions).catch(() => {}); return () => { active = false; clearTimeout(timer); }; }, [refresh]);
  const command = useCallback((name: string, values: Record<string, unknown> = {}) => {
    iframe.current?.contentWindow?.postMessage({ namespace: "groundcontrol", type: "command", command: name, ...values }, window.location.origin);
  }, []);
  // Push saved measurements into the viewer so they render as on-model geometry
  // with floating labels; re-sent whenever the set or the ready model changes.
  useEffect(() => {
    if (ready && project) command("measurements", { value: project.measurements.map((m) => ({ id: m.id, kind: m.kind, points: m.points, value: m.value, unit: m.unit, label: m.label, uncertainty: m.uncertainty, valid: m.valid })) });
  }, [ready, project, command]);
  // Push placed furniture so the viewer renders solid, collidable boxes with fit-
  // tinted labels; re-sent whenever the layout or the ready model changes.
  useEffect(() => {
    if (ready && project) command("placements", { value: project.placements.map((p) => ({ id: p.id, item: p.item, label: p.label, center_xz: p.center_xz, center_y: p.center_y, size: p.size, yaw_deg: p.yaw_deg, fit: p.fit })) });
  }, [ready, project, command]);
  useEffect(() => {
    const receive = (event: MessageEvent) => {
      if (event.source !== iframe.current?.contentWindow || event.origin !== window.location.origin || !event.data || event.data.namespace !== "groundcontrol") return;
      const data = event.data;
      if (data.type === "ready" || data.type === "state") {
        if (["orbit", "fly", "walk"].includes(data.mode)) setMode(data.mode);
        if (data.layers && typeof data.layers === "object") setLayers((previous) => Object.fromEntries(Object.keys(previous).map((key) => [key, typeof data.layers[key] === "boolean" ? data.layers[key] : previous[key as Layer]])) as Record<Layer, boolean>);
        if (data.type === "ready") { setReady(true); setMessage(""); if (data.capabilities && typeof data.capabilities === "object") setCapabilities(data.capabilities); }
      } else if (data.type === "pick" && moveRef.current && Array.isArray(data.point) && data.point.length === 3 && data.point.every((value: unknown) => typeof value === "number" && Number.isFinite(value))) {
        const target = moveRef.current;
        void workspace.movePlacement(scene, { id: target.id, item: target.item, point: data.point as Point, yaw_deg: target.yaw_deg, scale: target.scale, model_revision: revision.current ?? "" })
          .then((next) => { setProject(next); setMoving(null); command("pick", { value: false }); command("clear-picks"); setMessage("Moved. Re-fit checked against the new spot."); })
          .catch((cause) => setMessage(cause instanceof Error ? cause.message : String(cause)));
      } else if (data.type === "pick" && placeRef.current && Array.isArray(data.point) && data.point.length === 3 && data.point.every((value: unknown) => typeof value === "number" && Number.isFinite(value))) {
        const current = placeRef.current;
        void workspace.place(scene, { item: current.item, point: data.point as Point, yaw_deg: current.yaw, model_revision: revision.current ?? "" })
          .then((next) => { setProject(next); command("clear-picks"); command("pick", { value: true, limit: 1 }); setMessage("Placed. Click the floor again to drop another."); })
          .catch((cause) => setMessage(cause instanceof Error ? cause.message : String(cause)));
      } else if (data.type === "pick" && kind && Array.isArray(data.point) && data.point.length === 3 && data.point.every((value: unknown) => typeof value === "number" && Number.isFinite(value))) {
        setPoints((previous) => [...previous, data.point as Point].slice(0, kind === "point" ? 1 : kind === "height" ? 2 : 128));
        setMessage("Point selected on estimated collision geometry.");
      } else if ((data.type === "error" || data.type === "pick-miss") && typeof data.message === "string") setMessage(data.message.slice(0, 500));
      else if (data.type === "snapshot" && typeof data.dataUrl === "string" && data.dataUrl.startsWith("data:image/png;base64,") && data.dataUrl.length < 40000000) {
        const link = document.createElement("a"); link.href = data.dataUrl; link.download = `${scene}-view.png`; link.click(); setMessage("Viewport snapshot downloaded.");
      }
    };
    window.addEventListener("message", receive); return () => window.removeEventListener("message", receive);
  }, [command, kind, scene]);
  useEffect(() => {
    if (!logsOpen) return;
    let active = true; let timer: ReturnType<typeof setTimeout>;
    const poll = async () => { try { const result = await pipeline.tail(scene, cursor.current); if (active) { cursor.current = result.cursor; setLogs((previous) => [...previous, ...result.lines].slice(-400)); } } catch { if (active) setLogs(["Log service unavailable. Retry when the backend reconnects."]); } if (active) timer = setTimeout(poll, 2500); };
    void poll(); return () => { active = false; clearTimeout(timer); };
  }, [logsOpen, scene]);
  function clear() { setKind(null); setPoints([]); setPlace(null); setMoving(null); setMessage(""); if (ready) { command("pick", { value: false }); command("clear-picks"); } }
  function chooseTool(value: MeasurementKind) { setPlace(null); setMoving(null); setKind(value); setPoints([]); command("clear-picks"); command("pick", { value: true, limit: value === "point" ? 1 : value === "height" ? 2 : 128 }); setMessage("Click the collision surface to select points. Drag to move the view."); }
  function choosePlacement(item: string) { setKind(null); setPoints([]); setMoving(null); command("clear-picks"); setPlace({ item, yaw: 0 }); command("pick", { value: true, limit: 1 }); setMessage("Click the scanned floor to drop the item."); }
  function startMove(p: Placement) { setKind(null); setPoints([]); setPlace(null); command("clear-picks"); setMoving(p.id); command("pick", { value: true, limit: 1 }); setMessage(`Moving ${p.label}: click where it should go.`); }
  function yawDelta(delta: number) { setPlace((previous) => previous ? { ...previous, yaw: ((previous.yaw + delta) % 360 + 360) % 360 } : previous); }
  function clearPlace() { setPlace(null); setMoving(null); if (ready) { command("pick", { value: false }); command("clear-picks"); } }
  function undoPoint() { setPoints((previous) => { const next = previous.slice(0, -1); command("set-picks", { value: next }); return next; }); }
  async function rotatePlacement(p: Placement, yaw: number) { try { setProject(await workspace.movePlacement(scene, { id: p.id, item: p.item, point: [p.center_xz[0], p.center_y, p.center_xz[1]], yaw_deg: yaw, model_revision: revision.current ?? "" })); } catch (cause) { setMessage(cause instanceof Error ? cause.message : String(cause)); } }
  async function deletePlacement(id: string) { try { setProject(await workspace.removePlacement(scene, id)); } catch (cause) { setMessage(cause instanceof Error ? cause.message : String(cause)); } }
  function exportLayout() { if (project) downloadBlob(`${scene}-layout.geojson`, "application/geo+json", JSON.stringify(placementsGeoJSON(project.placements, scene), null, 2)); }
  function chooseFrame(index: number) { clear(); setSelectedFrame(index); setTab("layers"); const camera = project?.frames[index]?.camera_index; if (typeof camera === "number" && ready) command("frame", { index: camera }); else setMessage("Showing the original image. A recovered camera pose and ready viewer are needed to position the 3D view."); }
  function switchTab(value: string) { if (value !== "measure" && value !== "place") clear(); setTab(value); }
  async function cancel() { try { await workspace.cancel(scene); setConfirmCancel(false); await refresh(); } catch (cause) { setMessage(cause instanceof Error ? cause.message : String(cause)); } }
  const running = project?.job.scene === scene && ["running", "starting"].includes(project.job.status);
  if (!project) return <Studio connected={!error}><main className="library-main">{error ? <Offline message={error} retry={() => void refresh()} /> : <div className="loading-state"><span className="spinner" />Opening project…</div>}</main></Studio>;
  return <Studio project={project.name} connected={!error}>
    <main className="workspace-main">
      <header className="workspace-header"><div className="workspace-title"><span className="brand-symbol"><Icon name="cube" size={25} /></span><div><h1>{project.name}</h1><Status status={project.status} /></div></div><div className="workspace-actions"><button className="button secondary small" onClick={() => setLogsOpen(!logsOpen)}><Icon name="activity" size={15} /><span>{logsOpen ? "Hide processing" : "Processing"}</span></button><button className="button secondary small" onClick={() => switchTab("exports")}><Icon name="download" size={15} />Exports</button><button className="button primary small" onClick={() => setRunOpen(true)}><Icon name="play" size={14} />Reconstruct</button></div></header>
      {error && <div className="job-banner"><Icon name="info" /><span>Connection lost. Showing the last project state; changes cannot be saved until reconnected.</span><button className="text-button" onClick={() => void refresh()}>Retry</button></div>}
      {running && <div className="job-banner"><span className="spinner" /><span>Processing this capture<small>{project.job.step || "Starting reconstruction"} · you can keep exploring existing outputs</small></span><button className="button danger small" onClick={() => confirmCancel ? void cancel() : setConfirmCancel(true)}>{confirmCancel ? "Confirm stop" : "Stop job"}</button>{confirmCancel && <button className="text-button" onClick={() => setConfirmCancel(false)}>Keep running</button>}</div>}
      <div className="workspace-grid">
        <ViewerPanel project={project} iframeRef={iframe} ready={ready} mode={mode} selectedFrame={selectedFrame} message={message} framesOpen={framesOpen} onCommand={command} onFrame={chooseFrame} onFramesToggle={() => setFramesOpen(!framesOpen)} onRun={() => setRunOpen(true)} />
        <aside className="inspector" aria-label="Project tools"><div className="inspector-tabs">{[["layers", "Explore"], ["measure", "Measure"], ["place", "Place"], ["details", "Details"]].map(([value, label]) => <button key={value} className={tab === value ? "active" : ""} onClick={() => switchTab(value)}>{label}</button>)}</div>
          {tab === "layers" && <><section className="inspector-section"><div className="section-label"><span>SCENE LAYERS</span><Icon name="layers" size={14} /></div>{LAYERS.map((layer) => <label className="layer-control" key={layer.key}><Icon name={layer.icon} size={17} /><span>{layer.label}<small>{layer.detail}</small></span><input type="checkbox" aria-label={layer.label} checked={layers[layer.key]} disabled={!ready || (layer.key !== "splats" && !capabilities[layer.key])} onChange={(event) => { const value = event.target.checked; setLayers((previous) => ({ ...previous, [layer.key]: value })); command("layer", { layer: layer.key, value }); }} /></label>)}</section>{project.semantics && <section className="inspector-section"><div className="section-label"><span>SCENE CLASSIFICATION</span><Icon name="layers" size={14} /></div><div className="class-list">{Object.entries(project.semantics.counts).filter(([, n]) => n > 0).map(([cls, n]) => { const s = project.semantics?.summary?.[cls as string]; return <div className="datum-row" key={cls}><span><i className={`class-dot class-${cls}`} />{cls}</span><span>{n.toLocaleString()}{s?.area_m2 ? ` · ${s.area_m2.toLocaleString()} m²` : ""}</span></div>; })}</div><p className="inspector-copy">Heuristic geometric + colour labels; area is a grid-occupancy footprint proxy (coarse, not surveyed ground truth).</p></section>}<section className="inspector-section"><div className="section-label">SPATIAL REFERENCE</div><div className="datum-row"><span>Scale</span><span>{project.scale.status === "metric" ? "Metric · pose referenced" : project.scale.status === "estimated" ? "Estimated" : "Relative"}</span></div><div className="datum-row"><span>Location</span><span>{project.georeference.status === "local" ? "Local coordinates" : project.georeference.crs}</span></div><div className="datum-row"><span>Accuracy</span><span>{project.accuracy.status === "verified" ? `${project.accuracy.rmse_m} m RMSE` : "Not independently verified"}</span></div><p className="inspector-copy">{project.scale.source}</p></section><section className="inspector-section"><div className="section-label">{WORKFLOWS[project.workflow]?.label ?? "Explore"} WORKFLOW</div><p className="inspector-copy">{project.workflow === "inspection" ? "Select a source frame to inspect detail. Use annotations to record observations on the model." : project.workflow === "survey" ? "Inspect scale before measuring. Use Details for survey inputs and Exports for generated products." : project.workflow === "response" ? "Enable observed coverage to see view support. Unknown areas are not proof of safe passage." : project.workflow === "heritage" ? "Use orbit or walk to explore. Save a snapshot of the current viewpoint from the viewport toolbar." : "Orbit the reconstruction, inspect original camera views, and switch layers to understand its geometry."}</p></section>{selectedFrame >= 0 && project.frames[selectedFrame] && <div className="frame-inspection"><Image unoptimized width={640} height={400} src={project.frames[selectedFrame].url} alt={`Inspected source frame ${selectedFrame + 1}`} /><p>Frame {selectedFrame + 1} · {project.frames[selectedFrame].name}</p></div>}</>}
          {tab === "measure" && <MeasurementPanel project={project} ready={ready && !!capabilities.collider && !error} points={points} kind={kind} onTool={chooseTool} onClear={clear} onUndo={undoPoint} onSaved={setProject} onSelect={(id) => command("select", { id })} />}
          {tab === "place" && <PlacementPanel project={project} ready={ready && !!capabilities.collider && !error} place={place} moving={moving} onPick={choosePlacement} onYaw={yawDelta} onClear={clearPlace} onMove={startMove} onRotate={rotatePlacement} onDelete={deletePlacement} onSelect={(id) => command("select", { id })} onExport={exportLayout} onView={(value) => { command("view", { value }); if (value === "top") setMessage("Plan view: looking down at the floor. Drag to orbit, scroll to zoom."); }} />}
          {tab === "details" && <DetailsPanel key={scene} project={project} onSaved={setProject} refresh={() => void refresh()} />}
          {tab === "exports" && <ExportPanel project={project} />}
        </aside>
      </div>
      {logsOpen && <section className="job-panel" aria-label="Processing details"><ul className="step-list">{project.steps.map((step) => <li key={step.name} className={step.status}><Icon name={step.status === "done" || step.status === "recovered" ? "check" : "clock"} size={12} /><span>{step.name}</span><small>{duration(step.secs)}</small></li>)}</ul><pre className="job-terminal" role="log" aria-label="Processing output">{logs.length ? logs.join("\n") : "No processing output available yet."}</pre></section>}
      <footer className="workspace-statusbar"><span>{project.registered_count} CAMERAS</span><span>{project.scale.unit === "m" ? "METRES" : "RELATIVE UNITS"} · Y-UP</span><span>{ready ? "VIEWER CONNECTED" : "VIEWER NOT READY"}</span><span>{WORKFLOWS[project.workflow]?.label.toUpperCase()} WORKSPACE</span></footer>
    </main>
    {runOpen && <RunDialog project={project} options={options} onClose={() => setRunOpen(false)} onStarted={() => { cursor.current = "0"; setLogs([]); setLogsOpen(true); void refresh(); }} />}
  </Studio>;
}
