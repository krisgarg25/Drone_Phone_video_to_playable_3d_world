"use client";

import { useState } from "react";
import {
  SYMBOL_ROLES, TOOL_COPY,
  type Behaviour, type PhaseLineParams, type PlanFeature, type PlanState, type PlanTool, type ProposalIndex, type RouteParams,
  type SymbolParams, type SymbolRole,
} from "@/lib/plan";
import {
  fmtClock, type Conditions, type HlzCandidate, type LabelCandidate, type LosResult, type MissionAnalysis, type ObstacleList, type RouteReport, type RunListing,
  type ThreatMap, type Trafficability,
} from "@/lib/mission";
import { NumberField, Select } from "./plan-panel";
import { Icon, type IconName } from "./studio-icons";
import { OnViewer } from "./on-viewer";
import "./plan-panel.css";
import "./inspect.css";
import "./ops-panel.css";

type Props = {
  scene: string; state: PlanState | null; index: ProposalIndex | null; selected: string | null; tool: PlanTool | null;
  points: number; busy: boolean; message: string; ready: boolean; canUndo: boolean; canRedo: boolean;
  role: SymbolRole; onRole: (role: SymbolRole) => void;
  onProposal: (id: string) => void; onCreate: (name: string, source?: string) => void; onRename: (name: string) => void; onDeleteProposal: () => void;
  onTool: (tool: PlanTool | null) => void; onFinish: () => void; onUndoPoint: () => void; onUndo: () => void; onRedo: () => void;
  onUpdate: (feature: PlanFeature) => void; onDeleteFeature: (id: string) => void; onSelect: (id: string | null) => void;
  analysis: MissionAnalysis | null; analysisBusy: boolean; analysisError: string; showViewsheds: boolean; onViewsheds: (on: boolean) => void; onAnalyse: () => void;
  los: LosResult | null; onCoveredRoute: (routeId: string) => void;
  hlz: HlzCandidate[] | null; onFindHlz: (diameter: number) => void; onAddHlz: (c: HlzCandidate, diameter: number) => void;
  candidates: LabelCandidate[] | null; onFindCandidates: () => void; onAddCandidate: (c: LabelCandidate) => void;
  conditions: Conditions; onConditions: (c: Conditions) => void; onRehearse: () => void;
  runs: RunListing[]; onOpenRun: (id: string) => void; onDeleteRun: (id: string) => void; onPack: () => void;
  threat: ThreatMap | null; threatBusy: boolean; showThreat: boolean; onShowThreat: (on: boolean) => void; onThreat: (routeId: string, range: number) => void;
  onAddThreat: (position: [number, number]) => void;
  obstacles: ObstacleList | null; onObstacles: (minHeight: number) => void;
  onSandTable: () => void; onInstructor: () => void;
  traffic: Trafficability | null; showTraffic: boolean; onShowTraffic: (on: boolean) => void; onTraffic: (mobility: string) => void;
};

const TOOLS: { tool: PlanTool; icon: IconName; label: string }[] = [
  { tool: "symbol", icon: "pin", label: "Symbol" }, { tool: "route", icon: "route", label: "Route" },
  { tool: "phase_line", icon: "flag", label: "Phase line" }, { tool: "los", icon: "eye", label: "Line of sight" },
];
const TYPE_LABEL: Record<string, string> = { symbol: "Symbols", route: "Routes", phase_line: "Phase lines" };
const BEHAVIOUR_COPY: Record<Behaviour, string> = {
  sentry: "Stays put and watches its arc", patrol: "Walks a patrol route", overwatch: "Stays put and sees far through a narrow arc",
  reaction: "Moves towards noise and where it last saw you",
};
const dot = (f: PlanFeature) => f.type === "symbol" ? (f.params as SymbolParams).affiliation : f.type === "route" ? "route" : "phase";

function SymbolForm({ feature, routes, disabled, onUpdate }: { feature: PlanFeature; routes: PlanFeature[]; disabled: boolean; onUpdate: (f: PlanFeature) => void }) {
  const p = feature.params as SymbolParams;
  const set = (patch: Partial<SymbolParams>) => onUpdate({ ...feature, params: { ...p, ...patch } });
  const armed = !["objective", "rally_point", "hlz", "obstacle", "checkpoint"].includes(p.role);
  return <div className="plan-form">
    <Select label="What it is" value={p.role} disabled={disabled} options={SYMBOL_ROLES.map((r) => [r.role, r.label])} onChange={(role) => set({ role })} />
    <Select label="Side" value={p.affiliation} disabled={disabled} options={[["hostile", "Enemy"], ["friendly", "Friendly"], ["unknown", "Unknown"], ["neutral", "Neutral"]]} onChange={(affiliation) => set({ affiliation })} />
    {armed && <>
      <NumberField label="Facing (compass)" unit="°" value={Math.round(p.bearing_deg)} min={0} max={359} step={5} disabled={disabled} onCommit={(v) => set({ bearing_deg: v ?? 0 })} />
      <NumberField label="Arc it covers" unit="°" value={p.sector_deg} min={10} max={360} step={5} disabled={disabled} onCommit={(v) => set({ sector_deg: v ?? 90 })} />
      <NumberField label="How far it reaches" unit="m" value={p.range_m} min={5} max={2000} step={5} disabled={disabled} onCommit={(v) => set({ range_m: v ?? 120 })} />
    </>}
    {p.affiliation === "hostile" && armed && <>
      <Select label="Behaviour" value={p.behaviour} disabled={disabled} options={Object.entries(BEHAVIOUR_COPY).map(([k]) => [k as Behaviour, k[0].toUpperCase() + k.slice(1)])} onChange={(behaviour) => set({ behaviour })} />
      <p className="plan-note">{BEHAVIOUR_COPY[p.behaviour]}.</p>
      {p.behaviour === "patrol" && <Select label="Patrol route" value={p.patrol_route ?? ""} disabled={disabled}
        options={[["", routes.length ? "Choose a route" : "Draw a route first"], ...routes.map((r) => [r.id, r.name] as [string, string])]}
        onChange={(id) => set({ patrol_route: id || null })} />}
      <NumberField label="Enemies here" value={p.count} min={1} max={8} disabled={disabled} onCommit={(v) => set({ count: v ?? 1 })} />
    </>}
    {p.role === "hlz" && <NumberField label="Landing zone diameter" unit="m" value={p.diameter_m} min={5} max={200} disabled={disabled} onCommit={(v) => set({ diameter_m: v ?? 25 })} />}
    <NumberField label="Height above ground" unit="m" value={p.elevation_m} min={0} max={100} step={0.5} disabled={disabled} onCommit={(v) => set({ elevation_m: v ?? 0 })} />
    {p.source === "labels" && <p className="plan-note">Suggested from the scan&apos;s person/vehicle labels — a heuristic, not a confirmed detection.</p>}
    {p.source === "detections" && <p className="plan-note">Seen in the video by the detector and placed on the scan — an observation at capture time, not a confirmed enemy.</p>}
  </div>;
}

function RouteForm({ feature, disabled, onUpdate }: { feature: PlanFeature; disabled: boolean; onUpdate: (f: PlanFeature) => void }) {
  const p = feature.params as RouteParams;
  const set = (patch: Partial<RouteParams>) => onUpdate({ ...feature, params: { ...p, ...patch } });
  const [names, setNames] = useState(p.waypoints.map((_, i) => p.names[i] ?? "").join(", "));
  const [shown, setShown] = useState(p);
  if (shown !== p) { setShown(p); setNames(p.waypoints.map((_, i) => p.names[i] ?? "").join(", ")); }
  return <div className="plan-form">
    <Select label="Kind" value={p.kind} disabled={disabled} options={[["approach", "Approach"], ["withdrawal", "Withdrawal"], ["patrol", "Enemy patrol"]]} onChange={(kind) => set({ kind })} />
    <Select label="Pace" value={p.pace} disabled={disabled} options={[["walk", "Walk"], ["patrol", "Tactical (0.8×)"], ["run", "Run (1.5×)"]]} onChange={(pace) => set({ pace })} />
    <label className="plan-field plan-field-wide"><span>Waypoint names ({p.waypoints.length})</span>
      <input value={names} disabled={disabled} onChange={(e) => setNames(e.target.value)} placeholder="SP, WP2, OBJ"
        onBlur={() => set({ names: names.split(",").map((n) => n.trim()).slice(0, p.waypoints.length) })}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} /></label>
  </div>;
}

function Report({ report, onCovered, busy }: { report: RouteReport; onCovered: () => void; busy: boolean }) {
  return <div className="mission-report">
    <div className="insp-kpis">
      <div><b>{report.length_m.toFixed(0)} m</b><small>long</small></div>
      <div><b>{fmtClock(report.eta_s)}</b><small>walking time</small></div>
      <div className={report.exposure_s > 0 ? "bad" : "good"}><b>{report.exposure_s.toFixed(0)} s</b><small>seen by the enemy</small></div>
      <div className="good"><b>{report.dead_ground_pct.toFixed(0)}%</b><small>on hidden ground</small></div>
    </div>
    <div className="mission-profile" aria-label="Ground height along the route; orange and red where the enemy can see you" data-tip="Ground height along the route. Orange: seen by one enemy. Red: seen by more.">
      {report.profile.station_m.map((s, i) => <i key={i} style={{ height: `${20 + 80 * ((report.profile.height_m[i] - Math.min(...report.profile.height_m)) / Math.max(0.5, Math.max(...report.profile.height_m) - Math.min(...report.profile.height_m)))}%` }} className={report.profile.seen_by[i] > 1 ? "hot2" : report.profile.seen_by[i] ? "hot" : ""} />)}
    </div>
    {report.per_hostile.length > 0 && <table className="plan-table"><thead><tr><th>Enemy</th><th>Sees you for</th><th>Spots you</th></tr></thead>
      <tbody>{report.per_hostile.map((h) => <tr key={h.id}><td>{h.name}<small>{h.closest_m.toFixed(0)} m closest</small></td><td className={h.exposure_s ? "up" : ""}>{h.exposure_s.toFixed(0)} s</td><td>{h.first_seen_m === null ? "never" : `after ${h.first_seen_m.toFixed(0)} m`}</td></tr>)}</tbody></table>}
    <ol className="mission-timeline">{[...report.waypoints.map((w) => ({ t: w.time_s, text: w.name })), ...report.phase_lines.map((p) => ({ t: p.time_s, text: `crosses ${p.label}` }))]
      .sort((a, b) => a.t - b.t).map((e, i) => <li key={i}><span>{fmtClock(e.t)}</span>{e.text}</li>)}</ol>
    <p className="plan-note">Climbs {report.climb_m.toFixed(0)} m · steepest slope {report.max_slope_pct.toFixed(0)}%{report.unscanned_m > 0 ? ` · ${report.unscanned_m.toFixed(0)} m over unscanned ground` : ""}.</p>
    {report.exposure_s > 0 && <button className="button secondary full" disabled={busy} onClick={onCovered}><Icon name="shield" size={14} />Suggest a route that stays hidden</button>}
  </div>;
}

type Task = "mark" | "items" | "exposure" | "hlz" | "enemy" | "terrain" | "rehearse" | "brief";
const TASKS: { id: Task; label: string; icon: IconName; what: string }[] = [
  { id: "mark", label: "Mark", icon: "plus", what: "Put enemy posts, the objective, routes and phase lines on the ground." },
  { id: "items", label: "Plan items", icon: "list", what: "Everything marked so far. Click one here or on the model to change it." },
  { id: "exposure", label: "Exposure", icon: "eye", what: "How long each route is seen by the enemy, and where the hidden ground is." },
  { id: "hlz", label: "Landing zones", icon: "target", what: "Find flat, clear circles big enough for a helicopter." },
  { id: "enemy", label: "Enemy guess", icon: "alert", what: "Where an enemy would most likely watch your route from, and people or vehicles seen in the video." },
  { id: "terrain", label: "Terrain", icon: "mountain", what: "Where vehicles and people can move, and tall obstacles." },
  { id: "rehearse", label: "Rehearse", icon: "play", what: "Walk the mission in first person against the planned enemy, then review the run." },
  { id: "brief", label: "Brief", icon: "flag", what: "A sand-table view for the briefing, a live exercise, and a pack for GPS devices." },
];
const SIDES: { label: string; affiliation: string }[] = [{ label: "Enemy", affiliation: "hostile" }, { label: "Friendly", affiliation: "friendly" }, { label: "Other", affiliation: "unknown" }];

export function MissionPanel(props: Props) {
  const { state, index, selected, tool, points, busy, message, ready, analysis, conditions } = props;
  const [naming, setNaming] = useState<"new" | "rename" | null>(null);
  const [name, setName] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [diameter, setDiameter] = useState(25);
  const [threatRoute, setThreatRoute] = useState("");
  const [threatRange, setThreatRange] = useState(300);
  const [obstacleHeight, setObstacleHeight] = useState(3);
  const [mobility, setMobility] = useState("wheeled");
  const [task, setTask] = useState<Task>("mark");
  // A symbol or route picked on the model opens its settings.
  const [seen, setSeen] = useState(selected);
  if (seen !== selected) { setSeen(selected); if (selected) setTask("items"); }
  const proposal = state?.proposal ?? null;
  const feature = proposal?.features.find((f) => f.id === selected) ?? null;
  const locked = busy || !ready;
  const routes = proposal?.features.filter((f) => f.type === "route") ?? [];
  const hostiles = proposal?.features.filter((f) => f.type === "symbol" && (f.params as SymbolParams).affiliation === "hostile").length ?? 0;
  const objectives = proposal?.features.filter((f) => f.type === "symbol" && (f.params as SymbolParams).role === "objective").length ?? 0;
  const submit = () => { const v = name.trim(); if (!v) return; if (naming === "rename") props.onRename(v); else props.onCreate(v); setNaming(null); setName(""); };
  const active = TASKS.find((t) => t.id === task)!;
  const done: Partial<Record<Task, boolean>> = { exposure: !!analysis, hlz: !!props.hlz, enemy: !!props.threat || !!props.candidates, terrain: !!props.traffic || !!props.obstacles, rehearse: props.runs.length > 0 };
  const worst = analysis?.routes.reduce((max, r) => Math.max(max, r.report.exposure_s), 0) ?? 0;
  const legend: [string, string][] | null = task === "exposure" && analysis && props.showViewsheds && analysis.hostiles > 0 ? [["rgb(250,214,90)", "Seen by 1 enemy"], ["rgb(240,140,50)", "By 2"], ["rgb(214,50,50)", "By 3 or more"]]
    : task === "enemy" && props.threat && props.showThreat ? [["rgb(250,214,90)", "Possible"], ["rgb(240,140,50)", "Likely"], ["rgb(214,50,50)", "Most likely"]]
      : task === "terrain" && props.traffic && props.showTraffic ? [["rgb(70,190,90)", "Can move freely"], ["rgb(240,190,50)", "Slow going"], ["rgb(210,60,60)", "Cannot pass"]] : null;

  const missionBar = <>
    {index && index.proposals.length > 0 && !naming && <div className="plan-scheme">
      <select aria-label="Active mission" value={proposal?.id ?? ""} disabled={busy} onChange={(e) => props.onProposal(e.target.value)}>{index.proposals.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
      <button className="icon-button" data-tip="Undo (Ctrl+Z)" aria-label="Undo" disabled={!props.canUndo || busy} onClick={props.onUndo}><Icon name="reset" size={15} /></button>
      <button className="icon-button plan-redo" data-tip="Redo (Ctrl+Y)" aria-label="Redo" disabled={!props.canRedo || busy} onClick={props.onRedo}><Icon name="reset" size={15} /></button>
      <span className="plan-scheme-sep" />
      <button className="icon-button" data-tip="New mission" aria-label="New mission" disabled={busy} onClick={() => { setNaming("new"); setName(`Op ${["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"][index.proposals.length % 5]}`); }}><Icon name="plus" size={15} /></button>
      <button className="icon-button" data-tip="Duplicate this mission" aria-label="Duplicate mission" disabled={busy || !proposal} onClick={() => proposal && props.onCreate(`${proposal.name} copy`, proposal.id)}><Icon name="file" size={15} /></button>
      <button className="icon-button" data-tip="Rename" aria-label="Rename mission" disabled={busy || !proposal} onClick={() => { setNaming("rename"); setName(proposal?.name ?? ""); }}><Icon name="settings" size={15} /></button>
      <button className={`icon-button${confirmDelete ? " danger-armed" : ""}`} data-tip={confirmDelete ? "Click again to delete" : "Delete mission"} aria-label="Delete mission" disabled={busy || !proposal} onClick={() => { if (confirmDelete) { props.onDeleteProposal(); setConfirmDelete(false); } else setConfirmDelete(true); }} onBlur={() => setConfirmDelete(false)}><Icon name="trash" size={15} /></button>
    </div>}
    {(naming || (index && !index.proposals.length)) && <form className="plan-name" onSubmit={(e) => { e.preventDefault(); if (!naming) setNaming("new"); submit(); }}>
      {!naming && <p className="insp-lead">Plan an operation on this scan: mark enemy posts, the approach route and the objective, check what can see you, then rehearse it in first person. Name the mission:</p>}
      <input aria-label="Mission name" placeholder="Op ALPHA" value={name} onChange={(e) => { if (!naming) setNaming("new"); setName(e.target.value); }} maxLength={80} />
      <button className="button primary small" type="submit" disabled={busy || !name.trim()}>{naming === "rename" ? "Rename" : "Create mission"}</button>
      {naming && <button className="text-button" type="button" onClick={() => { setNaming(null); setName(""); }}>Cancel</button>}
    </form>}
  </>;

  const markTask = <>
    <div className="plan-tools mission-tools">{TOOLS.map(({ tool: t, icon, label }) => <button key={t} className={`plan-tool${tool === t ? " active" : ""}`} aria-pressed={tool === t} disabled={locked} data-tip={TOOL_COPY[t].hint} onClick={() => props.onTool(tool === t ? null : t)}><Icon name={icon} size={18} /><span>{label}</span></button>)}</div>
    {tool === "symbol" && <div className="mission-sides">{SIDES.map((side) => <div key={side.label}><span>{side.label}</span>
      <div className="mission-roles" role="group" aria-label={`${side.label} symbols`}>{SYMBOL_ROLES.filter((r) => r.affiliation === side.affiliation).map((r) => <button key={r.role} className={`mission-role ${r.affiliation}${props.role === r.role ? " active" : ""}`} aria-pressed={props.role === r.role} onClick={() => props.onRole(r.role)}><i />{r.label === "HLZ" ? "Landing zone" : r.label}</button>)}</div></div>)}</div>}
    {!tool && <p className="plan-note"><b>Phase line</b>: a line you name (e.g. PL RED) so timings can say when it is crossed. <b>Line of sight</b>: click two points to see if one can see the other.</p>}
    {tool && <OnViewer><div className="plan-drawing" role="status"><p>{TOOL_COPY[tool].hint}</p>
      {tool === "symbol" && <div className="plan-drawing-actions"><span>Placing: {SYMBOL_ROLES.find((r) => r.role === props.role)?.label}</span><button className="button secondary small" onClick={() => props.onTool(null)}>Done</button></div>}
      {tool !== "symbol" && <div className="plan-drawing-actions"><span>{points} point{points === 1 ? "" : "s"}</span>
        {tool !== "los" && <><button className="text-button" disabled={!points} onClick={props.onUndoPoint}>Undo point</button>
          <button className="button primary small" disabled={points < TOOL_COPY[tool].min || busy} onClick={props.onFinish}><Icon name="check" size={13} />Finish</button></>}
        <button className="button secondary small" onClick={() => props.onTool(null)}>{tool === "los" ? "Done" : "Cancel"}</button></div>}
      {tool === "los" && props.los && <p className={`insp-verdict ${props.los.visible ? "bad" : "good"}`}><Icon name={props.los.visible ? "eye" : "shield"} size={16} />{props.los.visible ? `Can see: clear line over ${props.los.distance_m.toFixed(0)} m.` : `Cannot see: the line is blocked at ${props.los.blocked_at?.map((v) => v.toFixed(1)).join(", ")}.`}</p>}
    </div></OnViewer>}
  </>;

  const detail = feature && <div className="insp-detail-view">
    <button className="insp-back" onClick={() => props.onSelect(null)}><Icon name="left" size={14} />All plan items</button>
    <div className="twin-asset-head"><span className={`twin-kind mission-kind ${dot(feature)}`}><Icon name={feature.type === "symbol" ? "pin" : feature.type === "route" ? "route" : "ruler"} size={20} /></span>
      <div><input className="plan-title" aria-label="Name" defaultValue={feature.name} key={feature.id + feature.name} maxLength={120} disabled={locked}
        onBlur={(e) => { const v = e.target.value.trim(); if (v && v !== feature.name) props.onUpdate({ ...feature, name: v }); }}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} /><p>{feature.type === "symbol" ? `${SIDES.find((s) => s.affiliation === (feature.params as SymbolParams).affiliation)?.label ?? "Neutral"} symbol` : feature.type === "route" ? "Route" : "Phase line"}{feature.hidden ? " · hidden" : ""}</p></div></div>
    {feature.type === "symbol" && <SymbolForm feature={feature} routes={routes} disabled={locked} onUpdate={props.onUpdate} />}
    {feature.type === "route" && <RouteForm feature={feature} disabled={locked} onUpdate={props.onUpdate} />}
    {feature.type === "phase_line" && <div className="plan-form"><label className="plan-field"><span>Label</span>
      <input defaultValue={(feature.params as PhaseLineParams).label} key={feature.id + (feature.params as PhaseLineParams).label} maxLength={40} disabled={locked}
        onBlur={(e) => props.onUpdate({ ...feature, params: { ...(feature.params as PhaseLineParams), label: e.target.value.trim() || "PL" } })} /></label></div>}
    <p className="plan-note"><b>In 3D:</b> {feature.type === "symbol" ? "the orange square moves it, the blue dot turns where it faces." : "drag a waypoint, drag a + to add one, double-click one to remove it."}</p>
    <div className="insp-actions">
      <button className="button secondary small" disabled={locked} onClick={() => props.onUpdate({ ...feature, hidden: !feature.hidden })}><Icon name="eye" size={14} />{feature.hidden ? "Show" : "Hide"}</button>
      <button className="button danger small" disabled={locked} onClick={() => props.onDeleteFeature(feature.id)}><Icon name="trash" size={14} />Delete</button>
    </div>
  </div>;

  const itemsTask = detail || (proposal && proposal.features.length ? (["symbol", "route", "phase_line"] as const).map((type) => {
    const list = proposal.features.filter((f) => f.type === type);
    if (!list.length) return null;
    return <div key={type} className="plan-group"><h3>{TYPE_LABEL[type]} <small>{list.length}</small></h3>
      <ul className="plan-list">{list.map((f) => <li key={f.id}><button className={selected === f.id ? "active" : ""} aria-pressed={selected === f.id} onClick={() => props.onSelect(f.id)}>
        <span className={`mission-dot ${dot(f)}`} /><span>{f.name}</span>{f.type === "symbol" && (f.params as SymbolParams).affiliation === "hostile" && (f.params as SymbolParams).behaviour !== "sentry" ? <small>{(f.params as SymbolParams).behaviour}</small> : f.hidden ? <small>hidden</small> : <Icon name="chevron" size={13} />}
      </button></li>)}</ul></div>;
  }) : <div className="tool-empty"><Icon name="plus" size={26} /><p>Nothing marked yet.</p><button className="button secondary small" onClick={() => setTask("mark")}><Icon name="plus" size={13} />Start marking</button></div>);

  const exposureTask = <>
    {!hostiles && !routes.length ? <p className="insp-lead">Mark at least one enemy post and one route first.</p> : null}
    <button className="button primary full" disabled={props.analysisBusy || !hostiles && !routes.length} onClick={props.onAnalyse}>{props.analysisBusy ? <><span className="spinner" />Tracing sight lines…</> : <><Icon name="eye" size={15} />{analysis ? "Check again" : "Check what can see each route"}</>}</button>
    {props.analysisError && <p className="form-error" role="alert">{props.analysisError}</p>}
    {analysis && <div className="insp-result">
      {analysis.routes.length > 0 && <p className={`insp-verdict ${worst > 0 ? "bad" : "good"}`}><Icon name={worst > 0 ? "alert" : "shield"} size={16} />{worst > 0 ? `The most exposed route is seen for ${worst.toFixed(0)} s.` : "No route is seen by any marked enemy."}</p>}
      <div className="insp-kpis">
        <div><b>{analysis.ground.watched_m2.toLocaleString()} m²</b><small>ground the enemy can see</small></div>
        <div className="good"><b>{analysis.ground.dead_ground_m2.toLocaleString()} m²</b><small>hidden ground (dead ground)</small></div>
      </div>
      <label className="insp-show"><input type="checkbox" checked={props.showViewsheds} onChange={(e) => props.onViewsheds(e.target.checked)} />Show what the enemy sees on the model</label>
      {analysis.routes.map((r) => <div key={r.id} className="twin-block"><h4>{r.name}</h4><Report report={r.report} busy={busy} onCovered={() => props.onCoveredRoute(r.id)} /></div>)}
      {analysis.notes.length > 0 && <details className="ops-notes"><summary>How this was worked out</summary>{analysis.notes.map((n) => <p key={n} className="plan-note">{n}</p>)}</details>}
    </div>}
  </>;

  const hlzTask = <>
    <div className="insp-setting"><NumberField label="Clear circle needed" unit="m" value={diameter} min={5} max={200} onCommit={(v) => setDiameter(v ?? 25)} /></div>
    <button className="button primary full" onClick={() => props.onFindHlz(diameter)}><Icon name="target" size={15} />Find landing zones</button>
    {props.hlz && (props.hlz.length ? <div className="insp-result">
      <p className="insp-verdict good"><Icon name="check" size={16} />{props.hlz.length} place{props.hlz.length === 1 ? "" : "s"} flat and clear enough for a {diameter} m landing circle.</p>
      <ul className="plan-existing">{props.hlz.map((c, i) => <li key={i}><span><strong>Landing zone {i + 1}</strong><small>{c.usable_diameter_m.toFixed(0)} m clear · slope up to {c.max_slope_deg.toFixed(1)}° · nothing taller than {c.max_obstacle_m.toFixed(1)} m</small></span>
        <button className="button secondary small" disabled={locked} onClick={() => props.onAddHlz(c, diameter)}><Icon name="plus" size={13} />Add to plan</button></li>)}</ul></div>
      : <p className="insp-verdict warn"><Icon name="alert" size={16} />No flat, clear circle that size on measured ground. Try a smaller circle.</p>)}
  </>;

  const enemyTask = <>
    <div className="twin-block"><h4>Where would they watch from?</h4>
      <div className="insp-form">
        <label className="insp-field"><span>Your approach route</span><select value={threatRoute} onChange={(e) => setThreatRoute(e.target.value)}>
          <option value="">{routes.length ? "Choose a route" : "Draw a route first"}</option>
          {routes.filter((r) => (r.params as RouteParams).kind !== "patrol").map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}</select></label>
        <div className="insp-setting"><NumberField label="How far they can see" unit="m" value={threatRange} min={20} max={3000} step={25} onCommit={(v) => setThreatRange(v ?? 300)} /></div>
      </div>
      <button className="button primary full" disabled={!threatRoute || props.threatBusy} onClick={() => props.onThreat(threatRoute, threatRange)}>{props.threatBusy ? <><span className="spinner" />Scoring the ground…</> : <><Icon name="alert" size={15} />Find likely enemy positions</>}</button>
      {props.threat && <div className="insp-result">
        <label className="insp-show"><input type="checkbox" checked={props.showThreat} onChange={(e) => props.onShowThreat(e.target.checked)} />Show the heatmap on the model</label>
        <ul className="plan-existing">{props.threat.candidates.map((c, i) => <li key={i}><span><strong>{c.elevated ? "High spot (roof or hilltop)" : "Ground position"} · sees {c.overwatch_pct.toFixed(0)}% of the route</strong>
          <small>{c.height_above_route_m > 0 ? `${c.height_above_route_m.toFixed(0)} m above it` : "level with it"} · {c.concealment_pct.toFixed(0)}% hidden{c.mgrs ? ` · ${c.mgrs}` : ""}</small></span>
          <button className="button secondary small" disabled={locked} onClick={() => props.onAddThreat(c.position)}><Icon name="plus" size={13} />Add</button></li>)}</ul>
        <p className="plan-note">Added as <b>unknown</b>: a likely spot, not a sighting. {props.threat.basis}.</p>
      </div>}
    </div>
    <div className="twin-block"><h4>Seen in the video</h4>
      <button className="button secondary full" onClick={props.onFindCandidates}><Icon name="eye" size={15} />Find people and vehicles in the footage</button>
      {props.candidates && (props.candidates.length ? <ul className="plan-existing">{props.candidates.map((c, i) => <li key={i}><span><strong>{c.role === "vehicle" ? "Vehicle" : "Person"} · {c.source === "detections" ? "seen in the video" : "from scan labels"}</strong><small>{c.source === "detections" ? c.basis : `${c.points} labelled points`}</small></span>
        <button className="button secondary small" disabled={locked} onClick={() => props.onAddCandidate(c)}><Icon name="plus" size={13} />Add</button></li>)}</ul> : <p className="plan-note">No people or vehicles found in this scan.</p>)}
    </div>
  </>;

  const terrainTask = <>
    <div className="twin-block"><h4>Where can we move?</h4>
      <div className="plan-view" role="group" aria-label="Moving by">{([["foot", "On foot"], ["wheeled", "Wheels"], ["tracked", "Tracks"]] as const).map(([k, label]) => <button key={k} className={mobility === k ? "active" : ""} aria-pressed={mobility === k} onClick={() => setMobility(k)}>{label}</button>)}</div>
      <button className="button primary full" onClick={() => props.onTraffic(mobility)}><Icon name="route" size={15} />Map where we can move</button>
      {props.traffic && <div className="insp-result">
        <div className="insp-kpis"><div className="good"><b>{Math.round(props.traffic.area_m2.go).toLocaleString()} m²</b><small>free to move</small></div><div className="warn"><b>{Math.round(props.traffic.area_m2.slow_go).toLocaleString()} m²</b><small>slow going</small></div>
          <div className="bad"><b>{Math.round(props.traffic.area_m2.no_go).toLocaleString()} m²</b><small>cannot pass</small></div></div>
        <label className="insp-show"><input type="checkbox" checked={props.showTraffic} onChange={(e) => props.onShowTraffic(e.target.checked)} />Show on the model</label>
        <details className="ops-notes"><summary>How this was worked out</summary><p className="plan-note">{props.traffic.mobility}: {props.traffic.basis}</p></details>
      </div>}
    </div>
    <div className="twin-block"><h4>Tall obstacles</h4>
      <div className="insp-setting"><NumberField label="Taller than" unit="m" value={obstacleHeight} min={0.5} max={100} step={0.5} onCommit={(v) => setObstacleHeight(v ?? 3)} /></div>
      <button className="button secondary full" onClick={() => props.onObstacles(obstacleHeight)}><Icon name="list" size={15} />List obstacles</button>
      {props.obstacles && <>
        <p className="plan-note">{props.obstacles.count} taller than {props.obstacles.min_height_m} m{props.obstacles.count > 12 ? "; the 12 tallest:" : ":"}</p>
        <ul className="plan-existing">{props.obstacles.obstacles.slice(0, 12).map((o, i) => <li key={i}><span><strong>{o.height_m.toFixed(1)} m · {o.kind}</strong><small>{o.area_m2.toFixed(0)} m²{o.mgrs ? ` · ${o.mgrs}` : ""}</small></span></li>)}</ul>
      </>}
    </div>
  </>;

  const rehearseTask = <>
    <div className="insp-field"><span>Light</span><div className="plan-view" role="group" aria-label="Light">{(["day", "dusk", "night"] as const).map((l) => <button key={l} className={conditions.light === l ? "active" : ""} aria-pressed={conditions.light === l} onClick={() => props.onConditions({ ...conditions, light: l })}>{l[0].toUpperCase() + l.slice(1)}</button>)}</div></div>
    <div className="insp-form">
      <label className="insp-show"><input type="checkbox" checked={conditions.nvg} disabled={conditions.light === "day"} onChange={(e) => props.onConditions({ ...conditions, nvg: e.target.checked })} />Night-vision goggles</label>
      <div className="insp-setting"><NumberField label="Fog: can see up to" unit="m" value={conditions.fog_m ?? undefined} min={10} max={2000} step={10} onCommit={(v) => props.onConditions({ ...conditions, fog_m: v ?? null })} /></div>
    </div>
    <button className="button primary full mission-go" disabled={!ready || !hostiles && !routes.length} onClick={props.onRehearse}><Icon name="play" size={16} />Start the rehearsal</button>
    <p className="plan-note">Enemies stand where you marked them, with their sectors and behaviours. Night halves their sight; fog caps it. Every run is recorded.</p>
    <div className="twin-block"><h4>Past runs <small className="plan-count">{props.runs.length}</small></h4>
      {props.runs.length ? <ul className="mission-runs">{props.runs.map((r) => <li key={r.id}>
        <button onClick={() => props.onOpenRun(r.id)}><span className={`insp-status ${r.summary.completed ? "s-closed" : "s-open"}`}>{r.summary.completed ? "Reached" : "Not reached"}</span>
          <span><strong>{new Date(r.created_at).toLocaleString()}</strong><small>{fmtClock(r.summary.duration_s)} long · seen for {r.summary.exposure_s.toFixed(0)} s · {r.conditions.light ?? "day"}</small></span><Icon name="play" size={14} /></button>
        <button className="icon-button" aria-label="Delete run" data-tip="Delete this run" onClick={() => props.onDeleteRun(r.id)}><Icon name="trash" size={14} /></button></li>)}</ul>
        : <p className="plan-note">No rehearsals yet. Runs appear here with a replay you can scrub through.</p>}
    </div>
  </>;

  const briefTask = <div className="mission-brief">
    <button onClick={props.onSandTable}><Icon name="grid" size={20} /><span><b>Sand table</b><small>A top-down model for the briefing: symbols, sectors, routes, and the approach played back on its timings. Works full screen on a projector.</small></span></button>
    <button disabled={!hostiles && !routes.length} onClick={props.onInstructor}><Icon name="activity" size={20} /><span><b>Live exercise</b><small>Several people rehearse together on this machine or headsets on the network. The instructor watches live, moves enemy posts and sends messages.</small></span></button>
    <button disabled={busy} onClick={props.onPack}><Icon name="download" size={20} /><span><b>Mission pack for GPS</b><small>KMZ for Google Earth and GPX for handheld GPS. Needs a GPS-referenced scene. Everything is marked as planned, not observed.</small></span></button>
  </div>;

  const body: Record<Task, React.ReactNode> = { mark: markTask, items: itemsTask, exposure: exposureTask, hlz: hlzTask, enemy: enemyTask, terrain: terrainTask, rehearse: rehearseTask, brief: briefTask };

  return <div className="plan-panel mission-panel">
    <section className="inspector-section insp-top">
      {missionBar}
      {state && <div className="plan-status">
        <span className={`plan-frame-chip ${state.frame.status === "georeferenced" ? "is-geo" : ""}`} data-tip={state.frame.status === "georeferenced" ? "Bearings are true north; every symbol has a grid reference" : "No GPS fit: bearings are from the scene's own north"}><Icon name={state.frame.status === "georeferenced" ? "globe" : "info"} size={13} />{state.frame.status === "georeferenced" ? "True north" : "Scene north"}</span>
        {proposal && <span className="plan-frame-chip"><i className="mission-dot hostile" />{hostiles} enem{hostiles === 1 ? "y" : "ies"}</span>}
        {proposal && <span className="plan-frame-chip"><i className="mission-dot route" />{routes.length} route{routes.length === 1 ? "" : "s"}</span>}
        {proposal && <span className="plan-frame-chip"><i className="mission-dot friendly" />{objectives} objective{objectives === 1 ? "" : "s"}</span>}
      </div>}
      {proposal && <div className="insp-tasks plan-tasks" role="tablist" aria-label="What do you want to do?">{TASKS.map((t) => <button key={t.id} role="tab" aria-selected={task === t.id} className={task === t.id ? "on" : ""} onClick={() => { setTask(t.id); if (t.id !== "mark" && tool && tool !== "los") props.onTool(null); }}>
        <Icon name={t.icon} size={17} /><span>{t.label}</span>{t.id === "items" && proposal.features.length > 0 && <em>{proposal.features.length}</em>}{done[t.id] && <i className="insp-has" aria-label="Has a result" />}
      </button>)}</div>}
    </section>
    {proposal && <section className="inspector-section insp-body" key={task}>
      <header className="insp-head"><div><h3>{active.label}</h3><p>{active.what}</p></div></header>
      {message && <p className="plan-message" role="status">{message}</p>}
      {body[task]}
    </section>}

    {legend && !tool && <OnViewer kicker={`On the model: ${active.label}`}><div className="ops-hud"><div className="plan-legend">{legend.map(([c, l]) => <span key={l}><i style={{ background: c }} />{l}</span>)}</div></div></OnViewer>}
  </div>;
}
