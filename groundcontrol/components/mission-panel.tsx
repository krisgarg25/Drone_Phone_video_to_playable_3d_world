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
import "./plan-panel.css";

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
  { tool: "symbol", icon: "pin", label: "Symbol" }, { tool: "route", icon: "arrow", label: "Route" },
  { tool: "phase_line", icon: "ruler", label: "Phase line" }, { tool: "los", icon: "eye", label: "Line of sight" },
];
const TYPE_LABEL: Record<string, string> = { symbol: "Symbols", route: "Routes", phase_line: "Phase lines" };
const BEHAVIOUR_COPY: Record<Behaviour, string> = {
  sentry: "Holds the post and scans its sector", patrol: "Walks a patrol route", overwatch: "Holds, sees far through a narrow arc",
  reaction: "Moves to noise and last-seen positions",
};
const dot = (f: PlanFeature) => f.type === "symbol" ? (f.params as SymbolParams).affiliation : f.type === "route" ? "route" : "phase";

function SymbolForm({ feature, routes, disabled, onUpdate }: { feature: PlanFeature; routes: PlanFeature[]; disabled: boolean; onUpdate: (f: PlanFeature) => void }) {
  const p = feature.params as SymbolParams;
  const set = (patch: Partial<SymbolParams>) => onUpdate({ ...feature, params: { ...p, ...patch } });
  const armed = !["objective", "rally_point", "hlz", "obstacle", "checkpoint"].includes(p.role);
  return <div className="plan-form">
    <Select label="Role" value={p.role} disabled={disabled} options={SYMBOL_ROLES.map((r) => [r.role, r.label])} onChange={(role) => set({ role })} />
    <Select label="Affiliation" value={p.affiliation} disabled={disabled} options={[["hostile", "Hostile"], ["friendly", "Friendly"], ["unknown", "Unknown"], ["neutral", "Neutral"]]} onChange={(affiliation) => set({ affiliation })} />
    {armed && <>
      <NumberField label="Facing (bearing)" unit="°" value={Math.round(p.bearing_deg)} min={0} max={359} step={5} disabled={disabled} onCommit={(v) => set({ bearing_deg: v ?? 0 })} />
      <NumberField label="Sector of fire" unit="°" value={p.sector_deg} min={10} max={360} step={5} disabled={disabled} onCommit={(v) => set({ sector_deg: v ?? 90 })} />
      <NumberField label="Range" unit="m" value={p.range_m} min={5} max={2000} step={5} disabled={disabled} onCommit={(v) => set({ range_m: v ?? 120 })} />
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
    <NumberField label="Height above surface" unit="m" value={p.elevation_m} min={0} max={100} step={0.5} disabled={disabled} onCommit={(v) => set({ elevation_m: v ?? 0 })} />
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
    <div className="mission-kpis">
      <div><b>{report.length_m.toFixed(0)} m</b><small>length</small></div>
      <div><b>{fmtClock(report.eta_s)}</b><small>ETA (Tobler)</small></div>
      <div className={report.exposure_s > 0 ? "bad" : "good"}><b>{report.exposure_s.toFixed(0)} s</b><small>exposed</small></div>
      <div><b>{report.dead_ground_pct.toFixed(0)}%</b><small>in dead ground</small></div>
    </div>
    <div className="mission-profile" aria-label="Height profile and exposure along the route">
      {report.profile.station_m.map((s, i) => <i key={i} style={{ height: `${20 + 80 * ((report.profile.height_m[i] - Math.min(...report.profile.height_m)) / Math.max(0.5, Math.max(...report.profile.height_m) - Math.min(...report.profile.height_m)))}%` }} className={report.profile.seen_by[i] > 1 ? "hot2" : report.profile.seen_by[i] ? "hot" : ""} />)}
    </div>
    {report.per_hostile.length > 0 && <table className="plan-table"><thead><tr><th>Enemy</th><th>Exposed</th><th>First seen</th></tr></thead>
      <tbody>{report.per_hostile.map((h) => <tr key={h.id}><td>{h.name}<small>{h.closest_m.toFixed(0)} m closest</small></td><td className={h.exposure_s ? "up" : ""}>{h.exposure_s.toFixed(0)} s</td><td>{h.first_seen_m === null ? "never" : `${h.first_seen_m.toFixed(0)} m in`}</td></tr>)}</tbody></table>}
    <ol className="mission-timeline">{[...report.waypoints.map((w) => ({ t: w.time_s, text: w.name })), ...report.phase_lines.map((p) => ({ t: p.time_s, text: `crosses ${p.label}` }))]
      .sort((a, b) => a.t - b.t).map((e, i) => <li key={i}><span>{fmtClock(e.t)}</span>{e.text}</li>)}</ol>
    <p className="plan-note">Climb {report.climb_m.toFixed(0)} m · steepest {report.max_slope_pct.toFixed(0)}%{report.unscanned_m > 0 ? ` · ${report.unscanned_m.toFixed(0)} m over unscanned ground` : ""}.</p>
    {report.exposure_s > 0 && <button className="button secondary small full" disabled={busy} onClick={onCovered}><Icon name="arrow" size={13} />Suggest a covered route</button>}
  </div>;
}

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
  const proposal = state?.proposal ?? null;
  const feature = proposal?.features.find((f) => f.id === selected) ?? null;
  const locked = busy || !ready;
  const routes = proposal?.features.filter((f) => f.type === "route") ?? [];
  const hostiles = proposal?.features.filter((f) => f.type === "symbol" && (f.params as SymbolParams).affiliation === "hostile").length ?? 0;
  const submit = () => { const v = name.trim(); if (!v) return; if (naming === "rename") props.onRename(v); else props.onCreate(v); setNaming(null); setName(""); };

  return <div className="plan-panel mission-panel">
    <section className="inspector-section plan-head">
      <div className="section-label"><span>MISSION</span>
        <span className="plan-history">
          <button className="icon-button" title="Undo (Ctrl+Z)" aria-label="Undo" disabled={!props.canUndo || busy} onClick={props.onUndo}><Icon name="reset" size={14} /></button>
          <button className="icon-button plan-redo" title="Redo (Ctrl+Y)" aria-label="Redo" disabled={!props.canRedo || busy} onClick={props.onRedo}><Icon name="reset" size={14} /></button>
        </span></div>
      {index && index.proposals.length > 0 && !naming && <div className="plan-scheme">
        <select aria-label="Active mission" value={proposal?.id ?? ""} disabled={busy} onChange={(e) => props.onProposal(e.target.value)}>{index.proposals.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        <button className="icon-button" title="New mission" aria-label="New mission" disabled={busy} onClick={() => { setNaming("new"); setName(`Op ${["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"][index.proposals.length % 5]}`); }}><Icon name="plus" size={15} /></button>
        <button className="icon-button" title="Duplicate mission" aria-label="Duplicate mission" disabled={busy || !proposal} onClick={() => proposal && props.onCreate(`${proposal.name} copy`, proposal.id)}><Icon name="file" size={15} /></button>
        <button className="icon-button" title="Rename mission" aria-label="Rename mission" disabled={busy || !proposal} onClick={() => { setNaming("rename"); setName(proposal?.name ?? ""); }}><Icon name="settings" size={15} /></button>
        <button className={`icon-button${confirmDelete ? " danger-armed" : ""}`} title={confirmDelete ? "Click again to delete" : "Delete mission"} aria-label="Delete mission" disabled={busy || !proposal} onClick={() => { if (confirmDelete) { props.onDeleteProposal(); setConfirmDelete(false); } else setConfirmDelete(true); }} onBlur={() => setConfirmDelete(false)}><Icon name="trash" size={15} /></button>
      </div>}
      {(naming || (index && !index.proposals.length)) && <form className="plan-name" onSubmit={(e) => { e.preventDefault(); if (!naming) setNaming("new"); submit(); }}>
        {!naming && <p className="inspector-copy">Plan an operation on this scan: enemy posts with their sectors, the approach route, phase lines and the objective. Then rehearse it in first person and review the run.</p>}
        <input aria-label="Mission name" placeholder="Op ALPHA" value={name} onChange={(e) => { if (!naming) setNaming("new"); setName(e.target.value); }} maxLength={80} />
        <button className="button primary small" type="submit" disabled={busy || !name.trim()}>{naming === "rename" ? "Rename" : "Create mission"}</button>
        {naming && <button className="text-button" type="button" onClick={() => { setNaming(null); setName(""); }}>Cancel</button>}
      </form>}
      {state && <div className={`plan-frame plan-frame-${state.frame.status}`}><Icon name={state.frame.status === "georeferenced" ? "globe" : "info"} size={13} />
        <span>{state.frame.status === "georeferenced" ? "Georeferenced — bearings are true north, MGRS on every symbol" : "No GPS fit — bearings are from scene north (−Z)"}</span></div>}
    </section>

    {proposal && <>
      <section className="inspector-section">
        <div className="section-label"><span>DRAW</span><span>{proposal.features.length} FEATURES</span></div>
        <div className="plan-tools mission-tools">{TOOLS.map(({ tool: t, icon, label }) => <button key={t} className={`plan-tool${tool === t ? " active" : ""}`} aria-pressed={tool === t} disabled={locked} onClick={() => props.onTool(tool === t ? null : t)}><Icon name={icon} size={16} /><span>{label}</span></button>)}</div>
        {tool === "symbol" && <div className="mission-roles" role="group" aria-label="Symbol role">{SYMBOL_ROLES.map((r) => <button key={r.role} className={`mission-role ${r.affiliation}${props.role === r.role ? " active" : ""}`} aria-pressed={props.role === r.role} onClick={() => props.onRole(r.role)}><i />{r.label}</button>)}</div>}
        {tool && <div className="plan-drawing" role="status"><p>{TOOL_COPY[tool].hint}</p>
          {tool !== "symbol" && <div className="plan-drawing-actions"><span>{points} point{points === 1 ? "" : "s"}</span>
            {tool !== "los" && <><button className="text-button" disabled={!points} onClick={props.onUndoPoint}>Undo point</button>
              <button className="button primary small" disabled={points < TOOL_COPY[tool].min || busy} onClick={props.onFinish}><Icon name="check" size={13} />Finish</button></>}
            <button className="button secondary small" onClick={() => props.onTool(null)}>{tool === "los" ? "Done" : "Cancel"}</button></div>}
          {tool === "los" && props.los && <p className={`mission-los ${props.los.visible ? "bad" : "good"}`}>{props.los.visible ? `VISIBLE over ${props.los.distance_m.toFixed(0)} m` : `BLOCKED at ${props.los.blocked_at?.map((v) => v.toFixed(1)).join(", ")}`}</p>}
        </div>}
        {message && <p className="plan-message" role="status">{message}</p>}
      </section>

      {feature && <section className="inspector-section plan-selected">
        <div className="section-label"><span>{feature.type.replace("_", " ").toUpperCase()} · SELECTED</span><button className="icon-button" aria-label="Deselect" onClick={() => props.onSelect(null)}><Icon name="close" size={13} /></button></div>
        <input className="plan-title" aria-label="Feature name" defaultValue={feature.name} key={feature.id + feature.name} maxLength={120} disabled={locked}
          onBlur={(e) => { const v = e.target.value.trim(); if (v && v !== feature.name) props.onUpdate({ ...feature, name: v }); }}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
        {feature.type === "symbol" && <SymbolForm feature={feature} routes={routes} disabled={locked} onUpdate={props.onUpdate} />}
        {feature.type === "route" && <RouteForm feature={feature} disabled={locked} onUpdate={props.onUpdate} />}
        {feature.type === "phase_line" && <div className="plan-form"><label className="plan-field"><span>Label</span>
          <input defaultValue={(feature.params as PhaseLineParams).label} key={feature.id + (feature.params as PhaseLineParams).label} maxLength={40} disabled={locked}
            onBlur={(e) => props.onUpdate({ ...feature, params: { ...(feature.params as PhaseLineParams), label: e.target.value.trim() || "PL" } })} /></label></div>}
        <p className="plan-hint-3d">{feature.type === "symbol" ? "In 3D: the orange square moves it, the blue dot turns its facing." : "In 3D: drag a waypoint, drag a + to add one, double-click one to remove it."}</p>
        <div className="plan-feature-actions">
          <button className="button secondary small" disabled={locked} onClick={() => props.onUpdate({ ...feature, hidden: !feature.hidden })}><Icon name="eye" size={13} />{feature.hidden ? "Show" : "Hide"}</button>
          <button className="button danger small" disabled={locked} onClick={() => props.onDeleteFeature(feature.id)}><Icon name="trash" size={13} />Delete</button>
        </div>
      </section>}

      {proposal.features.length > 0 && <section className="inspector-section">
        <div className="section-label"><span>PLAN</span></div>
        {(["symbol", "route", "phase_line"] as const).map((type) => {
          const list = proposal.features.filter((f) => f.type === type);
          if (!list.length) return null;
          return <div key={type} className="plan-group"><h3>{TYPE_LABEL[type]} <small>{list.length}</small></h3>
            <ul className="plan-list">{list.map((f) => <li key={f.id}><button className={selected === f.id ? "active" : ""} aria-pressed={selected === f.id} onClick={() => props.onSelect(selected === f.id ? null : f.id)}>
              <span className={`mission-dot ${dot(f)}`} /><span>{f.name}</span>{f.type === "symbol" && <small>{(f.params as SymbolParams).behaviour !== "sentry" && (f.params as SymbolParams).affiliation === "hostile" ? (f.params as SymbolParams).behaviour : ""}</small>}{f.hidden && <small>hidden</small>}
            </button></li>)}</ul></div>;
        })}
      </section>}

      <section className="inspector-section">
        <div className="plan-shadow-head"><div className="section-label"><span>ANALYSIS</span></div>
          <label className="plan-switch"><input type="checkbox" checked={props.showViewsheds} onChange={(e) => props.onViewsheds(e.target.checked)} />Viewsheds</label></div>
        <button className="button primary small full" disabled={props.analysisBusy || !hostiles && !routes.length} onClick={props.onAnalyse}><Icon name="activity" size={13} />{props.analysisBusy ? "Analysing…" : analysis ? "Re-analyse" : "Analyse routes and sight"}</button>
        {props.analysisError && <p className="plan-message" role="alert">{props.analysisError}</p>}
        {analysis && <>
          {props.showViewsheds && analysis.hostiles > 0 && <div className="plan-legend"><span><i style={{ background: "rgb(250,214,90)" }} />Seen by 1</span><span><i style={{ background: "rgb(240,140,50)" }} />2</span><span><i style={{ background: "rgb(214,50,50)" }} />3+</span></div>}
          <div className="datum-row"><span>Ground watched</span><span>{analysis.ground.watched_m2.toLocaleString()} m²</span></div>
          <div className="datum-row"><span>Dead ground</span><span>{analysis.ground.dead_ground_m2.toLocaleString()} m²</span></div>
          {analysis.routes.map((r) => <div key={r.id} className="plan-group"><h3>{r.name}</h3><Report report={r.report} busy={busy} onCovered={() => props.onCoveredRoute(r.id)} /></div>)}
          {analysis.notes.map((n) => <p key={n} className="plan-note">{n}</p>)}
        </>}
      </section>

      <section className="inspector-section">
        <div className="section-label"><span>FIND</span></div>
        <div className="plan-export"><NumberField label="HLZ diameter" unit="m" value={diameter} min={5} max={200} onCommit={(v) => setDiameter(v ?? 25)} />
          <button className="button secondary small" onClick={() => props.onFindHlz(diameter)}>Find HLZs</button></div>
        {props.hlz && (props.hlz.length ? <ul className="plan-existing">{props.hlz.map((c, i) => <li key={i}><span><strong>HLZ {i + 1}</strong><small>{c.usable_diameter_m.toFixed(0)} m clear · slope ≤ {c.max_slope_deg.toFixed(1)}° · obstacles ≤ {c.max_obstacle_m.toFixed(1)} m</small></span>
          <button className="button secondary small" disabled={locked} onClick={() => props.onAddHlz(c, diameter)}>Add</button></li>)}</ul> : <p className="plan-note">No clear, flat circle that size on measured ground.</p>)}
        <div className="plan-group"><h3>Likely enemy positions <small>heuristic</small></h3>
          <div className="plan-shadow-grid">
            <Select label="Approach route" value={threatRoute} options={[["", routes.length ? "Choose a route" : "Draw a route first"], ...routes.filter((r) => (r.params as RouteParams).kind !== "patrol").map((r) => [r.id, r.name] as [string, string])]} onChange={setThreatRoute} />
            <NumberField label="Enemy sight range" unit="m" value={threatRange} min={20} max={3000} step={25} onCommit={(v) => setThreatRange(v ?? 300)} />
          </div>
          <div className="plan-shadow-head"><button className="button secondary small" disabled={!threatRoute || props.threatBusy} onClick={() => props.onThreat(threatRoute, threatRange)}>{props.threatBusy ? "Scoring ground…" : "Score ground"}</button>
            {props.threat && <label className="plan-switch"><input type="checkbox" checked={props.showThreat} onChange={(e) => props.onShowThreat(e.target.checked)} />Heatmap</label>}</div>
          {props.threat && <>
            {props.showThreat && <div className="plan-legend"><span><i style={{ background: "rgb(250,214,90)" }} />Possible</span><span><i style={{ background: "rgb(240,140,50)" }} />Likely</span><span><i style={{ background: "rgb(214,50,50)" }} />Most likely</span></div>}
            <ul className="plan-existing">{props.threat.candidates.map((c, i) => <li key={i}><span><strong>{c.elevated ? "Elevated (roof or crown)" : "Ground"} · sees {c.overwatch_pct.toFixed(0)}% of the route</strong>
              <small>{c.height_above_route_m > 0 ? `${c.height_above_route_m.toFixed(0)} m above it` : "level with it"} · cover {c.concealment_pct.toFixed(0)}%{c.mgrs ? ` · ${c.mgrs}` : ""}</small></span>
              <button className="button secondary small" disabled={locked} onClick={() => props.onAddThreat(c.position)}>Add</button></li>)}</ul>
            <p className="plan-note">{props.threat.basis}. Added as UNKNOWN for the planner to judge.</p>
          </>}
        </div>
        <div className="plan-group"><h3>Trafficability</h3>
          <div className="plan-export"><Select label="Mobility" value={mobility} options={[["foot", "On foot"], ["wheeled", "Wheeled"], ["tracked", "Tracked"]]} onChange={setMobility} />
            <button className="button secondary small" onClick={() => props.onTraffic(mobility)}>Map</button>
            {props.traffic && <label className="plan-switch"><input type="checkbox" checked={props.showTraffic} onChange={(e) => props.onShowTraffic(e.target.checked)} />On scan</label>}</div>
          {props.traffic && <>
            <div className="plan-legend"><span><i style={{ background: "rgb(70,190,90)" }} />GO {Math.round(props.traffic.area_m2.go).toLocaleString()} m²</span>
              <span><i style={{ background: "rgb(240,190,50)" }} />SLOW-GO {Math.round(props.traffic.area_m2.slow_go).toLocaleString()} m²</span>
              <span><i style={{ background: "rgb(210,60,60)" }} />NO-GO {Math.round(props.traffic.area_m2.no_go).toLocaleString()} m²</span></div>
            <p className="plan-note">{props.traffic.mobility}: {props.traffic.basis}</p>
          </>}
        </div>
        <div className="plan-group"><h3>Obstacles</h3>
          <div className="plan-export"><NumberField label="Taller than" unit="m" value={obstacleHeight} min={0.5} max={100} step={0.5} onCommit={(v) => setObstacleHeight(v ?? 3)} />
            <button className="button secondary small" onClick={() => props.onObstacles(obstacleHeight)}>List</button></div>
          {props.obstacles && <>
            <p className="plan-note">{props.obstacles.count} above {props.obstacles.min_height_m} m{props.obstacles.count > 12 ? " — the 12 tallest:" : ":"}</p>
            <ul className="plan-existing">{props.obstacles.obstacles.slice(0, 12).map((o, i) => <li key={i}><span><strong>{o.height_m.toFixed(1)} m · {o.kind}</strong><small>{o.area_m2.toFixed(0)} m²{o.mgrs ? ` · ${o.mgrs}` : ` · x ${o.position[0].toFixed(0)}, z ${o.position[1].toFixed(0)}`}</small></span></li>)}</ul>
            <p className="plan-note">{props.obstacles.basis}.</p>
          </>}
        </div>
        <button className="button secondary small full" onClick={props.onFindCandidates}>Suggest enemy positions from video detections and scan labels</button>
        {props.candidates && (props.candidates.length ? <ul className="plan-existing">{props.candidates.map((c, i) => <li key={i}><span><strong>{c.role === "vehicle" ? "Vehicle" : "Person"} · {c.source === "detections" ? "seen in video" : "scan labels"}</strong><small>{c.source === "detections" ? c.basis : `${c.points} labelled points`}</small></span>
          <button className="button secondary small" disabled={locked} onClick={() => props.onAddCandidate(c)}>Add</button></li>)}</ul> : <p className="plan-note">No people or vehicles in this scan&apos;s detections or labels.</p>)}
      </section>

      <section className="inspector-section">
        <div className="section-label"><span>REHEARSE</span></div>
        <div className="plan-view" role="group" aria-label="Light">{(["day", "dusk", "night"] as const).map((l) => <button key={l} className={conditions.light === l ? "active" : ""} aria-pressed={conditions.light === l} onClick={() => props.onConditions({ ...conditions, light: l })}>{l[0].toUpperCase() + l.slice(1)}</button>)}</div>
        <div className="plan-shadow-grid">
          <label className="plan-switch"><input type="checkbox" checked={conditions.nvg} disabled={conditions.light === "day"} onChange={(e) => props.onConditions({ ...conditions, nvg: e.target.checked })} />Night vision</label>
          <NumberField label="Visibility (fog)" unit="m" value={conditions.fog_m ?? undefined} min={10} max={2000} step={10} onCommit={(v) => props.onConditions({ ...conditions, fog_m: v ?? null })} />
        </div>
        <button className="button primary small full plan-run" disabled={!ready || !hostiles && !routes.length} onClick={props.onRehearse}><Icon name="play" size={13} />Rehearse in first person</button>
        <p className="plan-note">Enemies stand where the plan puts them with its sectors and behaviours; night halves their sight and fog caps it. The run is recorded for review.</p>
      </section>

      <section className="inspector-section">
        <div className="section-label"><span>RUNS</span><span>{props.runs.length}</span></div>
        {props.runs.length ? <ul className="plan-existing">{props.runs.map((r) => <li key={r.id}>
          <button className="text-button mission-run" onClick={() => props.onOpenRun(r.id)}><strong>{new Date(r.created_at).toLocaleString()}</strong>
            <small>{fmtClock(r.summary.duration_s)} · seen {r.summary.exposure_s.toFixed(0)} s · {r.summary.completed ? "objective reached" : "not completed"} · {r.conditions.light ?? "day"}</small></button>
          <button className="icon-button" aria-label="Delete run" onClick={() => props.onDeleteRun(r.id)}><Icon name="trash" size={13} /></button></li>)}</ul>
          : <p className="plan-note">No rehearsals yet.</p>}
      </section>

      <section className="inspector-section">
        <button className="button secondary small full" onClick={props.onSandTable}><Icon name="grid" size={14} />Sand table (briefing)</button>
        <button className="button secondary small full" disabled={!hostiles && !routes.length} onClick={props.onInstructor}><Icon name="activity" size={14} />Multi-user exercise (instructor)</button>
        <p className="plan-note">Several players rehearse the same mission together — on this machine or headsets on the LAN — while the instructor watches live, moves enemy posts and sends messages.</p>
        <p className="plan-note">Top-down miniature for the orders group: symbols, sectors, routes and phase lines, the approach played back on its timings. Fullscreen for a projector.</p>
        <button className="button secondary small full" disabled={busy} onClick={props.onPack}><Icon name="download" size={14} />Mission pack (KMZ + GPX)</button>
        <p className="plan-note">For Google Earth and handheld GPS. Needs a GPS-fitted scene; every item is marked planned, not observed.</p>
      </section>
    </>}
  </div>;
}
