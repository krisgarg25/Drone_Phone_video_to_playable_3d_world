"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ATTRIBUTE_KEYS, pinMesh, twinApi, type Asset, type Epochs, type Inventory } from "@/lib/inspect";
import type { PlanMesh } from "@/lib/plan";
import { Icon, type IconName } from "./studio-icons";
import { OnViewer } from "./on-viewer";
import "./plan-panel.css";
import "./ops-panel.css";
import "./inspect.css";

export type TwinView = "visual" | "measured" | "evidence";
type Props = {
  scene: string; ready: boolean; capabilities: Record<string, boolean>; view: TwinView;
  onView: (view: TwinView) => void; onOverlay: (meshes: PlanMesh[]) => void;
  onFocus?: (point: [number, number, number]) => void;
};
type Task = "assets" | "versions" | "export";
const KIND_COLOR: Record<Asset["kind"], [number, number, number]> = { building: [90, 160, 230], tree: [70, 180, 90], pole: [240, 200, 60] };
const KIND: Record<Asset["kind"], { label: string; plural: string; icon: IconName }> = {
  building: { label: "Building", plural: "Buildings", icon: "building" },
  tree: { label: "Tree", plural: "Trees", icon: "mountain" },
  pole: { label: "Pole", plural: "Poles", icon: "angle" },
};
const VIEWS: { id: TwinView; label: string; icon: IconName; what: string }[] = [
  { id: "visual", label: "As it looks", icon: "eye", what: "The photo-real model. Good for seeing, not for measuring." },
  { id: "measured", label: "As measured", icon: "ruler", what: "The solid surface every distance, height and volume is taken on." },
  { id: "evidence", label: "How sure", icon: "scan", what: "Where the drone saw the surface well, and where it did not." },
];
const TASKS: { id: Task; label: string; icon: IconName; what: string }[] = [
  { id: "assets", label: "Assets", icon: "list", what: "Every building, tree and pole found in the scan. Open one to see its measurements and add your own details." },
  { id: "versions", label: "Versions", icon: "history", what: "Each flight of this site is a version. Nothing is overwritten." },
  { id: "export", label: "Game engine", icon: "cube", what: "Package the model for Unity, Unreal or Cesium." },
];
const FIELD_LABEL: Record<string, string> = { name: "Name", asset_tag: "Asset tag", owner: "Owner", use: "Used for", material: "Material", condition: "Condition", built_year: "Year built", last_inspected: "Last inspected", notes: "Notes" };
const rgb = (c: [number, number, number]) => `rgb(${c.join(",")})`;
const fmt = (v: unknown) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) : Array.isArray(v) ? v.map((x) => Number(x).toFixed(1)).join(" × ") : v == null ? "—" : String(v));
const nice = (key: string) => { const t = key.replace(/_m2$/, " (m²)").replace(/_m$/, " (m)").replace(/_deg$/, " (°)").replace(/_/g, " "); return t[0].toUpperCase() + t.slice(1); };
const nameOf = (a: Asset) => a.attributes.name || a.attributes.asset_tag || `${KIND[a.kind].label} ${a.id.replace(/^existing-(building|vegetation|pole)-/, "")}`;

export function TwinPanel({ scene, ready, capabilities, view, onView, onOverlay, onFocus }: Props) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [epochs, setEpochs] = useState<Epochs | null>(null);
  const [task, setTask] = useState<Task>("assets");
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState<"all" | Asset["kind"]>("all");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [pkg, setPkg] = useState<{ url: string; bytes: number; files: number; triangles: number } | null>(null);
  const [splats, setSplats] = useState(true);
  const [showPins, setShowPins] = useState(true);
  const fail = (cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause));

  useEffect(() => {
    const timer = setTimeout(() => {
      twinApi.inventory(scene).then(setInventory).catch(fail);
      twinApi.epochs(scene).then(setEpochs).catch(fail);
    }, 0);
    return () => clearTimeout(timer);
  }, [scene]);

  const assets = useMemo(() => (inventory?.assets ?? []).filter((a) => (filter === "all" || a.kind === filter)
    && `${nameOf(a)} ${a.id} ${Object.values(a.attributes).join(" ")}`.toLowerCase().includes(search.toLowerCase())), [inventory, filter, search]);
  const current = inventory?.assets.find((a) => a.id === selected) ?? null;
  useEffect(() => {
    if (!showPins || !inventory) { onOverlay([]); return; }
    const meshes: PlanMesh[] = [];
    for (const kind of ["building", "tree", "pole"] as const) {
      const m = pinMesh(inventory.assets.filter((a) => a.kind === kind && (filter === "all" || filter === kind)).map((a) => a.viewer), KIND_COLOR[kind], `asset_${kind}`, kind === "building" ? 4 : 2.5);
      if (m) meshes.push(m);
    }
    if (current) { const m = pinMesh([current.viewer], [230, 60, 180], "asset_selected", 7); if (m) meshes.push(m); }
    onOverlay(meshes);
  }, [inventory, filter, current, showPins, onOverlay]);

  const open = (a: Asset) => { setSelected(a.id); setDraft({ ...a.attributes }); setSaved(false); onFocus?.(a.viewer); };
  const save = () => current && (async () => {
    setBusy("save"); setError(""); setSaved(false);
    try {
      await twinApi.attributes(scene, current.id, Object.fromEntries(ATTRIBUTE_KEYS.map((k) => [k, draft[k] ?? ""])));
      setInventory(await twinApi.inventory(scene)); setSaved(true);
    } catch (cause) { fail(cause); } finally { setBusy(null); }
  })();
  const exportPackage = async () => {
    setBusy("package"); setError("");
    try { const r = await twinApi.package(scene, splats); setPkg({ url: r.url, bytes: r.bytes, files: r.manifest.files.length, triangles: r.manifest.triangles }); }
    catch (cause) { fail(cause); } finally { setBusy(null); }
  };
  const needs: Record<TwinView, string | null> = { visual: null, measured: capabilities.collider ? null : "This scan has no solid surface yet", evidence: capabilities.coverage ? null : "This scan has no coverage layer yet" };
  const active = TASKS.find((t) => t.id === task)!;
  const counts = inventory?.counts ?? {};

  const assetsTask = current ? <div className="insp-detail-view">
    <button className="insp-back" onClick={() => setSelected(null)}><Icon name="left" size={14} />All assets</button>
    <div className="twin-asset-head"><span className="twin-kind" style={{ ["--kind" as string]: rgb(KIND_COLOR[current.kind]) }}><Icon name={KIND[current.kind].icon} size={20} /></span>
      <div><h3>{nameOf(current)}</h3><p>{KIND[current.kind].label}{current.mgrs ? <> · <code>{current.mgrs}</code></> : null}</p></div>
      {onFocus && <button className="button secondary small" onClick={() => onFocus(current.viewer)}><Icon name="target" size={14} />Go to</button>}</div>
    {current.defects_open > 0 && <p className="insp-verdict warn"><Icon name="alert" size={16} />{current.defects_open} open defect{current.defects_open > 1 ? "s" : ""} logged in Inspect{current.worst_severity ? `, worst severity ${current.worst_severity} of 5` : ""}.</p>}
    <div className="twin-block"><h4>Measured from the scan</h4>
      <div className="insp-kpis">{Object.entries(current.measured).filter(([, v]) => v !== null).slice(0, 6).map(([k, v]) => <div key={k}><b>{fmt(v)}</b><small>{nice(k)}</small></div>)}</div>
      <p className="plan-note">{current.basis}</p></div>
    <div className="twin-block"><h4>Your details</h4>
      <div className="twin-fields">{ATTRIBUTE_KEYS.map((k) => <label key={k} className={`insp-field${k === "notes" ? " wide" : ""}`}><span>{FIELD_LABEL[k] ?? nice(k)}</span>
        {k === "notes" ? <textarea rows={2} value={draft[k] ?? ""} maxLength={2000} onChange={(e) => { setDraft({ ...draft, [k]: e.target.value }); setSaved(false); }} />
          : <input value={draft[k] ?? ""} maxLength={120} onChange={(e) => { setDraft({ ...draft, [k]: e.target.value }); setSaved(false); }} />}</label>)}</div>
      <div className="insp-actions"><button className="button primary full" disabled={busy === "save"} onClick={save}><Icon name="check" size={15} />{busy === "save" ? "Saving…" : saved ? "Saved" : "Save details"}</button></div>
    </div>
  </div> : <>
    <div className="insp-chips twin-kinds">{(["all", "building", "tree", "pole"] as const).map((k) => <button key={k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>
      {k !== "all" && <i style={{ background: rgb(KIND_COLOR[k]) }} />}{k === "all" ? "All" : KIND[k].plural}<em>{k === "all" ? inventory?.assets.length ?? 0 : counts[k] ?? 0}</em></button>)}</div>
    <label className="search-field twin-search"><Icon name="search" size={15} /><input placeholder="Search by name, tag or owner" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search assets" /></label>
    {!inventory ? <p className="insp-busy"><span className="spinner" />Loading the asset list…</p>
      : assets.length ? <ul className="twin-list">{assets.slice(0, 200).map((a) => <li key={a.id}><button onClick={() => open(a)}>
        <i style={{ background: rgb(KIND_COLOR[a.kind]) }} />
        <span><strong>{nameOf(a)}</strong><small>{KIND[a.kind].label}{typeof a.measured.height_m === "number" ? ` · ${fmt(a.measured.height_m)} m tall` : ""}{a.attributes.owner ? ` · ${a.attributes.owner}` : ""}</small></span>
        {a.defects_open > 0 && <em className="insp-status s-open">{a.defects_open} defect{a.defects_open > 1 ? "s" : ""}</em>}
        <Icon name="chevron" size={14} />
      </button></li>)}</ul>
      : <div className="tool-empty"><Icon name="search" size={26} /><p>{search ? "Nothing matches." : "No assets of this kind in the scan."}</p></div>}
    {assets.length > 200 && <p className="plan-note">Showing the first 200 of {assets.length}. Search to narrow down.</p>}
    {inventory?.orphaned_attributes.length ? <p className="plan-message">Details kept for {inventory.orphaned_attributes.length} asset(s) this scan no longer finds: {inventory.orphaned_attributes.join(", ")}.</p> : null}
    {inventory?.notes.length ? <details className="ops-notes"><summary>How assets are found</summary><p className="plan-note">{inventory.notes.join(" ")}</p></details> : null}
  </>;

  const versionsTask = <>
    {epochs?.note && <p className="insp-lead" title={epochs.note}>This scan is not linked to other flights yet. Give repeat flights the same <b>site id</b> under Explore → Project and they appear here as versions.</p>}
    <ol className="twin-timeline">{(epochs?.epochs ?? []).map((e) => <li key={e.scene} className={e.current ? "current" : ""}>
      <span className="twin-dot" /><div><strong>{e.name}{e.current && <em className="twin-now">This version</em>}</strong><small>{e.captured_at ? new Date(e.captured_at).toLocaleString() : "Capture time not recorded"}</small>
        {e.change && <small className="twin-change">Since {e.change.against}: <b className="up">+{e.change.gain_m3.toFixed(0)} m³</b> added, <b className="down">−{e.change.loss_m3.toFixed(0)} m³</b> removed in {e.change.regions} area{e.change.regions === 1 ? "" : "s"}</small>}
        {!e.current && <Link className="button secondary small" href={`/projects/${encodeURIComponent(e.scene)}/twin`}><Icon name="arrow" size={13} />Open this version</Link>}</div></li>)}</ol>
    {!epochs && <p className="insp-busy"><span className="spinner" />Loading versions…</p>}
    <p className="plan-note">To see exactly what changed between versions, use Operations → What changed (ground and buildings) or Inspect → What moved (walls and facades).</p>
  </>;

  const exportTask = <>
    <div className="twin-pack">
      <div><Icon name="cube" size={18} /><span><b>Solid surface</b>GLB and FBX, plus 3D Tiles for Cesium</span></div>
      <label className="twin-pack-opt"><input type="checkbox" checked={splats} onChange={(e) => setSplats(e.target.checked)} /><Icon name="eye" size={18} /><span><b>Photo-real layer</b>Splats in 50 m chunks. Much larger download.</span></label>
    </div>
    <button className="button primary full" disabled={!!busy} onClick={() => void exportPackage()}>{busy === "package" ? <><span className="spinner" />Building the package…</> : <><Icon name="download" size={15} />Build the package</>}</button>
    {pkg && <div className="insp-result">
      <p className="insp-verdict good"><Icon name="check" size={16} />Ready: {(pkg.bytes / 1e6).toFixed(1)} MB, {pkg.files} files, {pkg.triangles.toLocaleString()} triangles.</p>
      <a className="button secondary full" href={pkg.url} download><Icon name="download" size={15} />Download twin-package.zip</a>
    </div>}
    <p className="plan-note">The surface is the measured top surface, with holes where the drone saw nothing. FBX carries geometry only. For every other format, use the Export page.</p>
  </>;

  return <div className="plan-panel ops-panel twin-panel">
    <section className="inspector-section insp-top">
      <div className="twin-views" role="radiogroup" aria-label="See the site">{VIEWS.map((v) => <button key={v.id} role="radio" aria-checked={view === v.id} className={view === v.id ? "on" : ""} disabled={!ready || !!needs[v.id]} data-tip={!ready ? "Waiting for the 3D view" : needs[v.id] ?? v.what} onClick={() => onView(v.id)}>
        <Icon name={v.icon} size={18} /><b>{v.label}</b></button>)}</div>
      <p className="twin-view-note"><Icon name="info" size={14} />{VIEWS.find((v) => v.id === view)?.what} All three views share one coordinate frame.</p>
      <div className="insp-tasks twin-tasks" role="tablist" aria-label="What do you want to do?">{TASKS.map((t) => <button key={t.id} role="tab" aria-selected={task === t.id} className={task === t.id ? "on" : ""} onClick={() => setTask(t.id)}>
        <Icon name={t.icon} size={18} /><span>{t.label}</span>{t.id === "assets" && inventory ? <em>{inventory.assets.length}</em> : null}</button>)}</div>
    </section>
    <section className="inspector-section insp-body" key={task}>
      <header className="insp-head"><div><h3>{active.label}</h3><p>{active.what}</p></div>{task === "assets" && <label className="insp-show"><input type="checkbox" checked={showPins} onChange={(e) => setShowPins(e.target.checked)} />Show on model</label>}</header>
      {error && <p className="form-error" role="alert">{error}</p>}
      {task === "assets" ? assetsTask : task === "versions" ? versionsTask : exportTask}
    </section>

    {showPins && inventory && task === "assets" && <OnViewer kicker="On the model: assets">
      <div className="ops-hud"><div className="plan-legend">{(["building", "tree", "pole"] as const).filter((k) => counts[k]).map((k) => <span key={k}><i style={{ background: rgb(KIND_COLOR[k]) }} />{KIND[k].plural} {counts[k]}</span>)}{current && <span><i style={{ background: "rgb(230,60,180)" }} />Selected</span>}</div>
        <button className="button ghost small" onClick={() => setShowPins(false)}><Icon name="eye" size={14} />Hide</button></div>
    </OnViewer>}
  </div>;
}
