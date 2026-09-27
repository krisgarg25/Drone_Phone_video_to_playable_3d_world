"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ATTRIBUTE_KEYS, pinMesh, twinApi, type Asset, type Epochs, type Inventory } from "@/lib/inspect";
import type { PlanMesh } from "@/lib/plan";
import { Icon } from "./studio-icons";
import "./plan-panel.css";
import "./ops-panel.css";

export type TwinView = "visual" | "measured" | "evidence";
type Props = {
  scene: string; ready: boolean; capabilities: Record<string, boolean>; view: TwinView;
  onView: (view: TwinView) => void; onOverlay: (meshes: PlanMesh[]) => void;
};
const KIND_COLOR: Record<Asset["kind"], [number, number, number]> = { building: [90, 160, 230], tree: [70, 180, 90], pole: [240, 200, 60] };
const VIEW_COPY: Record<TwinView, string> = {
  visual: "Photoreal splats: how the site looks. Not a measurement.",
  measured: "The measured collision surface: what distances, heights and volumes are taken on.",
  evidence: "Splats with observed coverage: where the flight saw the surface and where it did not.",
};
const fmt = (v: unknown) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) : Array.isArray(v) ? v.map((x) => Number(x).toFixed(1)).join(" × ") : v == null ? "—" : String(v));

export function TwinPanel({ scene, ready, capabilities, view, onView, onOverlay }: Props) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [epochs, setEpochs] = useState<Epochs | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState<"all" | Asset["kind"]>("all");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
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

  const assets = useMemo(() => (inventory?.assets ?? []).filter((a) => filter === "all" || a.kind === filter), [inventory, filter]);
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

  const choose = (a: Asset) => { setSelected(selected === a.id ? null : a.id); setDraft({ ...a.attributes }); };
  const save = () => current && (async () => {
    setBusy("save"); setError("");
    try {
      await twinApi.attributes(scene, current.id, Object.fromEntries(ATTRIBUTE_KEYS.map((k) => [k, draft[k] ?? ""])));
      setInventory(await twinApi.inventory(scene));
    } catch (cause) { fail(cause); } finally { setBusy(null); }
  })();
  const exportPackage = async () => {
    setBusy("package"); setError("");
    try { const r = await twinApi.package(scene, splats); setPkg({ url: r.url, bytes: r.bytes, files: r.manifest.files.length, triangles: r.manifest.triangles }); }
    catch (cause) { fail(cause); } finally { setBusy(null); }
  };
  const needs: Record<TwinView, string | null> = { visual: null, measured: capabilities.collider ? null : "no collision surface", evidence: capabilities.coverage ? null : "no coverage layer" };

  return <div className="plan-panel ops-panel twin-panel">
    <section className="inspector-section plan-head">
      <div className="section-label"><span>DIGITAL TWIN</span>{inventory && <span>{inventory.assets.length} ASSETS</span>}</div>
      <div className="plan-view" role="group" aria-label="Twin view">{(["visual", "measured", "evidence"] as TwinView[]).map((v) =>
        <button key={v} className={view === v ? "active" : ""} aria-pressed={view === v} disabled={!ready || !!needs[v]} title={needs[v] ?? undefined} onClick={() => onView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>)}</div>
      <p className="inspector-copy">{VIEW_COPY[view]} All three are in one frame, so a point picked in one is the same point in the others.</p>
      {error && <p className="plan-message" role="alert">{error}</p>}
    </section>

    <section className="inspector-section">
      <div className="plan-shadow-head"><div className="section-label"><span>ASSETS</span></div>
        <label className="plan-switch"><input type="checkbox" checked={showPins} onChange={(e) => setShowPins(e.target.checked)} />On scan</label></div>
      <div className="plan-view" role="group" aria-label="Asset kind">{(["all", "building", "tree", "pole"] as const).map((k) =>
        <button key={k} className={filter === k ? "active" : ""} aria-pressed={filter === k} onClick={() => setFilter(k)}>{k === "all" ? "All" : `${k[0].toUpperCase()}${k.slice(1)}s`}{k !== "all" && inventory ? ` ${inventory.counts[k] ?? 0}` : ""}</button>)}</div>
      {assets.length ? <ul className="plan-existing">{assets.map((a) => <li key={a.id} className={selected === a.id ? "active" : ""}>
        <button className="text-button insp-item" onClick={() => choose(a)}><span><strong><i className="insp-sev" style={{ background: `rgb(${KIND_COLOR[a.kind].join(",")})` }} />{a.attributes.name || a.attributes.asset_tag || a.id}</strong>
          <small>{a.kind} · {fmt(a.measured.height_m)} m{a.defects_open ? ` · ${a.defects_open} open defect${a.defects_open > 1 ? "s" : ""}` : ""}{a.mgrs ? ` · ${a.mgrs}` : ""}</small></span></button></li>)}</ul>
        : <p className="plan-note">{inventory ? "No assets of this kind in the scan's labels." : "Loading the inventory…"}</p>}
      {current && <div className="insp-detail">
        <h3 className="twin-card-title">{current.kind[0].toUpperCase() + current.kind.slice(1)} · {current.id}</h3>
        <div className="twin-facts">{Object.entries(current.measured).map(([k, v]) => <div className="datum-row" key={k}><span>{k.replace(/_/g, " ")}</span><span>{fmt(v)}</span></div>)}
          {current.worst_severity && <div className="datum-row"><span>worst open defect</span><span>severity {current.worst_severity}</span></div>}</div>
        <p className="plan-note">Measured on the scan ({current.basis}). The fields below are yours.</p>
        <div className="plan-form">{ATTRIBUTE_KEYS.map((k) => k === "notes"
          ? <label key={k} className="plan-field plan-field-wide"><span>Notes</span><textarea rows={2} value={draft[k] ?? ""} maxLength={2000} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} /></label>
          : <label key={k} className="plan-field"><span>{k.replace(/_/g, " ")}</span><input value={draft[k] ?? ""} maxLength={120} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} /></label>)}</div>
        <button className="button primary small full" disabled={busy === "save"} onClick={save}><Icon name="check" size={13} />{busy === "save" ? "Saving…" : "Save attributes"}</button>
      </div>}
      {inventory?.orphaned_attributes.length ? <p className="plan-message">Attributes kept for {inventory.orphaned_attributes.length} asset id(s) the current scan no longer produces: {inventory.orphaned_attributes.join(", ")}.</p> : null}
      {inventory && <p className="plan-note">{inventory.notes.join(" ")}</p>}
    </section>

    <section className="inspector-section">
      <div className="section-label"><span>EPOCHS</span>{epochs?.site && <span>{epochs.site}</span>}</div>
      {epochs?.note && <p className="plan-note">{epochs.note} Set a site id in Details to link repeat flights.</p>}
      <ol className="twin-timeline">{(epochs?.epochs ?? []).map((e) => <li key={e.scene} className={e.current ? "current" : ""}>
        <span className="twin-dot" /><div><strong>{e.name}</strong><small>{e.captured_at ? new Date(e.captured_at).toLocaleString() : "capture time not recorded"}{e.current ? " · this scan" : ""}</small>
          {e.change && <small>vs {e.change.against}: +{e.change.gain_m3.toFixed(0)} / −{e.change.loss_m3.toFixed(0)} m³ in {e.change.regions} regions</small>}
          {!e.current && <Link className="text-button" href={`/projects/${encodeURIComponent(e.scene)}/twin`}>Open</Link>}</div></li>)}</ol>
      <p className="plan-note">A new flight of the site becomes the next epoch; change against the one before is measured in Operations (surfaces) or Inspect (M3C2 on facades). Nothing is overwritten.</p>
    </section>

    <section className="inspector-section">
      <div className="section-label"><span>EXPORT FOR ENGINES</span></div>
      <label className="plan-switch"><input type="checkbox" checked={splats} onChange={(e) => setSplats(e.target.checked)} />Include splat chunks (large)</label>
      <button className="button primary small full" disabled={busy === "package"} onClick={() => void exportPackage()}><Icon name="download" size={13} />{busy === "package" ? "Packaging…" : "Build twin package (GLB, FBX, 3D Tiles)"}</button>
      {pkg && <p className="plan-note"><a className="text-button" href={pkg.url} download>Download twin-package.zip</a> · {(pkg.bytes / 1e6).toFixed(1)} MB, {pkg.files} files, {pkg.triangles.toLocaleString()} triangles.</p>}
      <p className="plan-note">surface.glb / .fbx and tiles/ are the measured top surface (2.5D, holes where unobserved); splats/ is the photoreal layer in 50 m chunks. FBX carries geometry only.</p>
    </section>
  </div>;
}
