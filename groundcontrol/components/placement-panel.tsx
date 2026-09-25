"use client";

import { type Placement, type ProjectDetail } from "@/lib/workspace";
import { Icon } from "./studio-icons";

function fitBadge(p: Placement) {
  const fit = p.fit;
  if (!fit) return { label: "—", tone: "unknown" };
  if (!fit.valid) return { label: fit.reason || "does not fit", tone: "bad" };
  if (fit.tight) return { label: `fits · ${fit.floor_clearance_m ?? "?"} m from wall`, tone: "good" };
  return { label: "fits", tone: "good" };
}

export function PlacementPanel({ project, ready, place, moving, onPick, onYaw, onClear, onMove, onRotate, onDelete, onSelect, onExport, onView }: {
  project: ProjectDetail; ready: boolean;
  place: { item: string; yaw: number } | null; moving: string | null;
  onPick: (item: string) => void; onYaw: (delta: number) => void; onClear: () => void;
  onMove: (placement: Placement) => void;
  onRotate: (placement: Placement, yaw: number) => void; onDelete: (id: string) => void;
  onSelect: (id: string | null) => void; onExport: () => void; onView: (value: "fit" | "top") => void;
}) {
  const library = project.furniture || [];
  const placements = project.placements || [];
  const selected = place ? library.find((f) => f.item === place.item) : null;
  const totalArea = placements.reduce((sum, p) => sum + (p.footprint_m2 || 0), 0);
  return <>
    <section className="inspector-section"><h2>Place furniture</h2>
      <div className="view-toggle"><button className="button secondary small" onClick={() => onView("top")}><Icon name="layers" size={14} />Plan view</button><button className="button secondary small" onClick={() => onView("fit")}><Icon name="compass" size={14} />Reset view</button></div>
      <p className="inspector-copy">Pick an item, then click the scanned floor to drop it at real-world size. Each drop is snapped to the measured floor and checked against the walls and ceiling.</p>
      {library.length === 0 ? <p className="form-error">The furniture library is unavailable — the backend needs SciPy to read the floor grid.</p> : <div className="furniture-grid">{library.map((f) => <button key={f.item} className={`furniture-chip ${place?.item === f.item ? "active" : ""}`} disabled={!ready} onClick={() => onPick(f.item)}><strong>{f.label}</strong><small>{f.size[0].toLocaleString()}×{f.size[2].toLocaleString()}×{f.size[1].toLocaleString()} m</small></button>)}</div>}
      {selected && <div className="place-draft"><p>Placing <strong>{selected.label}</strong>. Click the floor to drop; drop several to lay out a room.</p><div className="rotate-row"><span>Rotation</span><button className="button secondary small" onClick={() => onYaw(-45)} aria-label="Rotate left">↺ 45°</button><strong>{(place?.yaw ?? 0) % 360}°</strong><button className="button secondary small" onClick={() => onYaw(45)} aria-label="Rotate right">45° ↻</button><button className="text-button" onClick={onClear}>Stop placing</button></div></div>}
      <div className="notice warning"><Icon name="info" /><p>Dimensions are nominal catalogue sizes, and the fit check reads the scan&apos;s coverage grid — a planning aid, not surveyed proof an item clears a doorway. {project.scale.status === "metric" ? "Scale is metric (pose-referenced)." : "Scale is not independently metric, so placements inherit that uncertainty."}</p></div>
    </section>
    <section className="inspector-section"><div className="section-label"><span>PLACED LAYOUT</span><span>{placements.length}{placements.length > 0 ? ` · ${totalArea.toFixed(1)} m²` : ""}</span></div>
      {placements.length === 0 ? <p className="inspector-copy">Nothing placed yet. Choose an item above and click the floor.</p> : <>
        {moving && <div className="place-draft"><p>Move mode: click a new spot on the floor to relocate the selected item.</p><button className="text-button" onClick={onClear}>Cancel move</button></div>}
        <ul className="place-list">{placements.map((p) => { const badge = fitBadge(p); return <li key={p.id} className={moving === p.id ? "moving" : ""} onMouseEnter={() => onSelect(p.id)} onMouseLeave={() => onSelect(null)}>
          <div className="place-row"><div><strong>{p.label}</strong><small>{p.size[0].toLocaleString()}×{p.size[2].toLocaleString()} m · {p.yaw_deg}°</small></div><span className={`fit fit-${badge.tone}`}>{badge.label}</span></div>
          {p.stale && <small className="stale">Previous model version · re-place before use</small>}
          <div className="place-actions"><button className={`button secondary small ${moving === p.id ? "active" : ""}`} onClick={() => onMove(p)} disabled={!ready}>Move</button><button className="button secondary small" onClick={() => onRotate(p, (p.yaw_deg + 45) % 360)} disabled={!ready}>Rotate 45°</button><button className="icon-button" aria-label={`Delete ${p.label}`} onClick={() => onDelete(p.id)} disabled={!ready}><Icon name="trash" size={14} /></button></div>
        </li>; })}</ul>
        <button className="button secondary small full" onClick={onExport}><Icon name="download" size={14} />Download layout (GeoJSON)</button>
      </>}
    </section>
  </>;
}
