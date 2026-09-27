"use client";

import { useState } from "react";
import {
  LIFT_STEP_M, MOVE_STEP_M, RESIZE_STEP_M, SIZE_MAX_M, SIZE_MIN_M,
  type FurnitureItem, type ImportedModel, type Placement, type ProjectDetail,
} from "@/lib/workspace";
import { Icon } from "./studio-icons";
import { FurnitureSymbol } from "./furniture-symbol";

type EditorProps = {
  project: ProjectDetail; placements: Placement[]; selected: string | null;
  place: { item: string; yaw: number } | null; saveState: "clean" | "queued" | "saving" | "error";
  offline: boolean; message: string; status: string; importing: boolean;
  onPickItem: (item: string) => void; onYawArm: (delta: number) => void; onCancelPlace: () => void;
  onRetrySave: () => void; onSelect: (id: string | null) => void;
  onSlide: (placement: Placement, dx: number, dz: number) => void;
  onLift: (placement: Placement, dy: number) => void;
  onYaw: (placement: Placement, delta: number) => void;
  onResize: (placement: Placement, axis: 0 | 1 | 2, delta: number) => void;
  onDuplicate: (placement: Placement) => void; onDelete: (id: string) => void;
  onExport: () => void; onFit: () => void; onImportModel: (file: File) => void;
};
const AXIS: { key: 0 | 1 | 2; label: string }[] = [{ key: 0, label: "length" }, { key: 1, label: "height" }, { key: 2, label: "depth" }];
const metres = (value: number) => `${value.toFixed(2)} m`;
/** Honest caption for an imported item: its file-read size, plus the scene's scale caveat. */
function modelNote(model: ImportedModel) {
  if (model.scale_status === "metric") return `${model.true_size_note}; scene scale is metric.`;
  if (model.scale_status === "estimated") return `${model.true_size_note}, but this scene's scale is estimated (${model.scale_source}), not measured — its on-scene size is only as accurate as that scale.`;
  return `${model.true_size_note}, but this scene has only relative scale (${model.scale_source}) — treat its size as nominal, not measured.`;
}

/** Slim catalogue strip plus the floating dock that edits the selected piece in 3D. */
export function PlacementPanel({
  project, placements, selected, place, saveState, offline, message, status, importing,
  onPickItem, onYawArm, onCancelPlace, onRetrySave, onSelect, onSlide, onLift, onYaw, onResize, onDuplicate, onDelete, onExport, onFit, onImportModel,
}: EditorProps) {
  const [search, setSearch] = useState("");
  const [confirming, setConfirming] = useState<string | null>(null);
  const active = placements.find((p) => p.id === selected) ?? null;
  const arming: FurnitureItem | undefined = project.furniture.find((f) => f.item === place?.item);
  const library = project.furniture.filter((f) => f.label.toLowerCase().includes(search.toLowerCase()));
  const locked = offline;
  return <>
    <div className="editor-strip">
      <div className="strip-row">
        <div className="strip-lead"><span className="eyebrow">3D LAYOUT STUDIO</span><h2>Placement editor</h2></div>
        <label className="search-field furniture-search"><Icon name="search" size={15} /><input aria-label="Search furniture" placeholder="Find furniture…" value={search} onChange={(e) => setSearch(e.target.value)} /></label>
        <div className="furniture-grid">{library.map((f) => <button key={f.item} className={`furniture-chip ${place?.item === f.item ? "active" : ""}`} aria-pressed={place?.item === f.item} disabled={locked} onClick={() => onPickItem(f.item)}><FurnitureSymbol item={f.item} /><strong>{f.label}</strong><small>{f.imported ? "real model · " : ""}{f.size[0]} × {f.size[2]} m</small></button>)}</div>
        {!library.length && <p className="strip-note">{project.furniture.length ? "No matching furniture — try another name." : "Catalogue unavailable. Check the backend connection."}</p>}
        <div className="strip-actions">
          <label className={`button secondary small import-model${importing || locked ? " is-busy" : ""}`} title="Import a self-contained .glb or .gltf as a placeable item; its real size is read from the file">
            <Icon name="cube" size={14} />{importing ? "Importing…" : "Import model"}
            <input type="file" accept=".glb,.gltf,model/gltf-binary,model/gltf+json" disabled={importing || locked} aria-label="Import glTF model" onChange={(e) => { const file = e.currentTarget.files?.[0]; if (file) onImportModel(file); e.currentTarget.value = ""; }} />
          </label>
          <button className="button secondary small" onClick={onFit}><Icon name="expand" size={14} />Fit view</button>{placements.length > 0 && <button className="button secondary small" onClick={onExport}><Icon name="download" size={14} />Layout</button>}
        </div>
      </div>
      <div className="strip-row">
        {!placements.length ? <p className="strip-note">Nothing placed yet. Pick a piece above, then click the scanned floor in the 3D view.</p>
          : <ul className="place-list">{placements.map((p) => <li key={p.id} className={selected === p.id ? "moving" : ""}>
            <button className="placed-item-select" aria-label={`Select ${p.label}`} aria-pressed={selected === p.id} onClick={() => onSelect(selected === p.id ? null : p.id)}>
              <FurnitureSymbol item={p.item} /><span><strong>{p.label}</strong><small>{p.model ? "real model · " : ""}{metres(p.size[0])} × {metres(p.size[2])} · {p.yaw_deg}°</small></span><span className={`support-dot ${p.fit?.supported ? "supported" : ""}`} title={p.fit?.supported ? "Floor supported" : "Check floor support"} />
            </button>
          </li>)}</ul>}
        {arming && place && <div className="place-draft"><strong>Placing {arming.label}</strong><div className="rotate-row"><button className="button secondary small" disabled={locked} onClick={() => onYawArm(-45)} aria-label="Rotate new item left">−45°</button><b>{place.yaw}°</b><button className="button secondary small" disabled={locked} onClick={() => onYawArm(45)} aria-label="Rotate new item right">+45°</button><button className="text-button" onClick={onCancelPlace}>Stop placing</button></div></div>}
        <p className="editor-status" role="status">{(placements.length || saveState !== "clean") && <span className={`save-state save-${saveState}`}>{status}</span>}{message && <span className="editor-message">{message}</span>}{saveState === "error" && <button className="text-button" onClick={onRetrySave}>Retry save</button>}</p>
      </div>
    </div>

    {active ? <aside className="editor-dock" aria-label="Selected piece controls">
      <div className="dock-head">
        <FurnitureSymbol item={active.item} />
        <div><strong>{active.label}</strong><small title="Position and heading in the local viewer frame">{`${metres(active.center_xz[0])}, ${metres(active.center_xz[1])} · up ${metres(active.center_y)} · ${active.yaw_deg}°`}</small></div>
        <button className="icon-button" aria-label="Deselect furniture" onClick={() => onSelect(null)}><Icon name="close" size={14} /></button>
      </div>
      {active.model && <p className="dock-model-note">{modelNote(active.model)}</p>}
      <div className="dock-group" role="group" aria-label="Move and lift">
        <div className="pad">
          <span />
          <button aria-label="Move forward" title={`Forward ${MOVE_STEP_M} m`} disabled={locked} onClick={() => onSlide(active, 0, -MOVE_STEP_M)}>↑</button>
          <span />
          <button aria-label="Move left" title={`Left ${MOVE_STEP_M} m`} disabled={locked} onClick={() => onSlide(active, -MOVE_STEP_M, 0)}>←</button>
          <b className="pad-step">{MOVE_STEP_M} m</b>
          <button aria-label="Move right" title={`Right ${MOVE_STEP_M} m`} disabled={locked} onClick={() => onSlide(active, MOVE_STEP_M, 0)}>→</button>
          <span />
          <button aria-label="Move back" title={`Back ${MOVE_STEP_M} m`} disabled={locked} onClick={() => onSlide(active, 0, MOVE_STEP_M)}>↓</button>
          <span />
        </div>
        <div className="lift">
          <button aria-label="Raise piece" title={`Up ${LIFT_STEP_M} m above the floor`} disabled={locked} onClick={() => onLift(active, LIFT_STEP_M)}>▲ {LIFT_STEP_M}</button>
          <button aria-label="Lower piece" title={`Down ${LIFT_STEP_M} m`} disabled={locked} onClick={() => onLift(active, -LIFT_STEP_M)}>▼ {LIFT_STEP_M}</button>
        </div>
      </div>
      <div className="dock-group" role="group" aria-label="Rotate">
        <button aria-label="Rotate left 45 degrees" disabled={locked} onClick={() => onYaw(active, -45)}>−45°</button>
        <button aria-label="Rotate left 15 degrees" disabled={locked} onClick={() => onYaw(active, -15)}>−15°</button>
        <b>{active.yaw_deg}°</b>
        <button aria-label="Rotate right 15 degrees" disabled={locked} onClick={() => onYaw(active, 15)}>+15°</button>
        <button aria-label="Rotate right 45 degrees" disabled={locked} onClick={() => onYaw(active, 45)}>+45°</button>
      </div>
      <div className="dock-group dock-resize" role="group" aria-label="Resize">
        {AXIS.map(({ key, label }) => <div className="axis" key={label}>
          <span>{label}</span>
          <button aria-label={`Decrease ${label}`} disabled={locked || active.size[key] <= SIZE_MIN_M} onClick={() => onResize(active, key, -RESIZE_STEP_M)}>−</button>
          <b title={`${SIZE_MIN_M}–${SIZE_MAX_M} m`}>{metres(active.size[key])}</b>
          <button aria-label={`Increase ${label}`} disabled={locked || active.size[key] >= SIZE_MAX_M} onClick={() => onResize(active, key, RESIZE_STEP_M)}>+</button>
        </div>)}
      </div>
      <div className="dock-group dock-foot">
        <button className="button secondary small" disabled={locked} onClick={() => onDuplicate(active)}><Icon name="cube" size={14} />Duplicate</button>
        {confirming === active.id ? <span className="draft-actions"><button className="button danger small" aria-label="Confirm delete" disabled={locked} onClick={() => { onDelete(active.id); setConfirming(null); }}>Confirm delete</button><button className="text-button" onClick={() => setConfirming(null)}>Keep</button></span>
          : <button className="button danger small" aria-label="Delete piece" disabled={locked} onClick={() => setConfirming(active.id)}><Icon name="trash" size={14} />Delete</button>}
      </div>
      <p className={`dock-verdict ${active.fit?.supported ? "supported" : "unsupported"}`}>{active.fit?.supported ? "Floor support found at the last save. Overlap and ceiling clearance are not checked." : active.fit?.reason || "Floor support is incomplete here — move it onto better-covered floor."}</p>
    </aside>
      : <p className="editor-hint">{placements.length ? "Click a piece in the 3D view to grab it, or pick one from the list above." : "Click the scanned floor in the 3D view to drop the chosen piece."}</p>}
  </>;
}
