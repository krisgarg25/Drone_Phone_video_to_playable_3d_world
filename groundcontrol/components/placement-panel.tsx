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
const AXIS: { key: 0 | 1 | 2; label: string }[] = [{ key: 0, label: "Width" }, { key: 1, label: "Height" }, { key: 2, label: "Depth" }];
const metres = (value: number) => `${value.toFixed(2)} m`;
const SAVE_CHIP: Record<EditorProps["saveState"], string> = { clean: "Saved", queued: "Unsaved", saving: "Saving…", error: "Not saved" };
/** Honest caption for an imported item: its file-read size, plus the scene's scale caveat. */
function modelNote(model: ImportedModel) {
  if (model.scale_status === "metric") return `${model.true_size_note}; scene scale is metric.`;
  if (model.scale_status === "estimated") return `${model.true_size_note}, but this scene's scale is estimated (${model.scale_source}), not measured — its on-scene size is only as accurate as that scale.`;
  return `${model.true_size_note}, but this scene has only relative scale (${model.scale_source}) — treat its size as nominal, not measured.`;
}

/** A side panel to pick and list pieces, and a dock that edits the selected one in 3D. */
export function PlacementPanel({
  project, placements, selected, place, saveState, offline, message, status, importing,
  onPickItem, onYawArm, onCancelPlace, onRetrySave, onSelect, onSlide, onLift, onYaw, onResize, onDuplicate, onDelete, onExport, onFit, onImportModel,
}: EditorProps) {
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"add" | "scene">("add");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [seen, setSeen] = useState(selected);
  // Grabbing a piece in 3D shows it in the list.
  if (seen !== selected) { setSeen(selected); if (selected) setTab("scene"); }
  const active = placements.find((p) => p.id === selected) ?? null;
  const arming: FurnitureItem | undefined = project.furniture.find((f) => f.item === place?.item);
  const library = project.furniture.filter((f) => f.label.toLowerCase().includes(search.toLowerCase()));
  const locked = offline;
  const unsupported = placements.filter((p) => !p.fit?.supported).length;
  return <>
    <aside className="place-side" aria-label="Furniture">
      <header className="place-side-head">
        <div><h2>Place items</h2><p>Real-size furniture or your own 3D model, dropped onto the scan.</p></div>
        <span className={`place-save save-${saveState}`} title={status}>{SAVE_CHIP[saveState]}</span>
      </header>
      <div className="plan-view place-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "add"} className={tab === "add" ? "active" : ""} onClick={() => setTab("add")}><Icon name="plus" size={14} />Add</button>
        <button role="tab" aria-selected={tab === "scene"} className={tab === "scene" ? "active" : ""} onClick={() => setTab("scene")}><Icon name="list" size={14} />In the scene<em>{placements.length}</em></button>
      </div>

      {tab === "add" ? <div className="place-body">
        {arming && place ? <div className="place-arming">
          <FurnitureSymbol item={arming.item} />
          <div><strong>Placing: {arming.label}</strong><small>Click the floor in the 3D view. Click again for another.</small></div>
          <div className="place-turn"><button disabled={locked} onClick={() => onYawArm(-45)} aria-label="Turn left 45°">⟲</button><b>{place.yaw}°</b><button disabled={locked} onClick={() => onYawArm(45)} aria-label="Turn right 45°">⟳</button></div>
          <button className="button secondary small" onClick={onCancelPlace}>Stop</button>
        </div> : <p className="place-tip"><Icon name="cursor" size={14} />Pick an item, then click the floor in the 3D view.</p>}
        <label className="search-field"><Icon name="search" size={15} /><input aria-label="Search furniture" placeholder="Search items" value={search} onChange={(e) => setSearch(e.target.value)} /></label>
        <div className="place-catalogue">{library.map((f) => <button key={f.item} className={`place-item${place?.item === f.item ? " on" : ""}`} aria-pressed={place?.item === f.item} disabled={locked} onClick={() => onPickItem(f.item)}>
          <FurnitureSymbol item={f.item} /><strong>{f.label}</strong><small>{f.imported ? "Your model · " : ""}{f.size[0]} × {f.size[2]} m</small></button>)}</div>
        {!library.length && <p className="plan-note">{project.furniture.length ? "Nothing matches. Try another name." : "The catalogue did not load. Check the service connection."}</p>}
        <label className={`button secondary full import-model${importing || locked ? " is-busy" : ""}`} data-tip="A .glb or .gltf file; its real size is read from the file">
          <Icon name="upload" size={15} />{importing ? "Importing…" : "Add your own 3D model"}
          <input type="file" accept=".glb,.gltf,model/gltf-binary,model/gltf+json" disabled={importing || locked} aria-label="Import glTF model" onChange={(e) => { const file = e.currentTarget.files?.[0]; if (file) onImportModel(file); e.currentTarget.value = ""; }} />
        </label>
      </div> : <div className="place-body">
        {!placements.length ? <div className="tool-empty"><Icon name="cube" size={26} /><p>Nothing placed yet.</p><button className="button secondary small" onClick={() => setTab("add")}><Icon name="plus" size={13} />Add an item</button></div> : <>
          {unsupported > 0 && <p className="insp-verdict warn"><Icon name="alert" size={16} />{unsupported} item{unsupported === 1 ? " is" : "s are"} not fully on the floor. Move {unsupported === 1 ? "it" : "them"} onto better-scanned ground.</p>}
          <ul className="place-scene">{placements.map((p) => <li key={p.id}><button className={selected === p.id ? "on" : ""} aria-pressed={selected === p.id} onClick={() => onSelect(selected === p.id ? null : p.id)}>
            <FurnitureSymbol item={p.item} /><span><strong>{p.label}</strong><small>{metres(p.size[0])} × {metres(p.size[2])} · faces {p.yaw_deg}°</small></span>
            <em className={p.fit?.supported ? "ok" : "bad"} data-tip={p.fit?.supported ? "Standing on scanned floor" : "Not fully on the floor"}>{p.fit?.supported ? "On floor" : "Check"}</em></button></li>)}</ul>
        </>}
      </div>}

      <footer className="place-foot">
        {message && <p className="place-message" role="status">{message}</p>}
        {saveState === "error" && <button className="button danger small full" onClick={onRetrySave}>Save failed. Try again</button>}
        <div className="insp-actions"><button className="button secondary small" onClick={onFit}><Icon name="expand" size={14} />Fit view</button>{placements.length > 0 && <button className="button secondary small" onClick={onExport}><Icon name="download" size={14} />Layout (GeoJSON)</button>}</div>
      </footer>
    </aside>

    {active ? <aside className="editor-dock" aria-label="Selected piece controls">
      <div className="dock-head">
        <FurnitureSymbol item={active.item} />
        <div><strong>{active.label}</strong><small>{active.fit?.supported ? "Standing on the floor" : "Not fully on the floor"} · faces {active.yaw_deg}°</small></div>
        <button className="icon-button" aria-label="Deselect furniture" data-tip="Done" onClick={() => onSelect(null)}><Icon name="close" size={14} /></button>
      </div>
      {active.model && <p className="dock-model-note">{modelNote(active.model)}</p>}
      <p className="dock-tip">Drag it in the 3D view, or nudge it exactly here.</p>
      <div className="dock-row">
        <div className="dock-section"><span>Move {MOVE_STEP_M * 100} cm</span>
          <div className="pad">
            <span />
            <button aria-label="Move forward" data-tip="Forward" disabled={locked} onClick={() => onSlide(active, 0, -MOVE_STEP_M)}>↑</button>
            <span />
            <button aria-label="Move left" data-tip="Left" disabled={locked} onClick={() => onSlide(active, -MOVE_STEP_M, 0)}>←</button>
            <b className="pad-step" />
            <button aria-label="Move right" data-tip="Right" disabled={locked} onClick={() => onSlide(active, MOVE_STEP_M, 0)}>→</button>
            <span />
            <button aria-label="Move back" data-tip="Back" disabled={locked} onClick={() => onSlide(active, 0, MOVE_STEP_M)}>↓</button>
            <span />
          </div></div>
        <div className="dock-section"><span>Raise</span>
          <div className="lift">
            <button aria-label="Raise piece" data-tip={`Up ${LIFT_STEP_M * 100} cm`} disabled={locked} onClick={() => onLift(active, LIFT_STEP_M)}>▲</button>
            <b>{metres(active.center_y)}</b>
            <button aria-label="Lower piece" data-tip={`Down ${LIFT_STEP_M * 100} cm`} disabled={locked} onClick={() => onLift(active, -LIFT_STEP_M)}>▼</button>
          </div></div>
      </div>
      <div className="dock-section"><span>Turn</span>
        <div className="dock-group" role="group" aria-label="Rotate">
          <button aria-label="Rotate left 45 degrees" disabled={locked} onClick={() => onYaw(active, -45)}>−45°</button>
          <button aria-label="Rotate left 15 degrees" disabled={locked} onClick={() => onYaw(active, -15)}>−15°</button>
          <b>{active.yaw_deg}°</b>
          <button aria-label="Rotate right 15 degrees" disabled={locked} onClick={() => onYaw(active, 15)}>+15°</button>
          <button aria-label="Rotate right 45 degrees" disabled={locked} onClick={() => onYaw(active, 45)}>+45°</button>
        </div></div>
      <div className="dock-section"><span>Size</span>
        <div className="dock-group dock-resize" role="group" aria-label="Resize">
          {AXIS.map(({ key, label }) => <div className="axis" key={label}>
            <span>{label}</span>
            <button aria-label={`Decrease ${label}`} disabled={locked || active.size[key] <= SIZE_MIN_M} onClick={() => onResize(active, key, -RESIZE_STEP_M)}>−</button>
            <b title={`${SIZE_MIN_M}–${SIZE_MAX_M} m`}>{metres(active.size[key])}</b>
            <button aria-label={`Increase ${label}`} disabled={locked || active.size[key] >= SIZE_MAX_M} onClick={() => onResize(active, key, RESIZE_STEP_M)}>+</button>
          </div>)}
        </div></div>
      <div className="dock-group dock-foot">
        <button className="button secondary small" disabled={locked} onClick={() => onDuplicate(active)}><Icon name="file" size={14} />Duplicate</button>
        {confirming === active.id ? <span className="draft-actions"><button className="button danger small" aria-label="Confirm delete" disabled={locked} onClick={() => { onDelete(active.id); setConfirming(null); }}>Delete</button><button className="text-button" onClick={() => setConfirming(null)}>Keep</button></span>
          : <button className="button danger small" aria-label="Delete piece" disabled={locked} onClick={() => setConfirming(active.id)}><Icon name="trash" size={14} />Delete</button>}
      </div>
      {!active.fit?.supported && <p className="dock-verdict unsupported">{active.fit?.reason || "Part of it is over floor the drone did not scan well. Move it onto better-covered floor."}</p>}
    </aside>
      : <p className="editor-hint">{arming ? `Click the floor to drop the ${arming.label.toLowerCase()}.` : placements.length ? "Click a piece in the 3D view to grab it." : "Pick an item on the left, then click the floor here."}</p>}
  </>;
}
