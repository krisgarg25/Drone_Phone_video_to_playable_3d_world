"use client";

import Image from "next/image";
import type { Layer, ProjectDetail } from "@/lib/workspace";
import { duration } from "@/lib/workspace";
import { MODE_INFO } from "@/lib/modes";
import { GnssReference } from "./project-details";
import { Icon } from "./studio-icons";
import "./inspect.css";

const CLASS_COLOR: Record<string, string> = { ground: "rgb(160,140,110)", road: "rgb(160,164,172)", building: "rgb(240,150,64)", vegetation: "rgb(76,196,104)", obstacle: "rgb(240,90,84)", person: "rgb(96,160,255)" };
const CLASS_LABEL: Record<string, string> = { ground: "Ground", road: "Roads", building: "Buildings", vegetation: "Trees and plants", obstacle: "Obstacles", person: "People" };
const NEXT = ["measure", "inspect", "ops", "plan", "twin", "walk"];
const TIP: Record<string, string> = {
  inspection: "Click a photo in the strip to see a spot up close, then log defects in Inspect.",
  survey: "Check the scale above before you measure. Survey inputs are under Project.",
  response: "Turn on Camera coverage in Layers: grey areas were not seen, which is not the same as safe.",
  heritage: "Walk through the site, and save a picture of any view with the camera button.",
  general: "Rotate around the model, click photos in the strip, and try the layers to see how it was built.",
};

/** The Explore panel's first tab: what this scene is, how far to trust it, and where to go next. */
export function ExploreOverview({ project, layers, capabilities, ready, selectedFrame, onLayer, onTab, onCloseFrame }: {
  project: ProjectDetail; layers: Record<Layer, boolean>; capabilities: Record<string, boolean>; ready: boolean; selectedFrame: number;
  onLayer: (layer: Layer, value: boolean) => void; onTab: (tab: string) => void; onCloseFrame: () => void;
}) {
  const frame = selectedFrame >= 0 ? project.frames[selectedFrame] : null;
  const classes = Object.entries(project.semantics?.counts ?? {}).filter(([, n]) => n > 0)
    .map(([cls, n]) => ({ cls, n, area: project.semantics?.summary?.[cls]?.area_m2 ?? 0 })).sort((a, b) => b.area - a.area || b.n - a.n);
  const total = classes.reduce((sum, c) => sum + (c.area || c.n), 0) || 1;
  const scale = project.scale.status;
  const tabs = project.application?.workspace_tabs;
  const next = NEXT.filter((t) => !tabs || tabs.includes(t));

  return <>
    {frame && <section className="inspector-section explore-photo">
      <div className="insp-head"><div><h3>Photo {selectedFrame + 1}</h3><p>{frame.t_sec != null ? `${duration(frame.t_sec)} into the video` : frame.name}{frame.camera_index != null ? " · the view jumped to where it was taken" : ""}</p></div>
        <button className="icon-button" aria-label="Close photo" data-tip="Close" onClick={onCloseFrame}><Icon name="close" size={15} /></button></div>
      <Image unoptimized width={640} height={400} src={frame.url} alt={`Video photo ${selectedFrame + 1}`} />
    </section>}

    <section className="inspector-section scene-summary">
      <div className="mini-stats">
        <div><span>Photos placed</span><b>{project.registered_count.toLocaleString()}<small>/ {project.frames.length || project.registered_count}</small></b></div>
        <div><span>Scale</span><b className={scale === "metric" ? "tone-good" : scale === "estimated" ? "tone-warnings" : ""}>{scale === "metric" ? "Metres" : scale === "estimated" ? "Approx. metres" : "No real scale"}</b></div>
        <div><span>Location</span><b className={project.georeference.status === "georeferenced" ? "tone-good" : ""}>{project.georeference.status === "georeferenced" ? "On the map" : "Local only"}</b></div>
        <div><span>Quality check</span><b className={project.quality ? `tone-${project.quality.status}` : ""}>{project.quality ? (project.quality.status === "failed" ? "Blocked" : project.quality.status === "warnings" ? `${project.quality.warnings.length} limit${project.quality.warnings.length === 1 ? "" : "s"}` : "Passed") : "—"}</b></div>
      </div>
      {project.quality && <button className="text-button explore-more" onClick={() => onTab("quality")}>See the quality check<Icon name="chevron" size={12} /></button>}
    </section>

    <section className="inspector-section">
      <div className="section-label"><span>Where to next</span></div>
      <div className="explore-next">{next.map((t) => <button key={t} onClick={() => onTab(t)} data-tip={MODE_INFO[t].what}><Icon name={MODE_INFO[t].icon} size={18} /><span>{MODE_INFO[t].label}</span></button>)}</div>
    </section>

    {classes.length > 0 && <section className="inspector-section">
      <div className="section-label"><span>What is in the scene</span></div>
      <div className="explore-classes">{classes.map((c) => <div key={c.cls} className="explore-class">
        <span><i style={{ background: CLASS_COLOR[c.cls] ?? "hsl(var(--st-muted))" }} />{CLASS_LABEL[c.cls] ?? c.cls}<b>{c.area ? `${Math.round(c.area).toLocaleString()} m²` : c.n.toLocaleString()}</b></span>
        <em><i style={{ width: `${Math.max(2, (100 * (c.area || c.n)) / total)}%`, background: CLASS_COLOR[c.cls] ?? "hsl(var(--st-muted))" }} /></em>
      </div>)}</div>
      <label className="insp-show explore-toggle"><input type="checkbox" checked={layers.semantics} disabled={!ready || !capabilities.semantics} onChange={(e) => onLayer("semantics", e.target.checked)} />Colour the model by type</label>
      <p className="plan-note">Rough labels from shape and colour. Areas are approximate, not surveyed.</p>
    </section>}

    <section className="inspector-section">
      <div className="section-label"><span>Scale and location</span></div>
      <div className={`measure-scale scale-${scale}`}><Icon name={scale === "metric" ? "check" : "alert"} size={16} />
        <span><b>{scale === "metric" ? "Distances are in real metres" : scale === "estimated" ? "Distances are approximate" : "Distances are not in metres"}</b>{project.scale.source}</span></div>
      <div className="datum-row"><span>Coordinates</span><span>{project.georeference.status === "local" ? "Local to this scene" : project.georeference.crs}</span></div>
      <div className="datum-row"><span>Checked against survey points</span><span>{project.accuracy.status === "verified" ? `Yes, ${project.accuracy.rmse_m} m typical error` : "Not yet"}</span></div>
      {project.scale_check && <div className="datum-row" data-tip="A second, AI-based estimate of scale. It never changes a measurement; a big gap is worth a look."><span>Second scale estimate</span><span>{project.scale_check.gap_percent === null ? "Not comparable" : `${project.scale_check.gap_percent}% different`}</span></div>}
      <details className="ops-notes"><summary>GPS details</summary><GnssReference project={project} /></details>
    </section>

    <section className="inspector-section">
      <p className="explore-tip"><Icon name="tip" size={16} />{TIP[project.workflow] ?? TIP.general}</p>
    </section>
  </>;
}
