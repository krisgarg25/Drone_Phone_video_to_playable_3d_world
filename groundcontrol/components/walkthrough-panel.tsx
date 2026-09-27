"use client";

import { useState } from "react";
import type { ProjectDetail } from "@/lib/workspace";
import { Icon } from "./studio-icons";

/**
 * The arena door, and nothing else: name the scene, choose a bot count, drop in.
 * The combat HUD inside the viewer is the play-time UI, so no controls are
 * duplicated here and none are stacked over the play area afterwards.
 */
export function WalkthroughPanel({ project, bots, onEnter }: { project: ProjectDetail; bots: number; onEnter: (bots: number) => void }) {
  const [count, setCount] = useState(bots);
  const playable = project.viewable && !!project.viewer_url;
  return <section className="arena-start" aria-label="Arena start">
    <div className="arena-card">
      <span className="eyebrow">PRESENCE / ARENA</span>
      <h2>Arena</h2>
      <p className="arena-scene">{project.name}</p>
      <p className="inspector-copy">First-person combat inside the reconstruction, with bots fighting around you. Everything you need while playing is already in the game view.</p>
      <div className="bot-choices" role="group" aria-label="Bot count">
        {[1, 3, 5].map((n) => <button key={n} type="button" className={count === n ? "active" : ""} aria-pressed={count === n} disabled={!playable} onClick={() => setCount(n)}><strong>{n}</strong><small>{n === 1 ? "Practice" : n === 3 ? "Skirmish" : "Challenge"}</small></button>)}
      </div>
      <button className="button primary full" disabled={!playable} onClick={() => onEnter(count)}><Icon name="play" size={16} />Enter arena</button>
      {!playable && <p className="form-error">This scene has no playable build yet. Run a reconstruction, then come back.</p>}
      <p className="arena-scale-note" title={`Scale source: ${project.scale.source} Accuracy: ${project.accuracy.status === "verified" ? `${project.accuracy.rmse_m ?? "unknown"} m RMSE` : "not independently verified"}`}>Scale is {project.scale.status === "metric" ? "metric" : project.scale.status} and not survey-verified, so distances read as relative.</p>
    </div>
  </section>;
}
