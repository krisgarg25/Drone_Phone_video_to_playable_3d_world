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
      <span className="eyebrow">First person</span>
      <h2>Walk the scene</h2>
      <p className="arena-scene">{project.name}</p>
      <p className="inspector-copy">Step inside the reconstruction at eye height. Bots move around the site, so you can test cover and sight lines for real.</p>
      <div className="arena-label">How many bots?</div>
      <div className="bot-choices" role="group" aria-label="Bot count">
        {[1, 3, 5].map((n) => <button key={n} type="button" className={count === n ? "active" : ""} aria-pressed={count === n} disabled={!playable} onClick={() => setCount(n)}><strong>{n}</strong><small>{n === 1 ? "Practice" : n === 3 ? "Skirmish" : "Challenge"}</small></button>)}
      </div>
      <div className="arena-keys" aria-label="Controls">
        <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>Move</span>
        <span><kbd>Mouse</kbd>Look around</span>
        <span><kbd>Shift</kbd>Run</span>
        <span><kbd>Space</kbd>Jump</span>
        <span><kbd>Esc</kbd>Free the mouse</span>
      </div>
      <button className="button primary full" disabled={!playable} onClick={() => onEnter(count)}><Icon name="play" size={16} />Start walking</button>
      {!playable && <p className="form-error">This scene has no playable build yet. Run a reconstruction, then come back.</p>}
      <p className="arena-scale-note" title={`Scale source: ${project.scale.source} Accuracy: ${project.accuracy.status === "verified" ? `${project.accuracy.rmse_m ?? "unknown"} m RMSE` : "not independently verified"}`}>Scale is {project.scale.status === "metric" ? "metric" : project.scale.status} and not survey-verified, so distances read as relative.</p>
    </div>
  </section>;
}
