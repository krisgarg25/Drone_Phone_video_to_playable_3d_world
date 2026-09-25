/**
 * The domain model, mirrored from what the pipeline actually emits.
 *
 * Two kinds of content live here:
 *  - types for the live survey API responses (lib/api.ts fills them);
 *  - the static verdict tables for the brief's eight challenges and eight
 *    application verticals, which are documentation, not measurements.
 *
 * The rule the UI obeys: a capability is never rendered as available unless the
 * backend reported it. `state` below is what makes that visible in the interface.
 */

export type Verdict = "fulfilled" | "mechanism" | "partial" | "not-fulfilled" | "absent";

export const VERDICT_LABEL: Record<Verdict, string> = {
  fulfilled: "Fulfilled · measured",
  mechanism: "Mechanism · awaiting data",
  partial: "Partial · by design",
  "not-fulfilled": "Not fulfilled",
  absent: "Not built",
};

export const VERDICT_TONE: Record<Verdict, "verified" | "mechanism" | "pending" | "refused"> = {
  fulfilled: "verified",
  mechanism: "mechanism",
  partial: "pending",
  "not-fulfilled": "refused",
  absent: "refused",
};

/* Key challenges, verbatim from SIH26158 page 37, with the evidence that decides
   the verdict. Numbers here were measured on real runs on 2026-09-23. */
export const CHALLENGES: {
  id: string; title: string; verdict: Verdict; evidence: string; limit: string;
}[] = [
  {
    id: "i", title: "Limited viewing angles from a single path", verdict: "mechanism",
    evidence: "Frame budget spent on flight geometry with a minimum-baseline floor; plan reports path coverage and mean parallax. Collinear trajectories are refused outright.",
    limit: "No code sees a face that was never imaged.",
  },
  {
    id: "ii", title: "Motion blur and compression artefacts", verdict: "mechanism",
    evidence: "Per-frame Laplacian, Tenengrad, anisotropy, spectral centroid and 8×8 blocking score; frames under 0.25 weight are dropped before matching.",
    limit: "0 of 72 frames dropped on the test clip — the rule has not met genuinely bad footage.",
  },
  {
    id: "iii", title: "Variable illumination and shadows", verdict: "mechanism",
    evidence: "Low-frequency illumination field divided out of the matching copy only; colourised frames untouched.",
    limit: "Ran together with masking, so the gain is not attributable to either alone.",
  },
  {
    id: "iv", title: "Dynamic objects — vehicles, humans, animals", verdict: "fulfilled",
    evidence: "72/72 epipolar masks accepted, 4.85% of pixels vetoed. Sparse tracks −18.4%, mean support −10.3%, mean reprojection error improved 7.8% (0.3586 → 0.3305 px), fused points −0.6%, registered cameras unchanged.",
    limit: "One 12-second clip, sub-1080p. Needs a scene with real traffic and people.",
  },
  {
    id: "v", title: "GPS inaccuracy and sensor noise", verdict: "mechanism",
    evidence: "GPS enters bundle adjustment as per-camera priors with per-axis uncertainty; deterministic RANSAC rejects outliers; clock-offset bounds and vertical-datum signals reported.",
    limit: "Absolute accuracy unproven without surveyed checkpoints.",
  },
  {
    id: "vi", title: "Real-time or near-real-time processing", verdict: "not-fulfilled",
    evidence: "Dense stage measured at 554.3 s for 72 frames at 1000 px. Rate model predicted 3960.1 s against 4001.4 s measured — 1.0% error — and says the deadline is missed.",
    limit: "Progressive windows are planned by code and executed by nothing.",
  },
  {
    id: "vii", title: "Reconstruction of occluded surfaces", verdict: "partial",
    evidence: "Measured / weak / unobserved classified and kept separate; hidden regions measured; plane-derived extensions labelled constrained, never measured.",
    limit: "No generative inpainting, deliberately: a guess is not a measurement.",
  },
  {
    id: "viii", title: "Metric accuracy without extensive GCPs", verdict: "mechanism",
    evidence: "Similarity fit to GPS plus a decision report naming what this scene's telemetry can and cannot identify, with four constraint options.",
    limit: "Only hold-out figure available is 1.14 m against drifting phone VIO — the wrong reference.",
  },
];

/* Each vertical is a panel with real tools behind it, or it is not a panel.
   `built` means the maths runs today; `needs-run` means it works but no
   qualifying scene has produced numbers; `not-built` is stated as such. */
export const VERTICALS: {
  id: string; name: string; blurb: string; state: "built" | "needs-run" | "not-built";
  tools: { name: string; state: "built" | "needs-run" | "not-built"; note: string }[];
}[] = [
  {
    id: "mapping", name: "Border & strategic area mapping", blurb: "Georeferenced terrain from one pass, offline on a field laptop.",
    state: "needs-run",
    tools: [
      { name: "UTM / WGS84 export", state: "built", note: "EPSG derived from the scene's own GPS origin" },
      { name: "DSM raster (GeoTIFF)", state: "built", note: "keeps rooftops and canopy — not bare earth" },
      { name: "Line-of-sight analysis", state: "not-built", note: "needs the mesh as an occluder" },
    ],
  },
  {
    id: "disaster", name: "Disaster damage assessment", blurb: "Film a structure before access is safe, then quantify the change.",
    state: "needs-run",
    tools: [
      { name: "Cut / fill volume", state: "built", note: "volume_between with an uncertainty budget" },
      { name: "Pre / post registration", state: "not-built", note: "two surveys must share a frame first" },
      { name: "Damage classification", state: "not-built", note: "needs a trained model" },
    ],
  },
  {
    id: "urban", name: "Urban planning & smart cities", blurb: "Buildings, roads and vegetation as objects, not as a point soup.",
    state: "not-built",
    tools: [
      { name: "Building footprint extraction", state: "not-built", note: "no segmentation in this repository" },
      { name: "Road / vegetation layers", state: "not-built", note: "the brief's (iii) and (iv) lines depend on this" },
      { name: "Metric mesh + support", state: "built", note: "what any extractor would sit on" },
    ],
  },
  {
    id: "inspection", name: "Infrastructure inspection", blurb: "Measure a facade, a clearance, a height above ground — with an error bar.",
    state: "built",
    tools: [
      { name: "Distance & segment", state: "built", note: "snaps to the cloud, reports local roughness" },
      { name: "Height above ground plane", state: "built", note: "RANSAC lower envelope" },
      { name: "Point picking in viewer", state: "not-built", note: "the maths has no click path yet" },
    ],
  },
  {
    id: "construction", name: "Construction progress monitoring", blurb: "Re-fly the same route, diff the geometry, get earthwork volumes.",
    state: "not-built",
    tools: [
      { name: "Time-series diff", state: "not-built", note: "needs registration first" },
      { name: "As-planned vs as-built (IFC)", state: "not-built", note: "no importer" },
      { name: "Volume between surveys", state: "built", note: "works once both are in one frame" },
    ],
  },
  {
    id: "archaeo", name: "Archaeological documentation", blurb: "Non-invasive metric archive at each dig phase.",
    state: "needs-run",
    tools: [
      { name: "Metric archive + DSM", state: "built", note: "six formats, provenance-bound" },
      { name: "Per-phase volumes", state: "built", note: "same tool as cut/fill" },
      { name: "Phase register", state: "not-built", note: "no project model yet" },
    ],
  },
  {
    id: "twin", name: "Digital twin · interior staging", blurb: "Someone bought a flat — put furniture in it and see the result at true scale.",
    state: "needs-run",
    tools: [
      { name: "Walkable indoor scan", state: "built", note: "measured: 65.3 m walked, 16/16 waypoints, 0 falls" },
      { name: "Scale lock from one anchor", state: "built", note: "height or speed anchor, never both" },
      { name: "Furniture placement + collision", state: "built", note: "drop real-scale items, snapped to the measured floor, with a fit/clearance check and walk-mode collision" },
    ],
  },
  {
    id: "recon", name: "Military recon & mission planning", blurb: "Rehearse the route against geometry scanned from the real ground.",
    state: "built",
    tools: [
      { name: "Navmesh from scanned terrain", state: "built", note: "baked collider, scripted traversal verified" },
      { name: "Cover from geometry", state: "built", note: "occlusion-checked support per point" },
      { name: "Route planning + sight-lines", state: "not-built", note: "needs the LOS pass" },
    ],
  },
];

export const OFFICIAL_FORMATS = ["obj", "ply", "las", "geotiff", "glb/gltf", "fbx"] as const;

export const TARGETS = [
  { key: "accuracy", label: "Spatial accuracy", target: "≤ 1 m", weight: 30 },
  { key: "completeness", label: "Coverage", target: "Entire visible scene", weight: 20 },
  { key: "speed", label: "Processing", target: "< 15 min / 10 min video", weight: 20 },
  { key: "innovation", label: "Innovation", target: "Controlled ablation", weight: 15 },
  { key: "scalability", label: "Scalability", target: "Measured resource scaling", weight: 10 },
  { key: "ui", label: "User interface", target: "Import → inspect → export", weight: 5 },
];

export const fmt = {
  m: (v?: number | null) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(3)} m`),
  s: (v?: number | null) => (v == null || !Number.isFinite(v) ? "—" : v >= 90 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`),
  pct: (v?: number | null) => (v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`),
  int: (v?: number | null) => (v == null ? "—" : Math.round(v).toLocaleString("en-US")),
};
