import type { IconName } from "@/components/studio-icons";

/**
 * What each workspace is for, in plain words. The mode bar tooltips, the panel
 * header and the "How it works" popover all read from here, so the wording stays
 * the same everywhere.
 */
export type ModeInfo = { label: string; icon: IconName; what: string; how: string[] };

export const MODE_INFO: Record<string, ModeInfo> = {
  layers: {
    label: "Explore", icon: "compass",
    what: "Look around the 3D model and see what it is made of.",
    how: ["Drag to rotate, scroll to zoom, right-drag to slide the view.", "Open Layers (top right of the view) to show the camera path, points or the solid surface.", "Click a photo in the strip at the bottom to jump to where it was taken."],
  },
  measure: {
    label: "Measure", icon: "ruler",
    what: "Get distances, heights, areas and volumes by clicking on the model.",
    how: ["Choose what to measure below.", "Click points on the model. The reading on the view updates as you move.", "Give it a name and press Save to keep it."],
  },
  inspect: {
    label: "Inspect", icon: "search",
    what: "Check a structure: log defects, find the photos that saw a spot, measure tilt and cable sag.",
    how: ["Choose a tool below.", "Click on the model where the guide on the view asks.", "Results show here and on the model. Download a report when done."],
  },
  ops: {
    label: "Operations", icon: "radar",
    what: "Compare two flights and work out damage, debris, flooding and vehicle access.",
    how: ["Choose Disaster, Construction or Border at the top.", "Start an analysis. If it needs a place, click it on the model.", "Download the results as a field pack."],
  },
  twin: {
    label: "Twin", icon: "twin",
    what: "See the site as it looks, as it was measured, and how sure each part is, with a list of every asset.",
    how: ["Switch between Visual, Measured and Evidence at the top.", "Click an asset in the list to fly to it.", "Export a package for a game engine or 3D Tiles."],
  },
  plan: {
    label: "Plan", icon: "building",
    what: "Draw new buildings, roads and zones on the scan and check them against planning rules.",
    how: ["Choose a shape to draw.", "Click corners on the ground, then press Finish or Enter.", "Select a shape to change its size or height. Rule problems are listed below."],
  },
  mission: {
    label: "Mission", icon: "flag",
    what: "Mark positions and routes, check what can be seen from where, then rehearse the plan.",
    how: ["Choose Symbol, Route, Phase line or Line of sight.", "Click on the ground to place it.", "Press Analyse for exposure, then Rehearse to walk it in first person."],
  },
  place: {
    label: "Place", icon: "cube",
    what: "Drop real-size furniture or your own 3D models into the scene.",
    how: ["Pick an item from the strip.", "Click the floor in the view to drop it.", "Drag it, or use the panel to move, turn and resize."],
  },
  walk: {
    label: "Walk", icon: "walk",
    what: "Step inside the scene in first person.",
    how: ["Choose how many bots to play against.", "Press Enter arena.", "WASD to move, mouse to look. Esc to leave."],
  },
};

MODE_INFO.quality = { ...MODE_INFO.layers, label: "Quality" };
MODE_INFO.details = { ...MODE_INFO.layers, label: "Project" };
MODE_INFO.exports = { ...MODE_INFO.layers, label: "Files" };
