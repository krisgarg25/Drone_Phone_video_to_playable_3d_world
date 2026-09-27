// Pure picked-geometry estimates. These are not surface-support or accuracy claims.
const KINDS = [null, "point", "annotation", "distance", "height", "area", "volume"];
const point = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const equal = (a, b) => a.every((v, i) => v === b[i]);

/**
 * Y-up coordinates, in metres or unchanged scene units. Segments are Euclidean
 * (including the closing edge for area/volume). Horizontal and vertical are
 * accumulated over the open path; perimeter is the closed XZ footprint.
 * Volume needs a supported surface and reference plane, so is always null here.
 */
export function previewMeasurement(kind, points, unit = "m") {
  if (!KINDS.includes(kind)) throw new TypeError("Invalid measurement kind");
  if (unit !== "m" && unit !== "units") throw new TypeError("Invalid measurement unit");
  if (!Array.isArray(points) || !points.every(point)) throw new TypeError("Invalid measurement point");
  const pts = points.slice();
  const closed = kind === "area" || kind === "volume";
  if (closed && pts.length > 1 && equal(pts[0], pts[pts.length - 1])) pts.pop();
  const segments = [];
  let horizontal = 0, vertical = 0, perimeter = 0, twiceArea = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    segments.push(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    horizontal += Math.hypot(b[0] - a[0], b[2] - a[2]);
    vertical += Math.abs(b[1] - a[1]);
  }
  if (pts.length > 1) {
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      perimeter += Math.hypot(b[0] - a[0], b[2] - a[2]);
      // Translate to the first point to avoid cancellation at large map offsets.
      twiceArea += (a[0] - pts[0][0]) * (b[2] - pts[0][2]) - (b[0] - pts[0][0]) * (a[2] - pts[0][2]);
    }
    if (closed) {
      const a = pts[pts.length - 1], b = pts[0];
      segments.push(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    }
  }
  let value = null;
  if (kind === "distance" && pts.length >= 2) value = segments.reduce((sum, v) => sum + v, 0);
  if (kind === "height" && pts.length >= 2) value = Math.abs(pts[1][1] - pts[0][1]);
  if (kind === "area" && pts.length >= 3) value = Math.abs(twiceArea) / 2;
  return { value, unit: unit + (kind === "area" ? "²" : kind === "volume" ? "³" : ""), segments, horizontal, vertical, perimeter };
}
