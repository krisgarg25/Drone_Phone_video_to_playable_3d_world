// =====================================================================
// lidar_overlay.js — sparse world-space "capture map" for the camera overlay
//
// The lidar-like effect on viewer/capture.html: while recording, image
// features are unprojected with the live pose + depth estimate into a
// voxel-hashed world map (persist), and every frame that map is
// re-projected back to overlay pixels (project). Because the points live
// in WORLD space, the green dots stick to scanned surfaces as the camera
// moves, instead of painting a screen-aligned 2D coverage grid.
//
// Pure functions: no DOM, no THREE. They operate on plain {x,y,z} and
// {x,y,z,w} objects, so the capture page (whose THREE vectors/quaternions
// are structurally compatible) and the node tests drive the exact same
// shipped code path.
// =====================================================================
var LidarOverlay = (function () {
  "use strict";

  var DEFAULTS = {
    voxel: 0.09,      // world voxel size (m) — keeps dot density lidar-like
    maxPoints: 6000,  // map cap; stalest point is recycled when full
    near: 0.25,       // clip points nearer than this (m)
    far: 14,          // ... and farther than this (m)
    focal: 0.75,      // pinhole focal in normalized-image units (160x120 proc frame)
    frameW: 160,
    frameH: 120,
    margin: 1.12,     // frustum margin in normalized coords
    backface: 0.0     // a dot seen from behind its surface is culled
  };

  function createMap(opts) {
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    if (opts) for (k in opts) o[k] = opts[k];
    return { opts: o, voxels: new Map(), points: [], clock: 0 };
  }

  function clear(map) {
    map.voxels.clear();
    map.points.length = 0;
    map.clock = 0;
  }

  // rotate v by quaternion q; result written into out (allows scratch reuse)
  function rot(q, x, y, z, out) {
    var tx = 2 * (q.y * z - q.z * y);
    var ty = 2 * (q.z * x - q.x * z);
    var tz = 2 * (q.x * y - q.y * x);
    out.x = x + q.w * tx + (q.y * tz - q.z * ty);
    out.y = y + q.w * ty + (q.z * tx - q.x * tz);
    out.z = z + q.w * tz + (q.x * ty - q.y * tx);
    return out;
  }

  var SCRATCH = { x: 0, y: 0, z: 0 };

  // persist: unproject one observed feature into the world map.
  // cam = {pos:{x,y,z}, quat:{x,y,z,w}} (camera-to-world),
  // normX/normY = normalized image coords of the feature ([-1,1]),
  // depth = estimated metric depth along its ray.
  // Returns the index of the (possibly merged) map point, or -1 if rejected.
  function persist(map, cam, normX, normY, depth, weight) {
    var o = map.opts;
    if (!isFinite(normX) || !isFinite(normY) || !isFinite(depth)) return -1;
    if (depth < o.near || depth > o.far) return -1;
    rot(cam.quat,
        normX * o.focal * depth,
        -normY * o.focal * depth,
        -depth, SCRATCH);
    var x = cam.pos.x + SCRATCH.x;
    var y = cam.pos.y + SCRATCH.y;
    var z = cam.pos.z + SCRATCH.z;
    var v = o.voxel;
    var key = Math.round(x / v) + "," + Math.round(y / v) + "," + Math.round(z / v);
    var idx = map.voxels.get(key);
    map.clock++;
    if (idx !== undefined) {
      var p = map.points[idx];
      p.seen += (weight || 1);
      p.last = map.clock;
      return idx;
    }
    if (map.points.length >= o.maxPoints) {
      // recycle the least recently seen point
      var oi = 0, ot = Infinity;
      for (var i = 0; i < map.points.length; i++) {
        if (map.points[i].last < ot) { ot = map.points[i].last; oi = i; }
      }
      map.voxels.delete(map.points[oi].key);
      idx = oi;
    } else {
      idx = map.points.length;
      map.points.push(null);
    }
    // surface orientation ~= direction back to the camera at first sight;
    // backface-culling with it stops dots shining through objects later
    var dx = cam.pos.x - x, dy = cam.pos.y - y, dz = cam.pos.z - z;
    var dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    map.points[idx] = { x: x, y: y, z: z, seen: weight || 1, last: map.clock,
                        key: key, vx: dx / dl, vy: dy / dl, vz: dz / dl };
    map.voxels.set(key, idx);
    return idx;
  }

  // project: world map + camera pose -> overlay pixels in the proc frame.
  // Returns [{px, py, seen, idx}] for every visible map point.
  function project(map, cam, frameW, frameH) {
    var o = map.opts;
    var fw = frameW || o.frameW, fh = frameH || o.frameH;
    var q = { x: -cam.quat.x, y: -cam.quat.y, z: -cam.quat.z, w: cam.quat.w };
    var t = { x: 0, y: 0, z: 0 };
    var dots = [];
    for (var i = 0; i < map.points.length; i++) {
      var p = map.points[i];
      if (!p) continue;
      var vx = cam.pos.x - p.x, vy = cam.pos.y - p.y, vz = cam.pos.z - p.z;
      // backface cull: view direction now (point->camera) vs at capture time;
      // stops dots shining through objects when seen from behind their surface
      if (vx * p.vx + vy * p.vy + vz * p.vz <=
          o.backface * (Math.sqrt(vx * vx + vy * vy + vz * vz) || 1)) continue;
      rot(q, -vx, -vy, -vz, t);
      if (t.z > -o.near) continue;          // behind / too near the camera
      var d = -t.z;
      if (d > o.far) continue;
      var nx = t.x / (o.focal * d), ny = -t.y / (o.focal * d);
      if (nx < -o.margin || nx > o.margin || ny < -o.margin || ny > o.margin) continue;
      dots.push({ px: (nx * 0.5 + 0.5) * fw, py: (ny * 0.5 + 0.5) * fh,
                  seen: p.seen, idx: i });
    }
    return dots;
  }

  // proc-frame pixel -> canvas pixel mapping for an object-fit: cover video
  function coverMapping(frameW, frameH, canvasW, canvasH, videoW, videoH) {
    var scale = Math.max(canvasW / videoW, canvasH / videoH);
    return {
      offX: (canvasW - videoW * scale) / 2,
      offY: (canvasH - videoH * scale) / 2,
      sx: scale * (videoW / frameW),
      sy: scale * (videoH / frameH)
    };
  }

  // coverage-gap report: which bearings were never scanned, per pitch band.
  // Orientation-anchored (compass/gyro) so it cannot drift the way
  // position-guessed world points do — this is the "did I cover the bed?" truth.
  // cov = Float32Array[pitchRows*yawCols], returns [{band, from, to, cells}]
  // sorted by size desc; `to` may exceed 360 for wrap-around arcs.
  function uncoveredArcs(cov, yawCols, pitchRows, thr, minCells) {
    if (thr === undefined) thr = 0.03;
    minCells = minCells || 4;
    var bands = [
      { band: "low",  r0: 0, r1: Math.round(pitchRows * 7 / 18) },
      { band: "mid",  r0: Math.round(pitchRows * 7 / 18), r1: Math.round(pitchRows * 11 / 18) },
      { band: "high", r0: Math.round(pitchRows * 11 / 18), r1: pitchRows }
    ];
    var out = [];
    for (var bi = 0; bi < bands.length; bi++) {
      var b = bands[bi];
      var unc = new Array(yawCols);
      var anyCovered = false;
      for (var c = 0; c < yawCols; c++) {
        var m = 0;
        for (var r = b.r0; r < b.r1; r++) {
          var v = cov[r * yawCols + c];
          if (v > m) m = v;
        }
        unc[c] = m <= thr;
        if (!unc[c]) anyCovered = true;
      }
      if (!anyCovered) {
        out.push({ band: b.band, from: 0, to: 360, cells: yawCols });
        continue;
      }
      // linearize the circle starting just after a covered column
      var start = 0;
      while (unc[start]) start++;
      var run = 0, runStart = -1;
      for (var i = 0; i <= yawCols; i++) {
        var idx = (start + i) % yawCols;
        if (unc[idx]) { if (run === 0) runStart = idx; run++; }
        else if (run > 0) {
          if (run >= minCells) {
            var from = runStart * (360 / yawCols);
            out.push({ band: b.band, from: from,
                       to: from + run * (360 / yawCols), cells: run });
          }
          run = 0;
        }
      }
    }
    out.sort(function (a, b2) { return b2.cells - a.cells; });
    return out;
  }

  return {
    DEFAULTS: DEFAULTS,
    createMap: createMap,
    clear: clear,
    persist: persist,
    project: project,
    coverMapping: coverMapping,
    uncoveredArcs: uncoveredArcs
  };
})();
if (typeof window !== "undefined") window.LidarOverlay = LidarOverlay;
