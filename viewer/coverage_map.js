// =====================================================================
// coverage_map.js — surfel coverage map with per-surface viewing-direction
// evidence. This is the "have I actually covered this?" engine.
//
// WHY THIS EXISTS (and why the old signals could not work)
//   The previous capture page answered coverage two ways, both unsound:
//     1. World dots unprojected with a position obtained by double-integrating
//        devicemotion acceleration. That position drifts metres in seconds, so
//        revisiting a surface produced DIFFERENT voxel keys and the map could
//        never say "already covered".
//     2. A yaw x pitch evidence sphere with no position term at all. It records
//        which BEARINGS you faced, globally. Walk around a cube and the far face
//        reads covered because you faced that bearing from the other side.
//   Coverage is a property of a SURFACE seen from a set of DIRECTIONS. It needs
//   real 6-DoF pose and real depth. On Android that is WebXR immersive-ar
//   (ARCore pose) + the depth-sensing module. This module consumes those.
//
// WHAT "COVERED" MEANS HERE
//   Not "I pointed at it" — that is what made the old display useless. A surfel
//   counts as covered when it has been seen from >= minDirsCovered distinct
//   viewing-direction bins within usable range. Direction diversity is the thing
//   that predicts whether COLMAP/3DGS can actually triangulate the surface: one
//   long stare from a single spot gives thousands of observations and zero
//   baseline, and reconstructs badly.
//
// Pure functions: no DOM, no THREE, no WebXR. Operates on plain {x,y,z} objects
// so the capture page and the node tests drive the exact same shipped code path.
// =====================================================================
var CoverageMap = (function () {
  "use strict";

  var DEFAULTS = {
    voxel: 0.10,          // surfel size (m); world point is quantized to this
    maxSurfels: 30000,    // insert cap; existing surfels keep updating past it
    near: 0.30,           // ignore depth nearer than this (m)
    far: 6.0,             // ARCore motion-stereo depth is unreliable past ~6 m
    azBins: 8,            // azimuth bins  \  8 * 3 = 24 direction bins,
    elBins: 3,            // elevation bins /  so the mask fits a Uint32
    minDirsPartial: 2,    // distinct directions to leave the "thin" state
    minDirsCovered: 3,    // ... and to count as covered
    goodDist: 3.5,        // observations beyond this do not count toward covered
    maxIncidenceDeg: 68,  // grazing hits do not reconstruct; rejected outright
    margin: 1.10          // frustum margin in normalized coords, for project()
  };

  var STATE_THIN = 0;     // seen, but from too few directions -> needs angles
  var STATE_PARTIAL = 1;  // getting there
  var STATE_COVERED = 2;  // enough baseline to reconstruct

  function createMap(opts) {
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    if (opts) for (k in opts) o[k] = opts[k];
    o.cosMaxIncidence = Math.cos(o.maxIncidenceDeg * Math.PI / 180);
    return {
      opts: o,
      voxels: new Map(),   // "i,j,k" -> index into surfels
      surfels: [],
      clock: 0,
      full: false,
      rejected: 0          // observations dropped (range / incidence / non-finite)
    };
  }

  function clear(map) {
    map.voxels.clear();
    map.surfels.length = 0;
    map.clock = 0;
    map.full = false;
    map.rejected = 0;
  }

  function popcount32(v) {
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    v = (v + (v >> 4)) & 0x0f0f0f0f;
    return (v * 0x01010101) >> 24;
  }

  // Bin a unit direction (surfel -> camera, world frame) into one of
  // azBins * elBins cells. World-frame binning keeps the mask meaningful even
  // as the surfel's normal estimate is refined.
  function dirBin(o, dx, dy, dz) {
    var az = Math.atan2(dx, dz);                       // -pi..pi
    var ai = Math.floor((az + Math.PI) / (2 * Math.PI) * o.azBins) % o.azBins;
    if (ai < 0) ai += o.azBins;
    var el = Math.max(-1, Math.min(1, dy));            // dy of a unit vector
    var ei = Math.floor((el + 1) / 2 * o.elBins);
    if (ei >= o.elBins) ei = o.elBins - 1;
    if (ei < 0) ei = 0;
    return ei * o.azBins + ai;
  }

  function stateOf(map, s) {
    var o = map.opts, n = popcount32(s.goodDirs);
    if (n >= o.minDirsCovered) return STATE_COVERED;
    if (n >= o.minDirsPartial) return STATE_PARTIAL;
    return STATE_THIN;
  }

  // observe: fold one depth sample into the map.
  //   camPos  {x,y,z}  camera position, world frame
  //   pt      {x,y,z}  the observed surface point, world frame
  //   normal  {x,y,z}  optional unit surface normal (from the depth grid); when
  //                    omitted, the running mean view direction stands in for it
  // Returns the surfel index, or -1 when the sample was rejected.
  function observe(map, camPos, pt, normal) {
    var o = map.opts;
    if (!isFinite(pt.x) || !isFinite(pt.y) || !isFinite(pt.z)) { map.rejected++; return -1; }

    var dx = camPos.x - pt.x, dy = camPos.y - pt.y, dz = camPos.z - pt.z;
    var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(dist >= o.near && dist <= o.far)) { map.rejected++; return -1; }
    var ux = dx / dist, uy = dy / dist, uz = dz / dist;   // surfel -> camera

    // Grazing observations carry almost no usable texture, so they must not be
    // allowed to mark a surface covered.
    if (normal) {
      var nd = normal.x * ux + normal.y * uy + normal.z * uz;
      if (nd < 0) { nd = -nd; }                 // normal sign is unreliable
      if (nd < o.cosMaxIncidence) { map.rejected++; return -1; }
    }

    var v = o.voxel;
    var key = Math.round(pt.x / v) + "," + Math.round(pt.y / v) + "," + Math.round(pt.z / v);
    map.clock++;

    var bin = dirBin(o, ux, uy, uz);
    var bit = (1 << bin) | 0;
    var idx = map.voxels.get(key);

    if (idx === undefined) {
      if (map.surfels.length >= o.maxSurfels) { map.full = true; map.rejected++; return -1; }
      idx = map.surfels.length;
      map.surfels.push({
        x: pt.x, y: pt.y, z: pt.z, key: key,
        dirs: 0, goodDirs: 0,       // all directions vs. in-range directions
        seen: 0, last: map.clock,
        bestDist: Infinity,
        sx: 0, sy: 0, sz: 0,        // running sum of view directions
        nx: ux, ny: uy, nz: uz      // normal estimate (mean view dir, or given)
      });
      map.voxels.set(key, idx);
    }

    var s = map.surfels[idx];
    s.seen++;
    s.last = map.clock;
    s.dirs |= bit;
    if (dist <= o.goodDist) s.goodDirs |= bit;
    if (dist < s.bestDist) s.bestDist = dist;
    s.sx += ux; s.sy += uy; s.sz += uz;

    if (normal) {
      s.nx = normal.x; s.ny = normal.y; s.nz = normal.z;
    } else {
      var sl = Math.sqrt(s.sx * s.sx + s.sy * s.sy + s.sz * s.sz) || 1;
      s.nx = s.sx / sl; s.ny = s.sy / sl; s.nz = s.sz / sl;
    }
    return idx;
  }

  // project: world surfels -> view pixels, with coverage state per dot.
  // cam = {pos:{x,y,z}, quat:{x,y,z,w}} camera-to-world.
  // fx/fy are normalized focal terms and cxOff/cyOff the principal-point skew:
  //   ndc_x = fx * X / -Z + cxOff,  ndc_y = fy * Y / -Z + cyOff
  // For a WebXR view read all four straight off the column-major projection
  // matrix: fx=P[0], fy=P[5], cxOff=P[8], cyOff=P[9]. Phone AR frusta are
  // usually off-centre, so dropping the skew terms puts every dot a few percent
  // off across the frame. cxOff/cyOff are optional and default to 0.
  // Returns [{px, py, nx, ny, state, dirs, seen, dist, idx}]; px/py in frame
  // pixels, nx/ny in normalized view coords (0..1, top-left origin) for
  // sampling a depth buffer at the dot.
  function project(map, cam, frameW, frameH, fx, fy, cxOff, cyOff) {
    var o = map.opts;
    var q = cam.quat;
    var ox = cxOff || 0, oy = cyOff || 0;
    // world -> camera is the conjugate rotation
    var cx = -q.x, cy = -q.y, cz = -q.z, cw = q.w;
    var out = [];
    for (var i = 0; i < map.surfels.length; i++) {
      var s = map.surfels[i];
      var vx = s.x - cam.pos.x, vy = s.y - cam.pos.y, vz = s.z - cam.pos.z;

      // backface cull: is the camera on the visible side of this surfel?
      var toCam = -(vx * s.nx + vy * s.ny + vz * s.nz);
      if (toCam <= 0) continue;

      // rotate (world delta) into camera space
      var tx = 2 * (cy * vz - cz * vy);
      var ty = 2 * (cz * vx - cx * vz);
      var tz = 2 * (cx * vy - cy * vx);
      var ex = vx + cw * tx + (cy * tz - cz * ty);
      var ey = vy + cw * ty + (cz * tx - cx * tz);
      var ez = vz + cw * tz + (cx * ty - cy * tx);

      if (ez > -o.near) continue;                 // behind or too close
      var d = -ez;
      if (d > o.far) continue;
      var nx = (ex / d) * fx + ox, ny = (ey / d) * fy + oy;
      if (nx < -o.margin || nx > o.margin || ny < -o.margin || ny > o.margin) continue;

      out.push({
        px: (nx * 0.5 + 0.5) * frameW,
        py: (1 - (ny * 0.5 + 0.5)) * frameH,   // NDC +Y is up, pixels grow down
        nx: nx * 0.5 + 0.5,                    // normalized view coords, for
        ny: 1 - (ny * 0.5 + 0.5),              // sampling a depth buffer
        state: stateOf(map, s), dirs: popcount32(s.goodDirs),
        seen: s.seen, dist: d, idx: i
      });
    }
    return out;
  }

  // unprojectDepth: one WebXR depth sample -> a world point. The inverse of
  // project(), and the single most error-prone step in the whole feature, so it
  // lives here where the tests can round-trip it.
  //
  //   m       column-major 4x4 camera-to-world matrix (XRView.transform.matrix)
  //   nxv,nyv normalized VIEW coords as getDepthInMeters() takes them:
  //           [0,1], origin TOP-LEFT, +X right, +Y DOWN
  //   zDepth  the value getDepthInMeters() returned. Per the depth-sensing
  //           spec this is the perpendicular distance from the camera PLANE,
  //           explicitly not the length of the ray, so it is |Z| in view space
  //           and needs no trigonometry to use.
  //   fx,fy,cxOff,cyOff  projection terms, as in project()
  // Writes into `out` and returns it (or a fresh object). Returns null when the
  // sample is unusable -- the spec says invalid depth pixels are exactly 0.
  function unprojectDepth(m, nxv, nyv, zDepth, fx, fy, cxOff, cyOff, out) {
    if (!(zDepth > 0) || !isFinite(zDepth)) return null;
    var ndcX = 2 * nxv - 1;
    var ndcY = 1 - 2 * nyv;                    // nyv grows downward, NDC +Y is up
    var X = (ndcX - (cxOff || 0)) * zDepth / fx;
    var Y = (ndcY - (cyOff || 0)) * zDepth / fy;
    var Z = -zDepth;
    var o = out || { x: 0, y: 0, z: 0 };
    o.x = m[0] * X + m[4] * Y + m[8] * Z + m[12];
    o.y = m[1] * X + m[5] * Y + m[9] * Z + m[13];
    o.z = m[2] * X + m[6] * Y + m[10] * Z + m[14];
    return o;
  }

  function stats(map) {
    var thin = 0, partial = 0, covered = 0;
    for (var i = 0; i < map.surfels.length; i++) {
      var st = stateOf(map, map.surfels[i]);
      if (st === STATE_COVERED) covered++;
      else if (st === STATE_PARTIAL) partial++;
      else thin++;
    }
    var total = map.surfels.length;
    var v = map.opts.voxel;
    return {
      total: total, thin: thin, partial: partial, covered: covered,
      coveredPct: total ? (covered / total) * 100 : 0,
      // surfels are voxel-quantized, so each stands for ~voxel^2 of surface
      coveredArea: covered * v * v,
      thinArea: thin * v * v,
      full: map.full, rejected: map.rejected
    };
  }

  // gapClusters: group surfels that still need angles into spatial blobs, so
  // the end-of-take report can say WHERE to go back to rather than just how
  // much is missing. Grid-bucket flood fill over 26-connected coarse cells.
  // Returns [{cx,cy,cz, count, area, minY, maxY}] largest first.
  function gapClusters(map, cellSize, minCount) {
    var cell = cellSize || 0.5;
    var minC = minCount || 6;
    var buckets = new Map();
    var i, k;

    for (i = 0; i < map.surfels.length; i++) {
      var s = map.surfels[i];
      if (stateOf(map, s) === STATE_COVERED) continue;
      k = Math.round(s.x / cell) + "," + Math.round(s.y / cell) + "," + Math.round(s.z / cell);
      var b = buckets.get(k);
      if (!b) { b = []; buckets.set(k, b); }
      b.push(i);
    }

    var seen = new Set(), clusters = [];
    buckets.forEach(function (_, startKey) {
      if (seen.has(startKey)) return;
      var stack = [startKey], members = [];
      seen.add(startKey);
      while (stack.length) {
        var ck = stack.pop();
        var arr = buckets.get(ck);
        if (arr) for (var m = 0; m < arr.length; m++) members.push(arr[m]);
        var p = ck.split(",");
        var ci = +p[0], cj = +p[1], ck2 = +p[2];
        for (var dx = -1; dx <= 1; dx++)
          for (var dy = -1; dy <= 1; dy++)
            for (var dz = -1; dz <= 1; dz++) {
              if (!dx && !dy && !dz) continue;
              var nk = (ci + dx) + "," + (cj + dy) + "," + (ck2 + dz);
              if (buckets.has(nk) && !seen.has(nk)) { seen.add(nk); stack.push(nk); }
            }
      }
      if (members.length < minC) return;
      var sx = 0, sy = 0, sz = 0, lo = Infinity, hi = -Infinity;
      for (var n = 0; n < members.length; n++) {
        var su = map.surfels[members[n]];
        sx += su.x; sy += su.y; sz += su.z;
        if (su.y < lo) lo = su.y;
        if (su.y > hi) hi = su.y;
      }
      var c = members.length, v = map.opts.voxel;
      clusters.push({
        cx: sx / c, cy: sy / c, cz: sz / c,
        count: c, area: c * v * v, minY: lo, maxY: hi
      });
    });

    clusters.sort(function (a, b) { return b.count - a.count; });
    return clusters;
  }

  // Human-readable bearing from an origin to a cluster, for the report text.
  function bearingTo(fromX, fromZ, toX, toZ) {
    // World is Y-up, -Z forward (WebXR local-floor): bearing 0 = -Z.
    var deg = (Math.atan2(toX - fromX, -(toZ - fromZ)) * 180 / Math.PI + 360) % 360;
    var names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    return { deg: deg, name: names[Math.round(deg / 45) % 8],
             dist: Math.sqrt((toX - fromX) * (toX - fromX) + (toZ - fromZ) * (toZ - fromZ)) };
  }

  return {
    DEFAULTS: DEFAULTS,
    STATE_THIN: STATE_THIN, STATE_PARTIAL: STATE_PARTIAL, STATE_COVERED: STATE_COVERED,
    createMap: createMap, clear: clear, observe: observe, project: project,
    unprojectDepth: unprojectDepth,
    stateOf: stateOf, stats: stats, gapClusters: gapClusters,
    dirBin: dirBin, popcount32: popcount32, bearingTo: bearingTo
  };
})();
if (typeof window !== "undefined") window.CoverageMap = CoverageMap;
if (typeof module !== "undefined" && module.exports) module.exports = CoverageMap;
