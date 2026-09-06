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
    minBaselineDeg: 25,   // ... or this much angle between the two extreme views
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

  // Angle in degrees between two unit directions.
  function angDeg(ax, ay, az, bx, by, bz) {
    var d = ax * bx + ay * by + az * bz;
    if (d > 1) d = 1; else if (d < -1) d = -1;
    return Math.acos(d) * 180 / Math.PI;
  }

  function stateOf(map, s) {
    var o = map.opts, n = popcount32(s.goodDirs);
    if (n >= o.minDirsCovered) return STATE_COVERED;
    // A wide baseline from only two stops is enough: it is the angle between
    // viewpoints that lets COLMAP triangulate, not the number of compass sectors
    // visited. Without this a single-sided surface -- a wardrobe front, a wall
    // against a neighbour, one side of a stairwell -- can never finish, because
    // the directions it cannot be seen from simply do not exist.
    if (s.baseDeg >= o.minBaselineDeg) return STATE_COVERED;
    if (n >= o.minDirsPartial || s.baseDeg >= o.minBaselineDeg * 0.5) return STATE_PARTIAL;
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
    // allowed to mark a surface covered. The incidence test ignores which way the
    // normal points along its own axis; the sign is still needed for the backface
    // cull in project(), so it is settled once here rather than guessed later.
    var sgn = 1;
    if (normal) {
      var nd = normal.x * ux + normal.y * uy + normal.z * uz;
      if (nd < 0) { nd = -nd; sgn = -1; }
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
        nx: ux, ny: uy, nz: uz,     // normal estimate (mean view dir, or given)
        px: 0, py: 0, pz: 0,        // extreme viewpoint P \
        qx: 0, qy: 0, qz: 0,        // extreme viewpoint Q  > widest pair so far
        baseDeg: -1                 // angle between them; -1 = no usable view yet
      });
      map.voxels.set(key, idx);
    }

    var s = map.surfels[idx];
    s.seen++;
    s.last = map.clock;
    s.dirs |= bit;
    if (dist <= o.goodDist) {
      s.goodDirs |= bit;
      // u points from the surface at the camera, so the angle between two u's is
      // the angle those two stops subtend at the surface -- the baseline that
      // lets COLMAP triangulate. Only in-range views may open it.
      if (s.baseDeg < 0) {
        s.px = ux; s.py = uy; s.pz = uz; s.baseDeg = 0;
      } else {
        var dP = angDeg(ux, uy, uz, s.px, s.py, s.pz);
        if (dP > s.baseDeg) {
          s.qx = ux; s.qy = uy; s.qz = uz; s.baseDeg = dP;          // pair (P,u)
        } else if (s.baseDeg > 0) {
          var dQ = angDeg(ux, uy, uz, s.qx, s.qy, s.qz);
          if (dQ > s.baseDeg) { s.px = ux; s.py = uy; s.pz = uz; s.baseDeg = dQ; }
        }
      }
    }
    if (dist < s.bestDist) s.bestDist = dist;
    s.sx += ux; s.sy += uy; s.sz += uz;

    if (normal) {
      s.nx = normal.x * sgn; s.ny = normal.y * sgn; s.nz = normal.z * sgn;
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
        state: stateOf(map, s), dirs: popcount32(s.goodDirs), base: s.baseDeg,
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

  // projectPacked: project() with the allocations taken out. The capture page
  // calls a projection every frame for the overlay and again for the dot view,
  // and at 30 000 surfels that was tens of thousands of short-lived objects a
  // second on a phone that has to garbage collect them mid-take. Five floats per
  // visible surfel go into a buffer the caller owns instead:
  //   [nx, ny, dist, state, surfelIndex]
  // Same order, same culling, same numbers -- `project()` above stays the
  // readable reference, and the tests hold the two of them equal. The caller owns
  // the buffer and must size it at surfels * PACK_STRIDE; the count written is the
  // return value.
  var PACK_STRIDE = 5;
  function projectPacked(map, cam, fx, fy, cxOff, cyOff, out) {
    var o = map.opts, s = map.surfels, q = cam.quat;
    var ox = cxOff || 0, oy = cyOff || 0;
    var cx = -q.x, cy = -q.y, cz = -q.z, cw = q.w;
    var px = cam.pos.x, py = cam.pos.y, pz = cam.pos.z;
    var near = o.near, far = o.far, margin = o.margin;
    var k = 0;
    if (!out || out.length < s.length * PACK_STRIDE) return -1;
    for (var i = 0; i < s.length; i++) {
      var su = s[i];
      var vx = su.x - px, vy = su.y - py, vz = su.z - pz;
      var toCam = -(vx * su.nx + vy * su.ny + vz * su.nz);
      if (toCam <= 0) continue;
      var tx = 2 * (cy * vz - cz * vy);
      var ty = 2 * (cz * vx - cx * vz);
      var tz = 2 * (cx * vy - cy * vx);
      var ex = vx + cw * tx + (cy * tz - cz * ty);
      var ey = vy + cw * ty + (cz * tx - cx * tz);
      var ez = vz + cw * tz + (cx * ty - cy * tx);
      if (ez > -near) continue;
      var d = -ez;
      if (d > far) continue;
      var nx = (ex / d) * fx + ox, ny = (ey / d) * fy + oy;
      if (nx < -margin || nx > margin || ny < -margin || ny > margin) continue;
      out[k] = nx * 0.5 + 0.5;               // normalized view coords, top-left
      out[k + 1] = 1 - (ny * 0.5 + 0.5);     // origin, exactly as project() reports
      out[k + 2] = d;
      out[k + 3] = stateOf(map, su); out[k + 4] = i;
      k += PACK_STRIDE;
    }
    return k / PACK_STRIDE;
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

  // =====================================================================
  // ROOMS — the floor, the ceiling and the walls
  //
  // WHY THIS IS NOT JUST "DIM THE FLAT DOTS"
  //   The plan view used to dim dots whose stored normal looked horizontal and
  //   that was the whole of it. That asks a per-surfel estimate -- the cross
  //   product of three depth taps 10 cm apart -- to answer a question about the
  //   room, and on a real ceiling that estimate is mostly noise: nothing was ever
  //   learned about how tall the space is, or whether the ceiling has been looked
  //   at at all. So the evidence is pooled instead: one pass over the map, heights
  //   binned into a histogram, and a normal signed to face the camera rather than
  //   left as an unsigned tilt.
  //
  // THE TEST THAT SEPARATES A WARDROBE TOP FROM A CEILING
  //   Both are horizontal. The ceiling faces DOWN at you, the wardrobe top faces
  //   UP. Flipping every normal to point at the camera and then reading its
  //   vertical component sorts them, which no threshold on "how flat is it" can.
  //
  //   Where the handset has its own plane finder (WebXR plane-detection, which
  //   carries a semantic floor/ceiling/wall label on ARCore builds) that wins
  //   outright: it is built from every frame the phone ever saw, not from this
  //   page's handful of taps per frame. The fit below is what runs when the phone
  //   has nothing to say, which is the common case.
  // =====================================================================
  var ROOM_DEFAULTS = {
    yBin: 0.05,          // height resolution of the slab histogram (m)
    minSlab: 18,         // surfels a level surface needs before it counts (~0.18 m^2)
    horizCos: 0.927,     // |ny| above this is level   (within 22 deg of vertical)
    vertCos: 0.374,      // |ny| below this is plumb    (within 68 deg of vertical)
    merge: 0.30,         // histogram bins this close are one surface, not two
    wallBins: 36,        // wall normal azimuth bins over 180 deg (5 deg each)
    wallDBin: 0.10,      // wall offset bins (m)
    minWall: 25,         // ... and the same idea for a wall
    maxWalls: 8,
    sample: 4000,        // surfels read per fit; this runs once a second
    minRoomH: 1.85,      // headroom a "room" needs; below that it is a table and a floor
    maxRoomH: 5.5,
    levelMerge: 0.15     // phone patches this close in height are one level surface
  };

  function createRoom(opts) {
    var o = {}, k;
    for (k in ROOM_DEFAULTS) o[k] = ROOM_DEFAULTS[k];
    if (opts) for (k in opts) o[k] = opts[k];
    return {
      opts: o,
      measured: { floor: null, ceiling: null, walls: [], slabs: 0, plumbs: 0 },
      phone: { floor: null, ceiling: null, walls: [], planes: 0, roomCapture: false },
      // running totals for the summary
      footprint: 0,        // resolved: the larger of the two below
      bboxFoot: 0,         // horizontal bounding box of what has been scanned
      phoneFoot: 0,        // a floor polygon the phone actually measured
      summary: null,
      clock: 0
    };
  }

  function clearRoom(room) {
    room.measured = { floor: null, ceiling: null, walls: [], slabs: 0, plumbs: 0 };
    room.phone = { floor: null, ceiling: null, walls: [], planes: 0, roomCapture: false };
    room.footprint = 0; room.bboxFoot = 0; room.phoneFoot = 0;
    room.summary = null;
  }

  // Group a Map<binIndex, {n,...>} into surfaces. Adjacent bins merge, because a
  // wall measured twice at 10 cm resolution lands in neighbouring height bins, and
  // reporting both would invent a second floor 5 cm below the first.
  function slabPeaks(bins, yBin, merge, minSupport) {
    var arr = [];
    bins.forEach(function (v, k) { arr.push({ k: k, n: v.n, wy: v.wy }); });
    arr.sort(function (a, b) { return a.k - b.k; });
    var out = [], cur = null;
    for (var i = 0; i < arr.length; i++) {
      var a = arr[i];
      if (cur && (a.k - cur.lastK) * yBin <= merge) {
        cur.n += a.n; cur.wy += a.wy; cur.lastK = a.k;
      } else {
        if (cur && cur.n >= minSupport)
          out.push({ y: cur.wy / cur.n, support: cur.n });
        cur = { n: a.n, wy: a.wy, lastK: a.k };
      }
    }
    if (cur && cur.n >= minSupport) out.push({ y: cur.wy / cur.n, support: cur.n });
    out.sort(function (a, b) { return b.support - a.support; });
    return out;
  }

  // Smallest angular gap between two plane azimuths in degrees. A plane is the
  // same plane seen from either side, so 175 deg and 4 deg are 9 deg apart.
  function azGapDeg(a, b) {
    var d = Math.abs(a - b) % 180;
    return d > 90 ? 180 - d : d;
  }

  // fitRoom: one pooled pass over the surfels. `bbox` is the horizontal extent of
  // the scan so far, used as the denominator for "how much of the ceiling have I
  // actually looked at" when the phone has not measured a floor polygon for us.
  // `refY` is the median camera height over the take; without it the level
  // surfaces get judged against the phone's height at this instant, which is
  // 0.5 m while you look down at the floor and up near the ceiling while you look
  // up -- and the room's shape is not allowed to depend on where you pointed.
  function fitRoom(room, map, camPos, bbox, refY) {
    var o = room.opts, S = map.surfels, v = map.opts.voxel;
    if (!S.length || !camPos) return room;
    var up = new Map(), down = new Map(), walls = new Map();
    var stride = Math.max(1, Math.ceil(S.length / o.sample));
    var camX = camPos.x, camZ = camPos.z, eyeY = camPos.y;
    var camY = Number.isFinite(refY) ? refY : camPos.y;   // Number., not isFinite: null is 0 to that
    var slabs = 0, plumbs = 0;

    for (var i = 0; i < S.length; i += stride) {
      var s = S[i];
      // Sign the normal toward the camera. observe() keeps whatever winding the
      // depth grid produced, so "which way does this face, relative to me" is the
      // question we can actually answer.
      var ny = s.ny;
      if (ny * (eyeY - s.y) < 0) ny = -ny;
      var an = ny < 0 ? -ny : ny;

      if (an >= o.horizCos) {
        var b = Math.round(s.y / o.yBin);
        var m = ny > 0 ? up : down;
        var e = m.get(b);
        if (!e) { e = { n: 0, wy: 0 }; m.set(b, e); }
        e.n++; e.wy += b * o.yBin;
        slabs++;
      } else if (an <= o.vertCos) {
        var hx = camX - s.x, hz = camZ - s.z;
        var nx = s.nx, nz = s.nz;
        if (nx * hx + nz * hz < 0) { nx = -nx; nz = -nz; }
        var hl = Math.sqrt(nx * nx + nz * nz);
        if (hl < 0.2) continue;              // normal too flat to name a direction
        nx /= hl; nz /= hl;
        // Canonicalise: the same wall seen from inside and outside the room must
        // land in one bin, so fold the azimuth into 0..180 and flip with it.
        if (nz < 0 || (nz === 0 && nx < 0)) { nx = -nx; nz = -nz; }
        var az = Math.atan2(nx, nz);
        if (az < 0) az += Math.PI;
        var ai = Math.floor(az / Math.PI * o.wallBins) % o.wallBins;
        var d = (nx * s.x + nz * s.z) / o.wallDBin;
        var di = Math.round(d) + 2048;
        if (di < 0 || di > 4095) continue;
        var key = ai * 4096 + di;
        var w = walls.get(key);
        if (!w) {
          w = { n: 0, nx: 0, nz: 0, x: 0, z: 0,
                x0: s.x, x1: s.x, z0: s.z, z1: s.z, y0: s.y, y1: s.y };
          walls.set(key, w);
        }
        w.n++; w.nx += nx; w.nz += nz; w.x += s.x; w.z += s.z;
        if (s.x < w.x0) w.x0 = s.x; if (s.x > w.x1) w.x1 = s.x;
        if (s.z < w.z0) w.z0 = s.z; if (s.z > w.z1) w.z1 = s.z;
        if (s.y < w.y0) w.y0 = s.y; if (s.y > w.y1) w.y1 = s.y;
        plumbs++;
      }
    }

    // Floor: a level surface that faces up and sits under the phone. Below the
    // camera is not enough -- the top of a cupboard faces up too -- and neither is
    // support, because a take spent staring at furniture produces more cupboard
    // surfels than floor surfels. So the strongest candidate sets a support bar and
    // the LOWEST surface clearing half of it wins.
    var fp = slabPeaks(up, o.yBin, o.merge, o.minSlab), below = [];
    for (i = 0; i < fp.length; i++)
      if (fp[i].y < camY - 0.20) below.push(fp[i]);
    var floor = null;
    if (below.length) {
      floor = below[0];
      var bar = below[0].support * 0.5;
      for (i = 1; i < below.length; i++)
        if (below[i].support >= bar && below[i].y < floor.y) floor = below[i];
    }
    // Ceiling: a level surface that faces DOWN, above the phone, busiest one.
    var cp = slabPeaks(down, o.yBin, o.merge, o.minSlab);
    var ceiling = null;
    for (i = 0; i < cp.length; i++)
      if (cp[i].y > camY + 0.15 && (!ceiling || cp[i].support > ceiling.support))
        ceiling = cp[i];
    // A "room" with no vertical extent left in it is a table and a floor.
    if (ceiling && (!floor || ceiling.y - floor.y < o.minRoomH ||
                    ceiling.y - floor.y > o.maxRoomH)) ceiling = null;

    // Walls, strongest first, folded into one plane per face.
    var cand = [];
    walls.forEach(function (w) {
      if (w.n < o.minWall) return;
      var nx = w.nx / w.n, nz = w.nz / w.n, nl = Math.sqrt(nx * nx + nz * nz) || 1;
      nx /= nl; nz /= nl;
      var az = Math.atan2(nx, nz); if (az < 0) az += Math.PI;
      // Offset through the centroid of the samples, not the bin's nominal value:
      // two bins that are really one wall then agree to within a few millimetres
      // and merge, while a parallel wall 40 cm away still does not.
      cand.push({
        nx: nx, nz: nz, deg: az * 180 / Math.PI,
        d: nx * (w.x / w.n) + nz * (w.z / w.n),
        support: w.n, area: w.n * v * v,
        span: Math.max(w.x1 - w.x0, w.z1 - w.z0), y0: w.y0, y1: w.y1
      });
    });
    cand.sort(function (a, b) { return b.support - a.support; });
    var acc = [];
    for (i = 0; i < cand.length && acc.length < o.maxWalls; i++) {
      var c = cand[i], dup = -1;
      for (var j = 0; j < acc.length; j++) {
        var w2 = acc[j];
        if (azGapDeg(c.deg, w2.deg) >= 15) continue;
        // n and -n describe the same plane, so the offset gap only means
        // something once the two normals face the same way. Comparing d against
        // -d for opposed normals is what keeps one wall from being reported
        // twice; comparing d against d is what keeps the two opposite walls of a
        // 6 m room from being merged into one at the centre.
        var opposed = c.nx * w2.nx + c.nz * w2.nz < 0;
        var gap = opposed ? Math.abs(c.d + w2.d) : Math.abs(c.d - w2.d);
        if (gap < 0.3) { dup = j; break; }
      }
      if (dup >= 0) continue;                 // already named, and weaker
      acc.push(c);
    }

    room.measured = { floor: floor, ceiling: ceiling, walls: acc,
                      slabs: slabs, plumbs: plumbs };
    room.clock++;
    if (bbox)
      room.bboxFoot = Math.max(0.25, bbox.x1 - bbox.x0) * Math.max(0.25, bbox.z1 - bbox.z0);
    return room;
  }

  // setPhonePlanes: feed in what the handset's own plane finder reported, already
  // turned into world coordinates by the caller (this stays free of WebXR types).
  // Each entry: {kind:"level"|"floor"|"ceiling"|"wall", label, y, nx, nz, d, area,
  //              cx, cy, cz, poly:[x,y,z,...]}
  //
  // Level planes are NOT taken at their word for floor or ceiling. A phone's room
  // capture reports one polygon per patch it has held in view, so a single ceiling
  // arrives as four records at 1.79 / 2.05 / 2.13 m, and choosing the biggest floor
  // and the biggest ceiling independently — as this function used to — paired a
  // step with a bulkhead and reported a real 2.3 m room as 1.60 m tall. What the
  // phone says a level *is* (its semanticLabel) still outranks all of this.
  function setPhonePlanes(room, planes, roomCapture, refY) {
    var o = room.opts;
    var p = { floor: null, ceiling: null, walls: [], planes: 0,
              roomCapture: !!roomCapture };
    var levels = [], i, j;

    for (i = 0; i < planes.length; i++) {
      var q = planes[i];
      q.poly = q.poly || null;
      p.planes++;
      if (q.kind === "wall") { p.walls.push(q); continue; }
      if (q.kind !== "level" && q.kind !== "floor" && q.kind !== "ceiling") continue;
      var lab = String(q.label || "").toLowerCase();
      var says = /floor|ground/.test(lab) ? "floor" : (/ceil/.test(lab) ? "ceiling" : null);
      var lv = null;
      for (j = 0; j < levels.length; j++)
        if (Math.abs(levels[j].y0 - q.y) <= o.levelMerge) { lv = levels[j]; break; }
      if (!lv) {
        lv = { y: q.y, y0: q.y, y1: q.y, wy: 0, ww: 0, area: 0, n: 0,
               says: null, best: null, x0: 0, x1: 0, z0: 0, z1: 0, box: false };
        levels.push(lv);
      }
      lv.y0 = Math.min(lv.y0, q.y); lv.y1 = Math.max(lv.y1, q.y);
      var w = (q.area || 0) + 1;      // area-weighted: a patch of floor is mostly its polygon
      lv.wy += q.y * w; lv.ww += w; lv.area += q.area || 0; lv.n++;
      if (says && !lv.says) lv.says = says;
      if (!lv.best || (q.area || 0) > (lv.best.area || 0)) lv.best = q;
      // The footprint, because a floor and a ceiling only make a room if one of
      // them is above the other.
      for (var pI = 0; q.poly && pI + 2 < q.poly.length; pI += 3) {
        var px = q.poly[pI], pz = q.poly[pI + 2];
        if (!lv.box) { lv.box = true; lv.x0 = lv.x1 = px; lv.z0 = lv.z1 = pz; }
        if (px < lv.x0) lv.x0 = px; if (px > lv.x1) lv.x1 = px;
        if (pz < lv.z0) lv.z0 = pz; if (pz > lv.z1) lv.z1 = pz;
      }
    }
    for (i = 0; i < levels.length; i++)
      if (levels[i].ww > 0) levels[i].y = levels[i].wy / levels[i].ww;
    levels.sort(function (a, b) { return a.y - b.y; });

    // Best floor/ceiling pairing over the merged levels. How much of one lies
    // above the other is the evidence, a phone-supplied label is worth far more
    // than any area, and the operator's median height is the tie-breaker between
    // two equally plausible rooms.
    var pair = null;
    var hasRef = Number.isFinite(refY);       // isFinite(null) is true; Number.isFinite is not
    for (i = 0; i < levels.length; i++) {
      if (levels[i].says === "ceiling") continue;
      for (j = levels.length - 1; j > i; j--) {
        var f = levels[i], c = levels[j], h = c.y - f.y;
        if (c.says === "floor" || h < o.minRoomH || h > o.maxRoomH) continue;
        var ev = f.area < c.area ? f.area : c.area;
        if (f.box && c.box) {
          var ox = Math.min(f.x1, c.x1) - Math.max(f.x0, c.x0);
          var oz = Math.min(f.z1, c.z1) - Math.max(f.z0, c.z0);
          if (ox <= 0 || oz <= 0) continue;       // one is nowhere over the other
          ev = ox * oz;
        }
        var score = ev;
        if (f.says === "floor") score += 1e4;
        if (c.says === "ceiling") score += 1e4;
        if (hasRef) { if (f.y < refY) score += 500; if (c.y > refY) score += 500; }
        var better = !pair || score > pair.score + 1e-6 ||
          (Math.abs(score - pair.score) <= 1e-6 && h < pair.h - 1e-6);
        if (better) pair = { f: f, c: c, score: score, h: h };
      }
    }
    if (pair) { p.floor = levelRecord(pair.f); p.ceiling = levelRecord(pair.c); }
    else {
      // No pair has real headroom in it. Report the single strongest level for
      // what it is rather than inventing a room out of a table and a floor.
      var top = null;
      for (i = 0; i < levels.length; i++)
        if (!top || levels[i].area > top.area) top = levels[i];
      if (top) {
        var rec = levelRecord(top);
        if (top.says === "ceiling" || (top.says === null && hasRef && top.y > refY))
          p.ceiling = rec;
        else p.floor = rec;
      }
    }

    room.phone = p;
    // A plane polygon only covers what the phone managed to see, so this is a
    // lower bound on the room's footprint, not the footprint itself.
    if (p.floor && p.floor.area) room.phoneFoot = Math.max(room.phoneFoot, p.floor.area);
    return room;
  }

  // One merged level as a plane record: the strongest polygon carries the shape
  // and the pose, the level as a whole carries the height and the area.
  function levelRecord(lv) {
    var r = {}, k;
    for (k in lv.best) r[k] = lv.best[k];
    r.y = lv.y; r.area = lv.area; r.count = lv.n;
    return r;
  }

  // roomSummary: the one merged answer, phone evidence in charge where it exists.
  // `source` is what the operator gets told -- a height the phone measured from a
  // whole session of tracking is worth a different sentence than one this page
  // guessed from a histogram of taps.
  function roomSummary(room, map) {
    var v = map.opts.voxel, m = room.measured, ph = room.phone;
    var floor = null, ceiling = null, walls = m.walls, src = "none";
    var floorArea = 0, ceilingArea = 0;
    if (m.floor) { floor = m.floor.y; floorArea = m.floor.support * v * v; }
    if (m.ceiling) { ceiling = m.ceiling.y; ceilingArea = m.ceiling.support * v * v; }
    var phoneSaid = false;
    if (ph.floor) { floor = ph.floor.y; floorArea = ph.floor.area || floorArea; phoneSaid = true; }
    if (ph.ceiling) { ceiling = ph.ceiling.y; ceilingArea = ph.ceiling.area || ceilingArea; phoneSaid = true; }
    if (ph.walls.length) { walls = ph.walls; phoneSaid = true; }
    if (phoneSaid && (m.floor || m.ceiling || m.walls.length)) src = "both";
    else if (phoneSaid) src = "phone";
    else if (floor !== null || ceiling !== null || walls.length) src = "measured";

    var height = (floor !== null && ceiling !== null) ? ceiling - floor : null;
    // Headroom is a property of the room, not of which source was read last. Each
    // one is consistent on its own, but a phone floor under this page's idea of a
    // ceiling can still add up to a table and a shelf, so say there is no ceiling
    // rather than publish a height nobody could stand up in.
    if (height !== null && (height < room.opts.minRoomH || height > room.opts.maxRoomH)) {
      ceiling = null; ceilingArea = 0; height = null;
    }
    // The denominator for "how much of the ceiling have I looked at". Either
    // estimate understates the room -- a bounding box only spans what has been
    // scanned, a plane polygon only what the phone saw -- so take the larger.
    room.footprint = Math.max(room.bboxFoot, room.phoneFoot);
    var foot = room.footprint || Math.max(floorArea, ceilingArea, 1);
    var spans = wallSpans(walls);
    var out = {
      floorY: floor, ceilingY: ceiling, height: height,
      walls: walls, wallCount: walls.length,
      floorArea: floorArea, ceilingArea: ceilingArea,
      footprint: foot,
      floorSeen: floor !== null ? Math.min(1, floorArea / foot) : 0,
      ceilingSeen: ceiling !== null ? Math.min(1, ceilingArea / foot) : 0,
      spanA: spans.a, spanB: spans.b,
      source: src,
      measuredFloor: m.floor ? m.floor.y : null,
      measuredCeiling: m.ceiling ? m.ceiling.y : null,
      slabs: m.slabs, plumbs: m.plumbs, phonePlanes: ph.planes,
      roomCapture: ph.roomCapture
    };
    room.summary = out;
    return out;
  }

  // Two numbers an operator can check with a tape measure: how far apart the
  // opposite walls are. Walls are binned by azimuth, so group by face direction
  // and take the widest spread within each of the two dominant groups.
  function wallSpans(walls) {
    var groups = [];
    for (var i = 0; i < walls.length; i++) {
      var w = walls[i], g = null;
      for (var j = 0; j < groups.length; j++)
        if (azGapDeg(w.deg, groups[j].deg) < 25) { g = groups[j]; break; }
      if (!g) { g = { deg: w.deg, nx: w.nx, nz: w.nz, ds: [] }; groups.push(g); }
      // Measured against the group's own reference normal: a face seen from both
      // sides arrives as n and -n, and subtracting those two offsets would report
      // the room as twice as wide as it is.
      g.ds.push((w.nx * g.nx + w.nz * g.nz < 0) ? -w.d : w.d);
    }
    groups.sort(function (a, b) { return b.ds.length - a.ds.length; });
    function spread(gr) {
      if (!gr) return null;
      var lo = gr.ds[0], hi = gr.ds[0];
      for (var k = 1; k < gr.ds.length; k++) {
        if (gr.ds[k] < lo) lo = gr.ds[k];
        if (gr.ds[k] > hi) hi = gr.ds[k];
      }
      return hi - lo;
    }
    return { a: spread(groups[0]) || null, b: spread(groups[1]) || null };
  }

  // gapClusters: group surfels that still need angles into spatial blobs, so
  // the end-of-take report can say WHERE to go back to rather than just how
  // much is missing. Grid-bucket flood fill over 26-connected coarse cells.
  // Returns [{cx,cy,cz, count, area, minY, maxY}] largest first.
  //
  // Cells are keyed as one packed integer rather than an "i,j,k" string: this
  // runs once a second inside the same tick that paints the live plan view, and
  // the string keys were the most expensive thing the HUD did.
  var GAP_OFF = 4096, GAP_STRIDE = 8192;      // +/-4096 cells: 2.5 km at 0.6 m
  function gapKey(i, j, k) {
    return ((i + GAP_OFF) * GAP_STRIDE + (j + GAP_OFF)) * GAP_STRIDE + (k + GAP_OFF);
  }
  var GAP_NEIGHBOURS = (function () {
    var out = [], s2 = GAP_STRIDE * GAP_STRIDE;
    for (var dx = -1; dx <= 1; dx++)
      for (var dy = -1; dy <= 1; dy++)
        for (var dz = -1; dz <= 1; dz++)
          if (dx || dy || dz) out.push(dx * s2 + dy * GAP_STRIDE + dz);
    return out;
  })();

  function gapClusters(map, cellSize, minCount) {
    var cell = cellSize || 0.5;
    var minC = minCount || 6;
    var buckets = new Map();
    var i;

    for (i = 0; i < map.surfels.length; i++) {
      var s = map.surfels[i];
      if (stateOf(map, s) === STATE_COVERED) continue;
      var k = gapKey(Math.round(s.x / cell), Math.round(s.y / cell), Math.round(s.z / cell));
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
        for (var n = 0; n < GAP_NEIGHBOURS.length; n++) {
          var nk = ck + GAP_NEIGHBOURS[n];
          if (buckets.has(nk) && !seen.has(nk)) { seen.add(nk); stack.push(nk); }
        }
      }
      if (members.length < minC) return;
      var sx = 0, sy = 0, sz = 0, lo = Infinity, hi = -Infinity;
      for (var q = 0; q < members.length; q++) {
        var su = map.surfels[members[q]];
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

  // ---- Which bearings were never pointed at, per pitch band. This one serves
  //      the non-AR fallback, so it reads a plain yaw x pitch evidence grid
  //      rather than this map: with no trustworthy position there is no surface
  //      to attribute a gap to, and "you never turned that way" is the only
  //      question it may answer.
  //      cov = Float32Array[pitchRows*yawCols] -> [{band, from, to, cells}]
  //      sorted by size desc; `to` may exceed 360 for a wrap-around arc.
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
    ROOM_DEFAULTS: ROOM_DEFAULTS,
    STATE_THIN: STATE_THIN, STATE_PARTIAL: STATE_PARTIAL, STATE_COVERED: STATE_COVERED,
    createMap: createMap, clear: clear, observe: observe, project: project,
    projectPacked: projectPacked, PACK_STRIDE: PACK_STRIDE,
    unprojectDepth: unprojectDepth,
    createRoom: createRoom, clearRoom: clearRoom, fitRoom: fitRoom,
    setPhonePlanes: setPhonePlanes, roomSummary: roomSummary,
    stateOf: stateOf, stats: stats, gapClusters: gapClusters,
    dirBin: dirBin, popcount32: popcount32, bearingTo: bearingTo,
    uncoveredArcs: uncoveredArcs
  };
})();
if (typeof window !== "undefined") window.CoverageMap = CoverageMap;
if (typeof module !== "undefined" && module.exports) module.exports = CoverageMap;
