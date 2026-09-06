# navbake - offline navmesh for the FPS runtime

Bakes a navigation mesh out of a scene's existing PlayCanvas collision GLB and
writes it as the small JSON file `viewer/pc.js` fetches at page load:

```
work/<scene>/pc/collision.collision.glb   ->  work/<scene>/pc/nav.json  (+ nav.obj)
```

The collider is a static trimesh shell voxelised out of the gaussian cloud
(~0.35 m voxels, airborne crust clipped). Recast builds the walkable surface
from it; the runtime only ever sees a flat triangle list. Everything is world
space, the same coordinates the viewer instantiates the collider entity in
(untransformed), so nav.json verts are directly comparable with
`work/<scene>/viewer_assets/collision.json`'s `spawn` / `walk_path` / heightfield.

## Setup

```sh
cd tools/navbake
npm install          # recast-navigation 0.43.x (+ its @recast-navigation/* deps), MIT, wasm
```

Node >= 20. The wasm is carried inline by the `-compat` build, so there is no
`.wasm` file to locate in Node.

## Commands

```sh
# room: single indoor scan - the defaults are right for it
node bake.mjs room_w_jsonl --obj
node verify.mjs room_w_jsonl

# rocks: outdoor rock scan, same defaults
node bake.mjs rocks --obj
node verify.mjs rocks

# temple: outdoor structure on a cloud-crust hillside, needs a more permissive agent
node bake.mjs temple --obj --cell 0.25 --radius 0.3 --height 1.6 --climb 0.9 --slope 55 --region 0.6
node verify.mjs temple
```

`verify.mjs` exits non-zero, so the three lines above are the acceptance gate.
Add `--dry` to any bake to build and report without writing.

### Re-bake after regenerating a collider

The bake only reads `collision.collision.glb`, so re-bake whenever that file
changes - which is whenever `scripts/build_collider.py` runs (it also runs
`clip_collider.py` afterwards). One command per scene, or all of them:

```sh
cd tools/navbake
node bake.mjs room_w_jsonl --obj && node bake.mjs rocks --obj && \
node bake.mjs temple --obj --cell 0.25 --radius 0.3 --height 1.6 --climb 0.9 --slope 55 --region 0.6
node verify.mjs room_w_jsonl && node verify.mjs rocks && node verify.mjs temple
```

`verify.mjs` fails the bake as *stale* if the GLB is newer than the nav.json's
`generated` timestamp, so you cannot ship a navmesh that predates its collider.

## Output contract

```jsonc
{
  "version": 1,
  "source": "collision.collision.glb",
  "generated": "<ISO timestamp>",
  "build": { "cellSize": 0.15, "cellHeight": 0.15, "walkableRadius": 0.4,
             "walkableHeight": 1.7, "walkableClimb": 0.5, "walkableSlopeDeg": 50 },
  "bbox": [minX, minY, minZ, maxX, maxY, maxZ],
  "verts": [x, y, z, ...],            // world space, 3 decimals, y = feet level
  "tris":  [i0, i1, i2, ...],         // every 3 indices = 1 triangle
  "stats": { "triCount": 0, "areaM2": 0.0, "largestComponentTris": 0 }
}
```

- `build.*` are **metres / degrees** (what the runtime thinks in), not Recast's
  internal voxel units.
- `verts` y is the walkable surface itself - the floor - quantised to
  `cellHeight`. Recast stores the floor as the top of the heightfield cell that
  contains it, so expect the surface to sit up to one `cellHeight` (0.15-0.25 m)
  *above* the scanned geometry. Snap the bot with a short downward raycast
  instead of trusting it to the millimetre.
- Triangles are the walkable top surface only: they are Recast's polygon detail
  mesh read off the detour tiles, so there are no volume side walls, no
  ceilings, and no off-mesh connections. Vertices are welded after rounding to
  3 decimals (that step is what keeps the files at 60-260 kB instead of MBs).
- Sizes: room 259 kB / 8146 tris, temple 77 kB / 2721 tris, rocks 63 kB /
  2287 tris. The budget is ~2 MB.

## Tuning

Recast's `walkableHeight`, `walkableClimb`, `walkableRadius`, `minRegionArea` and
`maxEdgeLen` are counts of *cells*, not metres; `bake.mjs` converts them
(`--height` by `ceil`, `--climb` / `--radius` / `--region` by round, on `cs` for
the horizontal ones and `ch` for the vertical ones). The flags below are metres.

| flag | default | effect | raise it when | lower it when |
|---|---|---|---|---|
| `--cell` | 0.15 | heightfield resolution in XZ and Y. Smaller = more triangles, thinner strips survive, finer y | navmesh misses thin ledges / the file is too small to be detailed | the file grows too big, or the bake is slow. Half the collider's smallest triangle edge is plenty (room ~1.0 m, temple ~0.8 m, rocks ~0.43 m pitch) |
| `--radius` | 0.4 | erosion: how far the navmesh backs off from obstacles. The player capsule is 0.34 | routes hug walls too much | corridors get eaten and coverage of the walk_path fails (temple needed 0.3) |
| `--height` | 1.7 | required headroom (floor to ceiling/overhang) | bots should not crawl under stuff | a scanned porch / low crust blocks every route (temple needed 1.6) |
| `--climb` | 0.5 | max step between adjacent walkable cells; also the raster merge threshold. `collision.json` says the autopilot takes `max_step` 0.8 | voxel stair-steps break connectivity | bots path up walls / over rubble |
| `--slope` | 50 | triangles flatter than this (from their **winding**) are floor candidates | the surface is coarse and steep-scanned (temple 55) | you are seeing navmesh on walls |
| `--region` | 1.0 | side of the smallest isolated island kept, in metres | too many stranded islands | the mesh is being eaten |
| `--max-edge`, `--merge`, `--detail`, `--detail-error`, `--tile-size` | see `--help` | contour simplification / height detail sampling / detour tile edge | mostly leave alone | - |
| `--surface poly` | `detail` | fan-triangulate the detour polygons instead of the height-detail mesh: about half the triangles for the same coverage, but flat polygons, so the y in the middle of a big poly is an interpolation | file size matters more than foot-level height fidelity | - |
| `--solo` | tiled | single-tile build | you want one big tile - **fails above 65535 verts in the tile** | scene is large (all four here are) |
| `--orient` | `auto` | winding repair, see below | - | - |
| `--no-slope-filter` | off | `--slope 89.9`: every face that is not exactly vertical is floor, and the heightfield filters (headroom / ledge / climb) alone decide | diagnosing whether the slope test or the shell is at fault | shipping |

## Winding / normals

Recast derives a triangle's normal from its **vertex winding**
(`rcMarkWalkableTriangles` keeps it only if `normal.y > cos(slope)`); it never
reads a NORMAL attribute. Scan-derived shells are unreliable there, and the
failure mode is silent - the build "succeeds" and returns an empty or holey
navmesh. `navmesh-orient.mjs` therefore, before handing anything to Recast:

1. welds coincident vertices and drops degenerate triangles,
2. makes the winding *locally consistent* per patch (BFS over edge adjacency,
   flipping faces so each shared edge is crossed in opposite directions),
3. picks the global side per patch: a closed manifold patch with a meaningful
   enclosed volume is oriented so its normals point out of that volume (that is
   the free side, whether the patch is a solid boulder or a room full of air);
   anything else - scan shells are never watertight, they run ~98 % manifold
   with a few hundred border edges - is oriented so the projected up-facing area
   wins, because these colliders have had the airborne crust clipped and carry
   no ceilings.

Bake output prints the census (up / down / side triangles and areas before and
after, per patch, and which rule fired). On the four colliders currently in
`work/` this pass changes **nothing**: they already have zero downward-facing
triangles (room 18107 up / 0 down / 32101 side; temple 15143 / 0 / 8459; rocks
26041 / 0 / 6639), and `--orient never` bakes byte-comparable navmeshes. It is a
guard, not a crutch - `--orient always`, which flips every patch so floors face
down, yields **zero triangles** on all three scenes while still reporting
success. That is the bug this pass exists to prevent.

## Debugging a bake

- `--obj` writes `work/<scene>/pc/nav.obj` next to nav.json: open it in Blender /
  three.js / Windows 3D Viewer to eyeball the mesh. It is regenerable, delete it
  freely.
- `--dtnav [path]` also writes the real detour binary
  (`work/<scene>/pc/nav.dtnav.bin`; 120 kB rocks, 167 kB temple, 661 kB room -
  it carries the detail mesh and the tile BVH) so the runtime can pathfind on a
  genuine `dtNavMesh` with `NavMeshQuery` instead of ray-casting a triangle
  soup. Opt-in, not part of the contract.
- The bake prints a `walk coverage` line (spawn + `walk_path` sampled at 1 m) so
  a parameter sweep shows immediately whether the mesh covers the walked path,
  without running verify separately.

## What each scene bakes to today

| scene | params | tris | walkable | Y range | coverage of walk_path | stranded (<20 tris) | verdict |
|---|---|---|---|---|---|---|---|
| room_w_jsonl | defaults | 8146 | 8496 m2 | -56.4 .. 37.4 | 46/46 = 100 % (all inside a triangle) | 9.27 % of tris, 264 m2 | **PASS** |
| rocks | defaults | 2287 | 2297 m2 | -14.3 .. -3.2 | 16/16 = 100 % (all inside) | 2.84 % of tris, 23 m2 | **PASS** |
| temple | cell .25 radius .3 height 1.6 climb .9 slope 55 region .6 | 2721 | 6015 m2 | -40.8 .. 11.2 | 166/170 = 97.6 % | 1.03 % of tris, 35 m2 | **FAIL, 4 samples** |

The temple failure is in the input, not the bake, and is left failing on
purpose. Its `collision.json` (Sep 1) is 7 days newer than the collider GLB
(Aug 25) it is supposed to describe, and the two disagree: the collider's top
surface sits +2.60 m at the median and +24.3 m at p95 above `heights.f32` over
711 grid cells, and the walk rectangle's spawn (58.4, 12.6) is *outside* the
rectangle it is supposed to start. The 4 uncovered samples are all on the west /
north legs, 1.76-2.14 m from the nearest navmesh, where the only up-facing
collider surface within 4 m is either a cloud shelf 8-24 m above the heightfield
ground or a few square metres of 50-70 deg crust - `videos/temple.mp4` was flown
above a cloud layer and the splat rebuilt the clouds as solid geometry (see
`PROBLEM-temple.md`). The coverage only closes once the agent is ant-sized
(`--radius 0.05 --slope 89 --climb 2 --height 1.0`), which is not a surface a
0.34 m capsule should be sent over. Fixes, in the order that costs least:
re-derive the temple's `walk_path` + spawn
from the collider (`python scripts/walk_path_from_glb.py --asset
work/temple/viewer_assets --glb work/temple/pc/collision.collision.glb`), or
rebuild the collider from the current splat, then re-bake.

Room and rocks are fine. The room's 173 components are real: the scan has walkable
surfaces from y = -56 to y = +37 (it is a multi-level hall, not one floor), and
the piece under the spawn is 1181 m2, so bots start on the largest island. Note
that the "stranded" metric is *triangle* based, so it depends on `--cell`: the
same room at `--cell 0.3` reports 37.9 % stranded (1581 m2) purely because a
26 m2 island is then fewer than 20 triangles. Read the m2 number, not the %.

## recast-navigation 0.43.1 API notes (for the runtime side)

Real names in the installed `.d.ts` files - several of the names floating around
(`NavMeshManager`, `generateNavMesh`, `buildNavMeshVolume`,
`filterWalkableTriangles`, `saveNavMeshToBinary`) do not exist in this version:

- `await init()` from `@recast-navigation/core` (also re-exported by
  `recast-navigation`) **must** resolve before touching any API; generators
  throw `"init" must be called` otherwise. No wasm path needed in Node.
- `generateSoloNavMesh` / `generateTiledNavMesh` / `generateTileCache` are in
  `@recast-navigation/generators` and take *flat* `positions` + `indices` arrays
  plus a config of Recast units, and return `{ navMesh, success, error,
  intermediates }`. `intermediates` only fills in with `keepIntermediates = true`.
- **`success: true` with an empty navmesh is possible.** The tiled generator
  returns success when every tile produced zero polys; only the solo generator
  reports `"Failed to create Detour navmesh data"` for that case. Always check
  the polygon count yourself (bake.mjs exits on zero triangles).
- Read the mesh back with `getNavMeshPositionsAndIndices(navMesh, flags?)` -
  that is the "give me the surface" call (it walks each tile's `header()`,
  `polys(i)`, `verts()`, `detailMeshes()`, `detailVerts()`, `detailTris()`,
  skips off-mesh polys by `poly.getType() === 1`, and emits **duplicated**
  per-corner vertices: weld them). `tile.header()` is null for unused slots in
  `getTile(i)` up to `getMaxTiles()`.
- Binary save/load is `exportNavMesh(navMesh) -> Uint8Array` and
  `importNavMesh(data)` (in `serdes`), not `saveNavMeshToBinary`. Tile caches go
  through `exportTileCache` / `importTileCache` and need a
  `TileCacheMeshProcess`.
- Runtime queries: `new NavMeshQuery(navMesh)` with
  `findNearestPoly(pos, { halfExtents })`, `findPath`, `raycast`,
  `computePathCosts`; `NavMeshQuery`'s default filter includes all flags, so set
  `includeFlags` if you bake flag-per-area polys later. The tile header keeps
  `walkableHeight/Climb/Radius` in world units if the runtime wants to validate
  its capsule against the bake.
- `dtCreateNavMeshData` caps a tile at 65535 verts, which is what actually kills
  `--solo` on these scenes; `--tile-size` (cells per tile edge) is the knob.
- Two halves of the library disagree on what a position is: the generators and
  `getNavMeshPositionsAndIndices` take/return **flat arrays** (`[x,y,z,x,y,z...]`),
  while `NavMeshQuery` uses core's `Vector3`, which is an **`{x, y, z}` object**
  (there is `vec3.toArray` / `vec3.fromRaw` to cross between them). `findNearestPoly`
  internally destructures `{x, y, z}`, so handing it an array gives NaN positions
  rather than a clear error.
- `findNearestPoly({x,y,z}, { halfExtents: {x, y, z} })` returns
  `{ success, nearestRef, nearestPoint, isOverPoly }`. Its default halfExtents is
  1 unit per axis; the y extent has to span the difference between where the
  agent is and the baked surface (a couple of `cellHeight`s plus the step height
  is enough here), otherwise it fails and you get `success: false` on geometry
  that is obviously there.
