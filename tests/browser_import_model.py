"""Browser verification for real glTF/GLB import into the Place screen.

Drives the live stack (Next on :3000 proxying to the workspace backend on :8137)
with Playwright: imports a hand-built .glb, places it by clicking the scanned
floor, moves + rotates it with the dock, reloads to prove persistence, checks a
walk-mode collision response, and captures the exact user-facing text for every
rejection path. Writes screenshots + a machine-readable report to scratch/.

  .venv/Scripts/python.exe tests/browser_import_model.py [--scene test2train]

Nothing here touches the reconstruction pipeline; it only uses the workspace HTTP
API and the viewer the product already ships.
"""
import argparse
import json
import math
import struct
import sys
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "gltf-verify"
BASE = "http://127.0.0.1:3000"

# ---- glTF synthesis (self-contained, hand-written GLB) -----------------------


def _box_faces(x0, y0, z0, x1, y1, z1):
    """6 outward-wound quads, matching viewer/furniture_geometry.js exactly."""
    return [
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
    ]


def glb_from_boxes(boxes):
    """One mesh from many axis-aligned boxes — recognisable geometry, real bounds."""
    return _glb_from(boxes, [{"mesh": 0, "name": "imported"}])


def _glb_from(boxes, nodes):
    positions, indices = [], []
    for (mn, mx) in boxes:
        for quad in _box_faces(*mn, *mx):
            base = len(positions)
            positions.extend(quad)
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    mn = [min(p[a] for p in positions) for a in range(3)]
    mx = [max(p[a] for p in positions) for a in range(3)]
    binarr = bytearray(b"".join(struct.pack("<3f", *p) for p in positions))
    while len(binarr) % 4:
        binarr += b"\x00"
    idx_start = len(binarr)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    binarr += idx_bytes
    while len(binarr) % 4:
        binarr += b"\x00"
    doc = {
        "asset": {"version": "2.0", "generator": "qoder-verify"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions) * 12},
            {"buffer": 0, "byteOffset": idx_start, "byteLength": len(idx_bytes)},
        ],
        "buffers": [{"byteLength": len(binarr)}],
    }
    return _wrap_glb(doc, bytes(binarr))


# A 1.2 x 0.76 x 0.8 m table hung under a root node that turns it 90 deg about Y and
# halves it - the up-axis fix every real exporter writes. The transformed size is
# 0.4 x 0.38 x 0.6 m, and the raw 1.2 m is what a viewer that overwrote the root
# transform would draw, so the two are impossible to confuse.
RAW_BOX = ((-0.6, 0.0, -0.4), (0.6, 0.76, 0.4))
ROOT_ROT = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]
ROOT_NODES = [{"name": "root", "rotation": ROOT_ROT, "scale": [0.5, 0.5, 0.5], "children": [1]},
              {"name": "mesh", "mesh": 0, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}]


def rooted_glb():
    return _glb_from([RAW_BOX], ROOT_NODES), [0.4, 0.38, 0.6]


def swap_glb_chunks(glb):
    """The same bytes with the BIN chunk emitted before the JSON chunk."""
    total = int.from_bytes(glb[8:12], "little")
    json_len = int.from_bytes(glb[12:16], "little")
    json_c, bin_c = glb[12:20 + json_len], glb[20 + json_len:total]
    assert int.from_bytes(bin_c[4:8], "little") == 0x004E4942, "fixture: second chunk is not BIN"
    return glb[:12] + bin_c + json_c


def compressed_glb(kind):
    """Geometry only a decoder this viewer does not ship could produce."""
    positions = [[-0.6, 0, -0.4], [0.6, 0, -0.4], [0.6, 0.76, 0.4], [-0.6, 0.76, 0.4]]
    binarr = b"".join(struct.pack("<3f", *p) for p in positions)
    base = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                       "min": [-0.6, 0.0, -0.4], "max": [0.6, 0.76, 0.4]}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binarr)}],
        "buffers": [{"byteLength": len(binarr)}],
    }
    if kind == "KHR_draco_mesh_compression":
        base["meshes"][0]["primitives"][0]["extensions"] = {kind: {"bufferView": 0, "attributes": {"POSITION": 0}}}
    else:
        base["bufferViews"][0]["extensions"] = {kind: {"buffer": 0, "byteOffset": 0, "byteLength": len(binarr),
                                                       "byteStride": 12, "count": 4, "mode": "ATTRIBUTES",
                                                       "decoder": "Meshoptimizer"}}
    base["extensionsUsed"] = [kind]
    return _wrap_glb(base, binarr)


def silhouette(size, yaw_deg):
    """The axis-aligned width a yawed box occupies - what the drawn model must fill."""
    a = math.radians(yaw_deg)
    length, height, depth = size
    return [round(length * abs(math.cos(a)) + depth * abs(math.sin(a)), 4), height,
            round(length * abs(math.sin(a)) + depth * abs(math.cos(a)), 4)]


# The world-space extent of the geometry the browser actually drew for one placement.
DRAWN_AABB = """(id) => {
  const root = window.__app.root.findByName('placed');
  const holder = root && root.children.find(c => c.name === 'place:' + id);
  if (!holder) return { error: 'no holder' };
  let wrapper = null;
  const walk = (n) => { if (n.name && n.name.indexOf('model:') === 0) wrapper = n; n.children.forEach(walk); };
  walk(holder);
  if (!wrapper) return { error: 'no model wrapper: only the box fallback drew' };
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  let meshes = 0;
  // PlayCanvas stores a Mat4 column-major; transform the eight corners by hand so
  // this probe needs no engine globals.
  const tp = (m, x, y, z) => [x * m[0] + y * m[4] + z * m[8] + m[12],
                              x * m[1] + y * m[5] + z * m[9] + m[13],
                              x * m[2] + y * m[6] + z * m[10] + m[14]];
  wrapper.findComponents('render').forEach(rc => (rc.meshInstances || []).forEach(mi => {
    const box = mi.mesh && mi.mesh.aabb, node = mi.node;
    if (!box || !node) return;
    meshes++;
    const t = node.getWorldTransform().data, c = box.center, h = box.halfExtents;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
      const w = tp(t, c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z);
      mn[0] = Math.min(mn[0], w[0]); mx[0] = Math.max(mx[0], w[0]);
      mn[1] = Math.min(mn[1], w[1]); mx[1] = Math.max(mx[1], w[1]);
      mn[2] = Math.min(mn[2], w[2]); mx[2] = Math.max(mx[2], w[2]);
    }
  }));
  const ok = a => a.every(Number.isFinite);
  return { meshes, worldSize: ok(mn) && ok(mx) ? mx.map((v, i) => v - mn[i]) : null };
}"""



def _wrap_glb(doc, binarr):
    j = json.dumps(doc).encode("utf-8")
    while len(j) % 4:
        j += b" "
    jp = struct.pack("<II", len(j), 0x4E4F534A) + j
    bp = struct.pack("<II", len(binarr), 0x004E4942) + binarr
    return b"glTF" + struct.pack("<II", 2, 12 + len(jp) + len(bp)) + jp + bp


def table_glb():
    """A side table: a top slab on four legs, ~0.50 x 0.55 x 0.50 m — clearly not a box."""
    top = ((-0.25, 0.50, -0.25), (0.25, 0.55, 0.25))
    legs = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            lx = -0.22 if sx < 0 else 0.17
            lz = -0.22 if sz < 0 else 0.17
            legs.append(((lx, 0.0, lz), (lx + 0.05, 0.50, lz + 0.05)))
    return glb_from_boxes([top, *legs]), [0.50, 0.55, 0.50]


def minimal_gltf_doc(buffers=None, accessor_extra=None, drop_bounds=False):
    accessors = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 1]}]
    if drop_bounds:
        accessors = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}]
    if accessor_extra:
        accessors[0].update(accessor_extra)
    return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], "accessors": accessors,
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
            "buffers": buffers if buffers is not None else [{"byteLength": 36}]}


def self_referencing_nodes_doc():
    """A node graph that contains itself: walking it must stop, not spin."""
    doc = minimal_gltf_doc()
    doc["nodes"] = [{"mesh": 0, "children": [0]}]
    return doc


# ---- browser helpers ---------------------------------------------------------

FLOOR_JS = """async (scene) => {
  const root = `/runtime/work/${scene}/viewer_assets/`;
  const h = await (await fetch(root + 'collision.json')).json();
  const c = new Uint8Array(await (await fetch(root + 'coverage.u8')).arrayBuffer());
  const g = new Float32Array(await (await fetch(root + 'ground.f32')).arrayBuffer());
  let fp = null;
  for (let z = 10; z < h.nz - 10 && !fp; z++) for (let x = 10; x < h.nx - 10; x++) {
    const i = z * h.nx + x;
    if (c[i] && Number.isFinite(g[i])) { fp = [h.origin_xz[0] + (x + .5) * h.cell, g[i], h.origin_xz[1] + (z + .5) * h.cell]; break; }
  }
  if (!fp) throw new Error('no supported floor cell');
  const app = window.__app, camEnt = app.root.findByName('camera'), dev = app.graphicsDevice;
  const s = camEnt.camera.camera.worldToScreen({ x: fp[0], y: fp[1], z: fp[2] }, dev.width, dev.height, { x: 0, y: 0, z: 0 });
  const fwd = camEnt.forward, eye = camEnt.getPosition();
  const depth = (fp[0] - eye.x) * fwd.x + (fp[1] - eye.y) * fwd.y + (fp[2] - eye.z) * fwd.z;
  return { point: fp, fx: s.x / dev.width, fy: s.y / dev.height, inFront: depth > 0 };
}"""


def wait_viewer_ready(frame, timeout=60000):
    frame.wait_for_function("window.__ready || window.__loadError", timeout=timeout)
    err = frame.evaluate("window.__loadError || null")
    if err:
        raise RuntimeError(f"viewer load error: {err}")


def iframe_of(page):
    page.wait_for_selector("iframe", timeout=30000)
    return page.locator("iframe").element_handle().content_frame()


def message_text(page):
    el = page.locator(".editor-message")
    return (el.inner_text() if el.count() else "").strip()


# The viewer already answers focus-placement; the dock does not send it yet, so the
# proof drives the same command a user's camera would use.
FRAME_ON = """(id) => {
  const f = document.querySelector('iframe');
  f.contentWindow.postMessage({ namespace: 'groundcontrol', type: 'command', command: 'focus-placement', id },
    window.location.origin);
}"""
CAMERA_DISTANCE = """(centre) => {
  const cam = window.__app.root.findByName('camera');
  return Math.hypot(cam.getPosition().x - centre[0], cam.getPosition().y - centre[1], cam.getPosition().z - centre[2]);
}"""


def run(scene):
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"scene": scene, "steps": [], "checks": [], "errors": []}
    good_glb, dims = table_glb()
    (OUT / "imported-table.glb").write_bytes(good_glb)
    fixtures = {
        "unsupported.txt": b"this is not a model",
        "no-bounds.gltf": json.dumps(minimal_gltf_doc(drop_bounds=True)).encode(),
        "traversal.gltf": json.dumps(minimal_gltf_doc(buffers=[{"byteLength": 36, "uri": "../../../../etc/passwd"}])).encode(),
        "nan.gltf": json.dumps(minimal_gltf_doc(accessor_extra={"max": [float("nan"), 1, 1]})).encode(),
        "absurd.gltf": json.dumps(minimal_gltf_doc(accessor_extra={"max": [1e6, 1, 1]})).encode(),
        # A .gltf whose only buffer is the file itself: reading it would recurse.
        "self-buffer.gltf": json.dumps(minimal_gltf_doc(buffers=[{"byteLength": 36, "uri": "self-buffer.gltf"}])).encode(),
        "node-cycle.gltf": json.dumps(self_referencing_nodes_doc()).encode(),
    }
    rejections = [("unsupported", "unsupported.txt"), ("unreadable-bounds", "no-bounds.gltf"),
                  ("traversal", "traversal.gltf"), ("nan-accessor", "nan.gltf"),
                  ("absurd-extent", "absurd.gltf"), ("self-referencing-buffer", "self-buffer.gltf"),
                  ("node-cycle", "node-cycle.gltf")]
    for name, data in fixtures.items():
        (OUT / name).write_bytes(data)
    # Snapshot the scene's imported-model store so the teardown restores exactly what
    # was there, instead of deleting another session's models along with ours.
    models_dir = ROOT / "work" / scene / "models"
    backup = OUT / "models-before-run"
    import shutil
    if backup.exists():
        shutil.rmtree(backup)
    if models_dir.is_dir():
        shutil.copytree(models_dir, backup)
    else:
        backup = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        errors = []
        console = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # pc.js reports a failed model render on the console, not as a page error,
        # so capture it: "the catalogue chip exists" is not proof it rendered.
        page.on("console", lambda m: console.append(m.text) if "[place]" in m.text or "glb" in m.text.lower() else None)
        page.on("requestfailed", lambda r: console.append(f"requestfailed {r.url}") if "/models/" in r.url else None)
        created_ids = []
        try:
            # ---- failures first (nothing should be imported) ----
            page.goto(f"{BASE}/projects/{scene}/place", wait_until="domcontentloaded")
            frame = iframe_of(page)
            wait_viewer_ready(frame)

            def imported_rows():
                d = page.evaluate(f"async () => (await (await fetch('/api/backend/api/workspace/project?scene={scene}')).json())")
                return {f["item"]: f.get("model", {}).get("file") for f in d["furniture"] if f.get("imported")}, d

            imported_before, detail0 = imported_rows()
            report["checks"].append(["12 catalogue primitives offered",
                                     sum(1 for f in detail0["furniture"] if not f.get("imported")) == 12,
                                     f"{sum(1 for f in detail0['furniture'] if not f.get('imported'))} catalogue items"])
            report["artifacts_before"] = sorted(a["name"] for a in detail0["artifacts"])
            previous = message_text(page)
            for label, filename in rejections:
                page.locator(".import-model input[type='file']").set_input_files(str(OUT / filename))
                page.wait_for_function("""(before) => { const e = document.querySelector('.editor-message');
                    const t = e ? e.textContent.trim() : ''; return t.length > 0 && t !== before; }""",
                    arg=previous, timeout=20000)
                text = message_text(page)
                previous = text
                report["failures"] = report.get("failures", {})
                report["failures"][label] = text
                page.screenshot(path=str(OUT / f"fail-{label}.png"))
                refused = bool(text) and "Imported" not in text
                report["checks"].append([f"refusal text shown ({label})", refused, text])
                # a rejection must not have registered a model, or left its bytes on disk
                still, _ = imported_rows()
                report["checks"].append([f"no model registered on rejection ({label})", still == imported_before,
                                         f"{len(still)} imported rows"])

            # ---- import the real model ----
            with page.expect_response(lambda r: r.url.endswith("/model/import") and r.request.method == "POST", timeout=25000) as imp_resp:
                page.locator(".import-model input[type='file']").set_input_files(str(OUT / "imported-table.glb"))
            ir = imp_resp.value
            report["import_response_status"] = ir.status
            try:
                report["import_response_error"] = (ir.json() or {}).get("error")
            except Exception:  # noqa: BLE001
                report["import_response_error"] = None
            page.wait_for_function("""(before) => { const t = document.querySelector('.editor-message');
                const s = t ? t.textContent.trim() : ''; return s.length > 0 && s !== before && /^Imported/.test(s); }""",
                arg=previous, timeout=25000)
            import_msg = message_text(page)
            after_import, detail = imported_rows()
            fresh = [k for k in after_import if k not in imported_before]
            imported = next((f for f in detail["furniture"] if f.get("imported") and f["item"] in fresh), None)
            report["checks"].append(["exactly one model registered by the import", len(fresh) == 1, fresh])
            report["checks"].append(["imported model appears in catalogue", bool(imported), imported and imported["label"]])
            report["import_message"] = import_msg
            page.screenshot(path=str(OUT / "imported-catalogue.png"))

            model_id = imported["item"]
            model_size = imported["size"]
            report["imported_model"] = {"id": model_id, "label": imported["label"], "size": model_size,
                                        "file": after_import[model_id]}
            report["checks"].append(["imported size is the file's real bounds", model_size == dims,
                                     f"{model_size} vs synthesised {dims}"])

            # ---- place it by clicking the scanned floor ----
            page.locator(".furniture-chip").filter(has_text=imported["label"]).first.click()
            floor = frame.evaluate(FLOOR_JS, scene)
            box = frame.locator("canvas").bounding_box()
            placed = None
            fx, fy = floor["fx"], floor["fy"]
            if floor["inFront"] and 0.05 < fx < 0.95 and 0.05 < fy < 0.95:
                with page.expect_response(lambda r: r.url.endswith("/placements") and r.request.method == "POST", timeout=15000) as resp:
                    page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
                placed = resp.value.json()
            if placed is None:
                # Deterministic fallback through the real placement endpoint on a grid floor point.
                with page.expect_response(lambda r: r.url.endswith("/placements") and r.request.method == "POST", timeout=15000) as resp:
                    page.evaluate("""async ({scene, model_id, point}) => {
                      const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
                      return fetch('/api/backend/api/workspace/placements', { method: 'POST', headers: {'content-type':'application/json'},
                        body: JSON.stringify({scene, item: model_id, point, yaw_deg: 0, model_revision: d.model_revision})});
                    }""", {"scene": scene, "model_id": model_id, "point": floor["point"]})
                placed = resp.value.json()
            p = placed["placements"][-1]
            created_ids.append(p["id"])
            report["placement"] = {"id": p["id"], "size": p["size"], "model_file": p.get("model", {}).get("file"),
                                   "center": [p["center_xz"][0], p["center_y"], p["center_xz"][1]]}
            report["checks"].append(["placed size == real model size", p["size"] == model_size, p["size"]])
            report["checks"].append(["placement carries a model file", bool(p.get("model", {}).get("file")), p.get("model", {}).get("file")])

            # The bytes must actually reach the viewer over the same URL it uses.
            report["model_url"] = page.evaluate("""async (u) => {
              const r = await fetch(u);
              return { url: u, status: r.status, type: r.headers.get('content-type'), bytes: (await r.arrayBuffer()).byteLength };
            }""", f"/runtime/work/{scene}/models/{p.get('model', {}).get('file')}")
            report["checks"].append(["model bytes served to the viewer", report["model_url"]["status"] == 200 and report["model_url"]["bytes"] > 0,
                                     report["model_url"]])

            # the viewer must build a collidable entity + render the imported geometry.
            # The model wrapper is a child of the item holder (not the placed root), so walk
            # the whole placed subtree for a 'model:' node and count its instantiated verts.
            probe_models = """() => {
              const root = window.__app.root.findByName('placed');
              if (!root) return { wrappers: 0, verts: 0, tris: 0 };
              let wrappers = 0, verts = 0, tris = 0;
              const walk = (node) => {
                if (node.name && node.name.indexOf('model:') === 0) {
                  wrappers++;
                  node.findComponents('render').forEach(rc => (rc.meshInstances || []).forEach(mi => {
                    if (mi.mesh && mi.mesh.vertexBuffer) verts += mi.mesh.vertexBuffer.numVertices;
                    const prim = mi.mesh && mi.mesh.primitive && mi.mesh.primitive[0];
                    if (prim) tris += (prim.count || 0) / 3;
                  }));
                }
                for (const c of node.children) walk(c);
              };
              walk(root);
              return { wrappers, verts, tris };
            }"""
            frame.wait_for_function("() => !!window.__app.root.findByName('placed')", timeout=20000)
            frame.wait_for_function(
                "() => { const root = window.__app.root.findByName('placed'); if (!root) return false;"
                " let found=false; const walk=n=>{ if(n.name&&n.name.indexOf('model:')===0) found=true; n.children.forEach(walk); }; walk(root); return found; }",
                timeout=20000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "placed-visible.png"))
            visible = frame.evaluate(probe_models)
            report["model_render"] = visible
            report["checks"].append(["imported model geometry rendered", visible["wrappers"] >= 1 and visible["verts"] > 8,
                                     f"{visible['verts']} verts, {visible['wrappers']} model wrapper(s)"])

            # Bring the camera to the item: a 0.5 m object in a 50 m scan is only
            # "visible in the 3D view" if the shot is actually framed on it.
            page.evaluate(FRAME_ON, p["id"])
            page.wait_for_timeout(2600)
            distance = frame.evaluate(CAMERA_DISTANCE, report["placement"]["center"])
            report["camera_distance_after_focus"] = distance
            report["checks"].append(["camera framed onto the placed model", distance is not None and distance < 6,
                                     f"{distance:.2f} units from the item" if distance is not None else "no camera"])
            page.screenshot(path=str(OUT / "model-framed.png"))

            # ---- select it: dock shows the honest model caption ----
            row = page.locator(".place-list li").filter(has_text=imported["label"]).first
            row.get_by_role("button", name=f"Select {imported['label']}", exact=True).click()
            page.wait_for_selector(".editor-dock", timeout=10000)
            note = page.locator(".dock-model-note").inner_text()
            report["dock_model_note"] = note
            report["checks"].append(["dock shows real file-read size", "true size" in note and f"{model_size[0]:.2f}" in note, note[:90]])
            page.screenshot(path=str(OUT / "placed-selected.png"))

            # ---- move + rotate + resize via the dock ----
            # A placement record carries center_xz + center_y, never a "center" key.
            centre_before = report["placement"]["center"]

            def dock_click(name):
                with page.expect_response(
                        lambda r: r.url.endswith("/placements/update") and r.request.method == "POST",
                        timeout=15000) as got:
                    page.get_by_role("button", name=name, exact=True).click()
                return got.value

            after_move = dock_click("Move forward").request.post_data_json["point"]
            after_yaw = dock_click("Rotate right 45 degrees").request.post_data_json["yaw_deg"]
            grown = dock_click("Increase length").request.post_data_json["size"]
            stored = page.evaluate("""async ({scene, id}) => {
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              return d.placements.find(x => x.id === id);
            }""", {"scene": scene, "id": p["id"]})
            report["move"] = {"from_z": centre_before[2], "to_z": after_move[2], "yaw": after_yaw, "size_request": grown,
                              "stored": {"center_xz": stored["center_xz"], "yaw_deg": stored["yaw_deg"],
                                         "size": stored["size"], "fit": stored["fit"]["reason"]}}
            report["checks"].append(["move changed z", abs(after_move[2] - centre_before[2]) > 0.01,
                                     f"{centre_before[2]} -> {after_move[2]}"])
            report["checks"].append(["rotate set yaw 45", after_yaw == 45, after_yaw])
            report["checks"].append(["resize grew length by the dock step",
                                     abs(grown[0] - model_size[0] - 0.1) < 0.02, f"{model_size[0]} -> {grown}"])
            report["checks"].append(["server stored the moved, rotated and resized item",
                                     abs(stored["center_xz"][1] - after_move[2]) < 0.01 and stored["yaw_deg"] == 45
                                     and abs(stored["size"][0] - grown[0]) < 0.01,
                                     [stored["center_xz"][1], stored["yaw_deg"], stored["size"]]])
            report["checks"].append(["every write came back re-fit-checked", bool(stored["fit"].get("reason")),
                                     (stored["fit"]["reason"] or "")[:160]])
            p = stored
            page.wait_for_timeout(900)
            page.screenshot(path=str(OUT / "moved-rotated.png"))

            # ---- walk-mode collision response on the placed model ----
            collider = frame.evaluate("""() => {
              const root = window.__app.root.findByName('placed');
              const boxes = root.children.filter(c => c.collision && c.collision.type === 'box');
              return { staticBody: root.rigidbody && root.rigidbody.type === 'static',
                       compound: root.collision && root.collision.type === 'compound',
                       boxCount: boxes.length,
                       halfExtents: boxes.map(b => [b.collision.halfExtents.x, b.collision.halfExtents.y, b.collision.halfExtents.z]) };
            }""")
            # A conservative box from the model's real bounds, yaw included: the ray must
            # stop on its surface, not merely hit something.
            half = [s / 2 for s in p["size"]]
            centre = [p["center_xz"][0], p["center_y"], p["center_xz"][1]]
            surface = half[0] * abs(math.cos(math.radians(p["yaw_deg"]))) \
                + half[2] * abs(math.sin(math.radians(p["yaw_deg"])))
            collide = frame.evaluate("""async ({ centre, half }) => {
              const app = window.__app;
              const pc = await import('/runtime/viewer/pc/playcanvas.mjs');
              const reach = half[0] + 1.0, cy = centre[1];
              const from = new pc.Vec3(centre[0] - reach, cy, centre[2]);
              const to = new pc.Vec3(centre[0] + reach, cy, centre[2]);
              const hit = app.systems.rigidbody.raycastFirst(from, to, { filterCollisionMask: pc.BODYMASK_STATIC });
              const dist = hit && hit.point ? Math.hypot(hit.point.x - from.x, hit.point.y - from.y, hit.point.z - from.z) : null;
              let body = hit && (hit.ent || hit.entity), placed = false;
              while (body) { if (body.name === 'placed') { placed = true; break; } body = body.parent; }
              return { blocked: !!(hit && hit.point), dist, fromPlacedRoot: placed };
            }""", {"centre": centre, "half": half})
            report["collision"] = {"collider": collider, "raycast": collide,
                                   "silhouette_half_width_x": round(surface, 4),
                                   "size": p["size"], "yaw_deg": p["yaw_deg"]}
            report["checks"].append(["placed root is a static compound body (B4)", collider["staticBody"] and collider["compound"], collider])
            report["checks"].append(["conservative box collider built from the model's real bounds",
                                     any(all(abs(a[i] - half[i]) < 0.01 for i in range(3)) for a in collider["halfExtents"]),
                                     f"{collider['boxCount']} boxes; expected {half}"])
            expected = half[0] + 1.0 - surface
            report["checks"].append(["walk ray stops on the model's own collider surface",
                                     bool(collide.get("blocked")) and collide.get("fromPlacedRoot")
                                     and abs(collide["dist"] - expected) < 0.05,
                                     f"hit at {collide['dist']} from the placed root={collide.get('fromPlacedRoot')}, "
                                     f"surface at {round(expected, 3)} (yaw {p['yaw_deg']} deg)"])

            # ---- reload: must persist ----
            page.reload(wait_until="domcontentloaded")
            frame = iframe_of(page)
            wait_viewer_ready(frame)
            page.wait_for_timeout(1500)
            listed = page.locator(".place-list li").filter(has_text=imported["label"]).count()
            still_rendered = frame.evaluate("""() => {
              const root = window.__app.root.findByName('placed');
              if (!root) return { wrappers: 0, verts: 0 };
              let wrappers = 0, verts = 0;
              const walk = (n) => {
                if (n.name && n.name.indexOf('model:') === 0) { wrappers++; n.findComponents('render').forEach(rc => (rc.meshInstances || []).forEach(mi => { if (mi.mesh && mi.mesh.vertexBuffer) verts += mi.mesh.vertexBuffer.numVertices; })); }
                for (const c of n.children) walk(c);
              };
              walk(root);
              return { wrappers, verts };
            }""")
            report["checks"].append(["placement persists after reload (list)", listed >= 1, listed])
            report["checks"].append(["model geometry persists after reload", bool(still_rendered["wrappers"]) and still_rendered["verts"] > 8, still_rendered])
            page.evaluate(FRAME_ON, p["id"])
            page.wait_for_timeout(2600)
            report["camera_distance_after_reload"] = frame.evaluate(CAMERA_DISTANCE, report["placement"]["center"])
            page.screenshot(path=str(OUT / "after-reload.png"))

            # ---- the real GeoJSON deliverable, from the panel's own export button ----
            with page.expect_download(timeout=15000) as dl:
                page.get_by_role("button", name="Layout").click()
            geo_path = Path(dl.value.path())
            geo = json.loads(geo_path.read_text(encoding="utf-8"))
            mine = next((f for f in geo.get("features", []) if f.get("id") == f"place:{p['id']}"), None)
            report["geojson"] = {"file": geo_path.name, "features": len(geo.get("features", [])),
                                 "name": mine and geo.get("name"),
                                 "properties": mine and mine["properties"]}
            report["checks"].append(["downloaded GeoJSON carries the imported item",
                                     bool(mine) and mine["properties"]["source"] == "gltf-import"
                                     and mine["properties"]["true_size_m"] == model_size,
                                     mine and mine["properties"]])

            # ---- rooms.json must now be a visible deliverable (B3 artefact allowlist) ----
            detail = page.evaluate(f"async () => (await (await fetch('/api/backend/api/workspace/project?scene={scene}')).json())")
            rooms = next((a for a in detail["artifacts"] if a["name"].endswith("rooms.json")), None)
            report["rooms_artifact"] = rooms
            report["checks"].append(["rooms.json published as an artefact", bool(rooms), rooms])
            page.get_by_role("button", name="Exports").click()
            page.wait_for_selector(".artifact-list", timeout=15000)
            shown = page.locator(".artifact-link").filter(has_text="rooms.json").count()
            report["checks"].append(["rooms.json listed in the Exports panel", shown >= 1, f"{shown} link(s)"])
            if rooms:
                served = page.evaluate("""async (u) => {
                  const r = await fetch(u);
                  const text = await r.text();
                  let rooms_count = null;
                  try { rooms_count = (JSON.parse(text).rooms || []).length; } catch { /* not a room file */ }
                  return { url: u, status: r.status, bytes: text.length, rooms_count };
                }""", rooms["url"])
                report["rooms_served"] = served
                report["checks"].append(["rooms.json bytes are served", served["status"] == 200 and served["bytes"] > 50, served])
            page.screenshot(path=str(OUT / "artifacts-rooms.png"))

            # ---- the 12 built-in primitives must still draw as component geometry ----
            page.get_by_role("button", name="Place").click()
            page.wait_for_selector(".furniture-chip", timeout=15000)
            sofa = page.locator(".furniture-chip").filter(has_text="Sofa").first
            sofa.click()
            spot = frame.evaluate(FLOOR_JS, scene)
            canvas = frame.locator("canvas").bounding_box()
            with page.expect_response(lambda r: r.url.endswith("/placements") and r.request.method == "POST", timeout=15000) as so:
                page.mouse.click(canvas["x"] + canvas["width"] * spot["fx"], canvas["y"] + canvas["height"] * spot["fy"])
            sofa_row = so.value.json()["placements"][-1]
            created_ids.append(sofa_row["id"])
            page.wait_for_timeout(1200)
            primitive = frame.evaluate("""(id) => {
              const root = window.__app.root.findByName('placed');
              const holder = root && root.children.find(c => c.name === 'place:' + id);
              if (!holder) return null;
              let verts = 0;
              (holder.render.meshInstances || []).forEach(mi => { if (mi.mesh && mi.mesh.vertexBuffer) verts += mi.mesh.vertexBuffer.numVertices; });
              return { verts, wrapper: holder.children.some(c => c.name && c.name.indexOf('model:') === 0), model: !!holder };
            }""", sofa_row["id"])
            report["primitive"] = {"label": sofa_row["label"], "size": sofa_row["size"], "render": primitive}
            report["checks"].append(["built-in primitive still places", bool(primitive) and sofa_row["item"] == "sofa", sofa_row["item"]])
            report["checks"].append(["built-in primitive draws component geometry, not a box",
                                     bool(primitive) and primitive["verts"] >= 200 and not primitive["wrapper"], primitive])
            page.screenshot(path=str(OUT / "primitive-regression.png"))

            # ---- an exporter-shaped model: its root node's transform must survive ----
            # Real exporters hang the mesh under a root node carrying the up-axis
            # rotation and unit scale. The fit that maps the model onto its box must
            # not overwrite that node, or the item draws at its raw size while the
            # collider stays at the size the file declared - invisible in a screenshot
            # of a box, decisive when the two numbers are 1.2 m and 0.4 m.
            rooted, rooted_dims = rooted_glb()
            (OUT / "rooted-table.glb").write_bytes(rooted)
            with page.expect_response(lambda r: r.url.endswith("/model/import") and r.request.method == "POST",
                                      timeout=25000) as imp2:
                page.locator(".import-model input[type='file']").set_input_files(str(OUT / "rooted-table.glb"))
            report["rooted_import_status"] = imp2.value.status
            page.wait_for_timeout(1800)
            rooted_rows, detail2 = imported_rows()
            fresh2 = [k for k in rooted_rows if k not in imported_before and k != model_id]
            report["checks"].append(["the exporter-shaped import registered one model", len(fresh2) == 1, fresh2])
            root_id = fresh2[0]
            row2 = next(f for f in detail2["furniture"] if f.get("imported") and f["item"] == root_id)
            report["checks"].append(["exporter-shaped size follows the node graph, not the raw mesh",
                                     row2["size"] == rooted_dims, f"{row2['size']} vs {rooted_dims} (raw 1.2)"])
            spot2 = frame.evaluate(FLOOR_JS, scene)
            placed2 = page.evaluate("""async ({scene, item, point, rev}) => (await (await fetch(
              '/api/backend/api/workspace/placements', { method: 'POST', headers: {'content-type':'application/json'},
              body: JSON.stringify({scene, item, point, yaw_deg: 30, model_revision: rev})})).json())""",
                {"scene": scene, "item": root_id, "point": spot2["point"], "rev": detail2["model_revision"]})
            p2 = placed2["placements"][-1]
            created_ids.append(p2["id"])
            frame.wait_for_function(
                """(id) => { const root = window.__app.root.findByName('placed');
                  const h = root && root.children.find(c => c.name === 'place:' + id); if (!h) return false;
                  let f = false; const walk = n => { if (n.name && n.name.indexOf('model:') === 0) f = true; n.children.forEach(walk); };
                  walk(h); return f; }""", arg=p2["id"], timeout=20000)
            page.wait_for_timeout(1500)
            drawn = frame.evaluate(DRAWN_AABB, p2["id"])
            want = silhouette(p2["size"], p2["yaw_deg"])
            report["rooted_model"] = {"size": p2["size"], "yaw_deg": p2["yaw_deg"], "drawn": drawn,
                                      "collider_silhouette": want}
            report["checks"].append(["drawn geometry fills the collider box, not the raw model size",
                                     bool(drawn.get("worldSize")) and all(
                                         abs(drawn["worldSize"][i] - want[i]) < 0.02 for i in range(3)),
                                     f"drawn {[round(v, 3) for v in (drawn.get('worldSize') or [])]} vs box {want}"])
            page.evaluate(FRAME_ON, p2["id"])
            page.wait_for_timeout(2400)
            page.screenshot(path=str(OUT / "rooted-model.png"))

            # ---- geometry the shipped viewer cannot decode is refused, not boxed ----
            for label, filename, payload in (
                    ("bin-chunk-first", "bin-first.glb", swap_glb_chunks(rooted)),
                    ("draco", "draco.glb", compressed_glb("KHR_draco_mesh_compression")),
                    ("meshopt", "meshopt.glb", compressed_glb("EXT_meshopt_compression"))):
                (OUT / filename).write_bytes(payload)
                previous = message_text(page)
                page.locator(".import-model input[type='file']").set_input_files(str(OUT / filename))
                page.wait_for_function("""(before) => { const e = document.querySelector('.editor-message');
                    const t = e ? e.textContent.trim() : ''; return t.length > 0 && t !== before; }""",
                    arg=previous, timeout=20000)
                text = message_text(page)
                report["failures"][label] = text
                rows_now, _ = imported_rows()
                report["checks"].append([f"refused because the viewer cannot render it ({label})",
                                         rows_now == rooted_rows and "cannot" in text, text[:120]])
                page.screenshot(path=str(OUT / f"fail-{label}.png"))
        except Exception as e:  # noqa: BLE001 - capture so the report still reflects the failure point
            report["fatal"] = repr(e)
        finally:
            # Clean up: delete placements we created and remove imported model files from the scene.
            report["errors"] = errors
            report["console"] = console[:20]
            try:
                for pid in created_ids:
                    page.evaluate("""async ({scene, id}) => {
                      await fetch('/api/backend/api/workspace/placements/delete', { method: 'POST', headers: {'content-type':'application/json'},
                        body: JSON.stringify({scene, id})});
                    }""", {"scene": scene, "id": pid})
            except Exception as e:  # noqa: BLE001 - cleanup must not mask the report
                print("[verify] placement cleanup warning:", e)
            browser.close()

    # Restore the scene's imported-model store to its pre-run state.
    if models_dir.is_dir():
        shutil.rmtree(models_dir, ignore_errors=True)
    if backup is not None and backup.is_dir():
        shutil.copytree(backup, models_dir)

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    passed = sum(1 for c in report["checks"] if c[1])
    total = len(report["checks"])
    # Every check that ran passing is not the same as the run finishing: an abort
    # partway leaves a short list of green rows behind.
    incomplete = report.get("fatal") or not report.get("checks")
    if incomplete:
        print(f"[gltf-verify] ABORTED: {report.get('fatal')}")
    print(f"\n[gltf-verify] {passed}/{total} checks passed; page errors: {errors}")
    return 0 if passed == total and not errors and not incomplete else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="test2train")
    args = ap.parse_args()
    sys.exit(run(args.scene))


if __name__ == "__main__":
    main()
