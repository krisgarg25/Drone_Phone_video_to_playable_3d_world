import os
import unittest
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("WORKSPACE_URL", "http://127.0.0.1:3000")
SCENE = os.environ.get("WORKSPACE_SCENE", "rocks")


class WorkspaceBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel="chrome", headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.page = self.context.new_page()
        self.errors = []
        self.created = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.route("**/api/backend/api/workspace/placements", self.capture_create)
        self.page.route("**/api/backend/api/workspace/measurements", self.capture_create)

    def capture_create(self, route):
        import json
        if route.request.method != "POST":
            route.continue_()
            return
        data = route.request.post_data_json
        collection = route.request.url.rsplit("/", 1)[-1]
        if collection == "placements":
            data["label"] = f"{data.get('label') or 'Coffee table'} QA {uuid.uuid4().hex}"
        response = route.fetch(post_data=json.dumps(data))
        result = response.json()
        for record in result.get(collection, []):
            if record.get("label") == data.get("label"):
                self.created.append((data["scene"], collection, record["id"]))
        route.fulfill(response=response)

    def tearDown(self):
        try:
            self.page.unroute_all(behavior="wait")
            for scene, collection, identity in self.created:
                self.context.request.post(
                    BASE + f"/api/backend/api/workspace/{collection}/delete",
                    headers={"Origin": BASE}, data={"scene": scene, "id": identity})
        finally:
            self.context.close()

    def test_home_is_project_library(self):
        self.page.goto(BASE, wait_until="domcontentloaded")
        expect(self.page.get_by_role("heading", name="Projects", exact=True)).to_be_visible()
        expect(self.page.get_by_role("button", name="New reconstruction", exact=True)).to_be_visible()
        self.assertNotIn("Official targets", self.page.locator("main").inner_text())
        self.assertEqual(self.errors, [])

    def test_project_search_and_workspace(self):
        self.page.goto(BASE, wait_until="domcontentloaded")
        self.page.get_by_placeholder("Search projects…").fill(SCENE)
        project = self.page.locator(f'a[href="/projects/{SCENE}"]').first
        expect(project).to_be_visible(timeout=30000)
        project.click()
        expect(self.page.locator('iframe[title="3D reconstruction"]')).to_have_attribute("src", __import__("re").compile(r"/runtime/viewer/pc.html.*embed=1"))
        expect(self.page.get_by_role("button", name="Source frames", exact=True)).to_be_visible()
        expect(self.page.get_by_role("button", name="Exports", exact=True)).to_be_visible()
        self.assertEqual(self.errors, [])

    def test_default_overview_contains_visible_geometry(self):
        import io
        from PIL import Image
        self.page.goto(BASE + "/projects/" + SCENE, wait_until="domcontentloaded")
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready || window.__loadError", timeout=60000)
        self.assertIsNone(frame.evaluate("window.__loadError || null"))
        image = Image.open(io.BytesIO(frame.locator("canvas").screenshot())).convert("RGB")
        width, height = image.size
        centre = image.crop((width // 5, height // 5, width * 4 // 5, height * 4 // 5)).resize((64, 64))
        self.assertGreater(len(centre.getcolors(4096) or []), 100, "The ready viewport contains no visible scene detail")

    def test_saved_measurement_roundtrip_and_export(self):
        import json
        import math
        import uuid
        label = "Browser QA " + uuid.uuid4().hex[:8]
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("checkbox", name="Camera positions", exact=True).check()
        frame.wait_for_function('window.__app.root.findByName("camerasEntity").enabled')
        self.page.get_by_role("button", name="Inspect frame 12", exact=True).click()
        frame.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        with self.page.expect_download() as screenshot:
            self.page.get_by_role("button", name="Save viewport snapshot", exact=True).click()
        self.assertTrue(Path(screenshot.value.path()).read_bytes().startswith(b"\x89PNG"))
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Distance", exact=True).click()
        bounds = frame.locator("canvas").bounding_box()
        for x in (0.45, 0.55):
            self.page.mouse.click(bounds["x"] + bounds["width"] * x, bounds["y"] + bounds["height"] * 0.65)
        expect(self.page.locator(".measure-draft")).to_contain_text("2 selected.")
        self.page.get_by_role("textbox", name="Measurement name", exact=True).fill(label)
        self.page.get_by_role("button", name="Save to project", exact=True).click()
        expect(self.page.get_by_text(label, exact=True)).to_be_visible()
        self.page.reload()
        self.page.get_by_role("button", name="Measure", exact=True).click()
        expect(self.page.get_by_text(label, exact=True)).to_be_visible()
        self.page.get_by_role("button", name="Exports", exact=True).click()
        with self.page.expect_download() as exported:
            self.page.get_by_role("button", name="Download JSON", exact=True).click()
        data = json.loads(Path(exported.value.path()).read_text())
        saved = next(item for item in data["measurements"] if item["label"] == label)
        # With a point cloud behind the scene the engine snaps the clicks and adds
        # an uncertainty budget, so the value tracks the raw click distance only
        # within snapping tolerance — and never fabricates one when unsupported.
        self.assertIn("engine", saved)
        self.assertTrue(saved["engine"])
        self.assertIsNotNone(saved.get("uncertainty"))
        if saved.get("valid"):
            self.assertAlmostEqual(saved["value"], math.dist(*saved["points"]), delta=1.5)
        self.assertFalse(saved["stale"])
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Delete " + label, exact=True).click()
        self.page.get_by_role("button", name="Confirm delete", exact=True).click()
        expect(self.page.get_by_text(label, exact=True)).not_to_be_visible()
        self.assertEqual(self.errors, [])

    def test_project_artifact_download_preserves_attachment(self):
        import json
        self.page.goto(BASE + "/projects/" + SCENE)
        self.page.get_by_role("button", name="Exports", exact=True).click()
        artifact = self.page.locator("a.artifact-link").filter(has_text="frame.json").first
        with self.page.expect_download(timeout=10000) as downloaded:
            artifact.click()
        result = json.loads(Path(downloaded.value.path()).read_text())
        self.assertIn("scale_source", result)
        self.assertEqual(downloaded.value.suggested_filename, "frame.json")

    def test_switching_to_survey_drops_hidden_scale_anchor(self):
        submitted = []
        def launch(route):
            submitted.append(route.request.post_data_json)
            route.fulfill(status=409, content_type="application/json", body='{"error":"Prepare survey inputs first."}')
        self.page.route("**/api/backend/api/workspace/run", launch)
        self.page.goto(BASE + "/projects/" + SCENE)
        self.page.get_by_role("button", name="Reconstruct", exact=True).click()
        dialog = self.page.get_by_role("dialog")
        dialog.get_by_role("combobox", name="Optional scale reference", exact=True).select_option("height")
        dialog.locator('input[type="number"]').fill("1.6")
        dialog.get_by_role("combobox", name="Output", exact=True).select_option("survey")
        dialog.get_by_role("checkbox").check()
        dialog.get_by_role("button", name="Start reconstruction", exact=True).click()
        expect(dialog.get_by_role("alert")).to_contain_text("Prepare survey inputs first.")
        self.assertEqual(submitted[0]["engine"], "survey")
        self.assertNotIn("anchor", submitted[0])

    def test_height_tool_stops_after_two_visible_picks(self):
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("button", name="Inspect frame 12", exact=True).click()
        frame.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        self.page.evaluate("window.qaPicks = []; window.addEventListener('message', e => { if (e.data?.namespace === 'groundcontrol' && e.data.type === 'pick') window.qaPicks.push(e.data.point); })")
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Height", exact=True).click()
        bounds = frame.locator("canvas").bounding_box()
        for x in (0.45, 0.55, 0.6):
            self.page.mouse.click(bounds["x"] + bounds["width"] * x, bounds["y"] + bounds["height"] * 0.65)
        frame.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        self.assertEqual(len(self.page.evaluate("window.qaPicks")), 2)

    def test_source_frames_work_without_a_ready_viewer(self):
        self.page.goto(BASE + "/projects/auditorium_hd")
        button = self.page.get_by_role("button", name="Inspect frame 1", exact=True)
        expect(button).to_be_enabled(timeout=30000)
        button.click()
        expect(self.page.get_by_role("img", name="Inspected source frame 1", exact=True)).to_be_visible()

    def test_processing_view_can_filter_ready_projects(self):
        self.page.goto(BASE + "/?view=processing")
        # Warm the first fetch (cold backend import can take seconds) before navigating.
        expect(self.page.locator(".project-grid, .empty-state").first).to_be_visible(timeout=30000)
        self.page.locator(".filter-tabs").get_by_role("link", name="Ready", exact=True).click()
        self.page.wait_for_url("**/?view=ready", timeout=5000)
        expect(self.page.locator(".project-card").first).to_be_visible(timeout=15000)
        self.assertEqual(self.page.locator(".project-card .status-failed").count(), 0)

    def test_semantics_layer_toggles_and_reports_classes(self):
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        toggle = self.page.get_by_role("checkbox", name="Semantic classes", exact=True)
        expect(toggle).to_be_enabled()
        toggle.check()
        expect(toggle).to_be_checked()
        self.assertTrue(frame.evaluate('window.__app.root.findByName("semanticsEntity").enabled'))
        rows = self.page.locator(".class-list .datum-row").all_text_contents()
        self.assertTrue(any("vegetation" in r for r in rows), rows)

    def test_volume_tool_and_geojson_csv_export(self):
        import json
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        # A2: the Volume tool is present and selects.
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Volume", exact=True).click()
        expect(self.page.locator(".measure-draft")).to_contain_text("footprint", timeout=5000)
        # A4: create a measurement deterministically via the API, then export it.
        label = "QA export " + uuid.uuid4().hex[:6]
        created = self.page.evaluate(
            """async ({scene, label}) => {
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              const r = await fetch('/api/backend/api/workspace/measurements', {
                method: 'POST', headers: {'content-type': 'application/json'},
                body: JSON.stringify({scene, kind: 'distance', label, points: [[-2, 0, 0], [2, 0, 0]], model_revision: d.model_revision})});
              const j = await r.json();
              return {status: r.status, engine: (j.measurements || []).slice(-1)[0]?.engine};
            }""", {"scene": SCENE, "label": label})
        self.assertEqual(created["status"], 200, created)
        self.page.reload()
        self.page.get_by_role("button", name="Exports", exact=True).click()
        with self.page.expect_download() as csv:
            self.page.get_by_role("button", name="Download CSV", exact=True).click()
        csv_text = Path(csv.value.path()).read_text()
        self.assertIn("name,kind,value,unit,uncertainty_m,valid,reason", csv_text.splitlines()[0])
        self.assertIn(label, csv_text)
        with self.page.expect_download() as gj:
            self.page.get_by_role("button", name="Download GeoJSON (local)", exact=True).click()
        geo = json.loads(Path(gj.value.path()).read_text())
        self.assertEqual(geo["type"], "FeatureCollection")
        self.assertIn("local", geo["coordinate_reference_system"].lower())
        self.assertTrue(any(f["properties"]["name"] == label for f in geo["features"]))

    def test_classification_panel_shows_area(self):
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        expect(self.page.locator(".class-list")).to_be_visible(timeout=20000)
        self.assertIn("m²", self.page.locator(".class-list").inner_text())

    def test_saved_measurement_renders_floating_label(self):
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        label = "QA line " + uuid.uuid4().hex[:6]
        created = self.page.evaluate(
            """async ({scene, label}) => {
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              const r = await fetch('/api/backend/api/workspace/measurements', {
                method: 'POST', headers: {'content-type': 'application/json'},
                body: JSON.stringify({scene, kind: 'distance', label, points: [[-2, 0, 0], [2, 0, 0]], model_revision: d.model_revision})});
              return r.status;
            }""", {"scene": SCENE, "label": label})
        self.assertEqual(created, 200)
        self.page.reload()
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        label_layer = frame.locator("#measure-labels")
        expect(label_layer).to_be_attached(timeout=15000)
        expect(label_layer.get_by_text(label, exact=False).first).to_be_visible(timeout=15000)

    def test_import_requires_video(self):
        self.page.goto(BASE, wait_until="domcontentloaded")
        self.page.get_by_role("button", name="New reconstruction", exact=True).click()
        dialog = self.page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        expect(dialog.get_by_role("button", name="Create project", exact=True)).to_be_disabled()
        dialog.get_by_role("textbox", name="Project name", exact=True).fill("Browser check")
        expect(dialog.get_by_role("button", name="Create project", exact=True)).to_be_disabled()
        self.page.keyboard.press("Escape")
        expect(dialog).not_to_be_visible()

    def test_mobile_has_no_page_overflow(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.goto(BASE, wait_until="domcontentloaded")
        expect(self.page.get_by_role("heading", name="Projects", exact=True)).to_be_visible()
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))

    def test_unavailable_backend_is_actionable(self):
        self.page.route("**/api/backend/api/workspace/projects", lambda route: route.fulfill(status=502, content_type="application/json", body='{"error":"Reconstruction service is offline"}'))
        self.page.goto(BASE, wait_until="domcontentloaded")
        expect(self.page.get_by_role("heading", name="Reconstruction service offline")).to_be_visible()
        expect(self.page.get_by_role("button", name="Retry connection", exact=True)).to_be_visible()
        self.assertEqual(self.errors, [])

    def test_place_tab_lists_library_and_renders_placed_item(self):
        self.page.goto(BASE + "/projects/room_w_jsonl")
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("button", name="Place", exact=True).click()
        # The furniture library renders with real metric dimensions.
        expect(self.page.locator(".furniture-grid").get_by_role("button", name="Coffee table")).to_be_visible(timeout=5000)
        # Create a placement through the API on the measured floor, then confirm the
        # viewer builds a collidable placed entity and the panel lists it after a poll.
        created = self.page.evaluate(
            """async () => {
              const scene = 'room_w_jsonl';
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              const r = await fetch('/api/backend/api/workspace/placements', {
                method: 'POST', headers: {'content-type': 'application/json'},
                body: JSON.stringify({scene, item: 'coffee_table', point: [0.45, 0, -1.55], yaw_deg: 0, model_revision: d.model_revision})});
              const j = await r.json();
              return {count: (j.placements || []).length, id: (j.placements || []).slice(-1)[0]?.id};
            }""")
        self.assertIsNotNone(created["id"])
        frame.wait_for_function("() => !!window.__app.root.findByName('placed')", timeout=15000)
        expect(self.page.locator(".place-list li")).to_have_count(created["count"], timeout=15000)
        self.assertEqual(self.errors, [])

    def test_click_to_place_drops_an_item_on_the_floor(self):
        # Uses the default scene, where the collision mesh is reliably pickable in
        # the overview (the same surface the height-tool test clicks). A pick while
        # the Place tool is armed must POST a placement and list it.
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("button", name="Place", exact=True).click()
        expect(self.page.locator(".workspace-place iframe")).to_be_visible(timeout=5000)
        self.page.locator(".furniture-grid").get_by_role("button", name="Coffee table").click()
        expect(self.page.locator(".place-draft")).to_be_visible(timeout=5000)
        bounds = frame.locator("canvas").bounding_box()
        # The editor writes on a debounce after the last change, so wait for the POST
        # rather than assuming a click blocks on the network.
        with self.page.expect_response(lambda r: r.url.endswith("/placements") and r.request.method == "POST", timeout=20000) as added:
            for fx in (0.45, 0.5, 0.55, 0.6):
                self.page.mouse.click(bounds["x"] + bounds["width"] * fx, bounds["y"] + bounds["height"] * 0.65)
                self.page.wait_for_timeout(200)
        self.assertEqual(added.value.status, 200, added.value.json())
        expect(self.page.locator(".place-list li")).not_to_have_count(0, timeout=15000)
        placed = self.page.evaluate(
            """async (scene) => {
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              return (d.placements || []).length;
            }""", SCENE)
        self.assertGreaterEqual(placed, 1, "no click landed on the measured floor to place an item")
        self.assertEqual(self.errors, [])

    def test_floor_plan_is_gone_and_measurement_undo_still_works(self):
        import re
        # The 2D floor plan was rejected outright: the editor surface is the 3D view,
        # and undo of the last measurement point must still work.
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("button", name="Place", exact=True).click()
        self.assertEqual(self.page.locator(".room-plan").count(), 0, "a 2D floor plan surface came back")
        for label in ("Plan view", "3D view"):
            self.assertEqual(self.page.get_by_role("button", name=label, exact=True).count(), 0)
        self.assertEqual(self.page.locator(".editor-dock").count(), 0, "the dock must stay hidden until a piece is selected")
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Distance", exact=True).click()
        bounds = frame.locator("canvas").bounding_box()
        for fx in (0.45, 0.55):
            self.page.mouse.click(bounds["x"] + bounds["width"] * fx, bounds["y"] + bounds["height"] * 0.55)
            self.page.wait_for_timeout(200)
        undo = self.page.get_by_role("button", name=re.compile(r"Undo last point"))
        expect(undo).to_be_visible(timeout=5000)
        self.assertIn("(2)", undo.inner_text())
        undo.click()
        expect(self.page.get_by_role("button", name=re.compile(r"Undo last point \(1\)"))).to_be_visible(timeout=5000)
        self.assertEqual(self.errors, [])

    def floor_point(self, scene):
        """One genuinely supported floor point, read from the baked collision grid.

        The placement tests need a spot the fit check will accept, so they take it
        from the same coverage/ground buffers the viewer collides against rather than
        hoping a hard-coded coordinate still lands on floor.
        """
        return self.page.evaluate("""async (scene) => {
          const root = `/runtime/work/${scene}/viewer_assets/`;
          const h = await (await fetch(root + 'collision.json')).json();
          const c = new Uint8Array(await (await fetch(root + 'coverage.u8')).arrayBuffer());
          const floor = new Float32Array(await (await fetch(root + 'ground.f32')).arrayBuffer());
          for (let z = 8; z < h.nz - 8; z++) for (let x = 8; x < h.nx - 8; x++) {
            const i = z * h.nx + x;
            if (c[i] && Number.isFinite(floor[i])) return [h.origin_xz[0] + (x + .5) * h.cell, floor[i], h.origin_xz[1] + (z + .5) * h.cell];
          }
          throw new Error('No supported floor cell in ' + scene);
        }""", scene)

    def create_placement(self, scene, point):
        return self.page.evaluate("""async ({scene, point}) => {
          const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
          const r = await fetch('/api/backend/api/workspace/placements', {
            method: 'POST', headers: {'content-type': 'application/json'},
            body: JSON.stringify({scene, item: 'coffee_table', point, yaw_deg: 0, model_revision: d.model_revision})});
          const j = await r.json();
          const p = (j.placements || []).slice(-1)[0];
          return {status: r.status, error: j.error, id: p && p.id, label: p && p.label, size: p && p.size,
                  x: p && p.center_xz[0], y: p && p.center_y, z: p && p.center_xz[1], yaw: p && p.yaw_deg};
        }""", {"scene": scene, "point": point})

    def test_placement_editor_move_rotate_resize_and_drag_persist(self):
        # The whole point of the Place screen: grab a piece in 3D and it moves, lifts,
        # turns and resizes for real. Every dock control writes through the debounced
        # save, and the viewer's placement-move stream drives the same path.
        self.page.goto(BASE + "/projects/test2train/place")
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        created = self.create_placement("test2train", self.floor_point("test2train"))
        self.assertEqual(created["status"], 200, created)
        row = self.page.locator(".place-list li").filter(has_text=created["label"])
        expect(row).to_be_visible(timeout=20000)
        row.get_by_role("button", name=f"Select {created['label']}", exact=True).click()
        expect(self.page.locator(".editor-dock")).to_be_visible(timeout=5000)
        # Slide: the readout answers before the write is acknowledged.
        with self.page.expect_response(lambda r: r.url.endswith("/placements/update"), timeout=20000) as sent:
            self.page.get_by_role("button", name="Move forward", exact=True).click()
        body = sent.value.request.post_data_json
        self.assertEqual(body["id"], created["id"])
        self.assertIn("size", body, "a write must carry the footprint, or a resize can never persist")
        self.assertAlmostEqual(body["point"][0], created["x"], delta=0.01)
        self.assertAlmostEqual(body["point"][2], created["z"] - 0.1, delta=0.02)
        # Raise, turn, grow.
        with self.page.expect_response(lambda r: r.url.endswith("/placements/update"), timeout=20000) as lifted:
            self.page.get_by_role("button", name="Raise piece", exact=True).click()
        self.assertAlmostEqual(lifted.value.request.post_data_json["point"][1], created["y"] + 0.1, delta=0.02)
        with self.page.expect_response(lambda r: r.url.endswith("/placements/update"), timeout=20000) as turned:
            self.page.get_by_role("button", name="Rotate right 45 degrees", exact=True).click()
        self.assertEqual(turned.value.request.post_data_json["yaw_deg"], (created["yaw"] + 45) % 360)
        with self.page.expect_response(lambda r: r.url.endswith("/placements/update"), timeout=20000) as grown:
            self.page.get_by_role("button", name="Increase length", exact=True).click()
        size = grown.value.request.post_data_json["size"]
        self.assertAlmostEqual(size[0], created["size"][0] + 0.1, delta=0.02)
        # Drag in 3D: x/z come from the scanned surface, and so does the height.
        with self.page.expect_response(lambda r: r.url.endswith("/placements/update"), timeout=20000) as dragged:
            frame.evaluate("""async ({id, point}) => {
              window.parent.postMessage({namespace: 'groundcontrol', type: 'placement-move', id, point, final: true}, window.location.origin);
            }""", {"id": created["id"], "point": [-2.34, 0.25, 1.78]})
        moved = dragged.value.request.post_data_json
        self.assertEqual([moved["point"][0], moved["point"][2]], [-2.34, 1.78])
        self.assertEqual(moved["point"][1], 0.25, "the height the viewer reports during a drag is the height to keep")
        expect(self.page.locator(".dock-head small")).to_contain_text("-2.34 m, 1.78 m", timeout=5000)
        self.page.reload()
        expect(self.page.locator(".place-list li").filter(has_text=created["label"])).to_contain_text("45°", timeout=20000)
        self.page.screenshot(path=str(Path(__file__).parents[1] / "scratch/placement-editor-tested.png"))
        self.assertEqual(self.errors, [])

    def test_placement_drag_moves_locally_then_saves_on_the_debounce(self):
        # A drag frame must not wait on the network: the local layout moves at once,
        # and only the newest position is written when the debounce elapses.
        self.page.goto(BASE + "/projects/test2train/place")
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        created = self.create_placement("test2train", self.floor_point("test2train"))
        self.assertEqual(created["status"], 200, created)
        row = self.page.locator(".place-list li").filter(has_text=created["label"])
        expect(row).to_be_visible(timeout=20000)
        row.get_by_role("button", name=f"Select {created['label']}", exact=True).click()
        for step in range(6):
            frame.evaluate("""async ({id, point}) => {
              window.parent.postMessage({namespace: 'groundcontrol', type: 'placement-move', id, point, final: false}, window.location.origin);
            }""", {"id": created["id"], "point": [created["x"] + (step + 1) * 0.1, created["y"], created["z"]]})
            self.page.wait_for_timeout(60)
        expect(self.page.locator(".dock-head small")).to_contain_text(f"{created['x'] + 0.6:.2f} m", timeout=5000)
        expect(self.page.locator(".editor-status")).to_contain_text("Layout saved.", timeout=20000)
        saved = self.page.evaluate("""async (id) => {
          const d = await (await fetch(`/api/backend/api/workspace/project?scene=test2train`)).json();
          return d.placements.find((p) => p.id === id) || null;
        }""", created["id"])
        self.assertIsNotNone(saved)
        self.assertAlmostEqual(saved["center_xz"][0], created["x"] + 0.6, delta=0.02)
        self.assertEqual(self.errors, [])

    def test_live_distance_before_save_and_height_undo_rearms(self):
        import re
        self.page.goto(BASE + "/projects/" + SCENE)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        self.page.get_by_role("button", name="Inspect frame 12", exact=True).click()
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Distance", exact=True).click()
        bounds = frame.locator("canvas").bounding_box()
        for x in (.45, .55):
            self.page.mouse.click(bounds["x"] + bounds["width"] * x, bounds["y"] + bounds["height"] * .65)
        expect(self.page.get_by_role("status", name="Live measurement")).to_contain_text(re.compile(r"\d+\.\d+"))
        expect(frame.locator("#measure-labels")).to_contain_text(re.compile(r"\d+\.\d+"))
        expect(self.page.get_by_role("button", name="Save to project", exact=True)).to_be_enabled()
        self.page.get_by_role("button", name="Height", exact=True).click()
        for x in (.45, .55):
            self.page.mouse.click(bounds["x"] + bounds["width"] * x, bounds["y"] + bounds["height"] * .65)
        self.page.get_by_role("button", name=re.compile(r"Undo last point")).click()
        # Let the undo land before clicking again: without this the next pick races
        # the state update, and the run depends on machine load, not behaviour.
        expect(self.page.locator(".measure-draft")).to_contain_text("1 selected.")
        self.page.mouse.click(bounds["x"] + bounds["width"] * .5, bounds["y"] + bounds["height"] * .65)
        expect(self.page.locator(".measure-draft")).to_contain_text("2 selected.")
        self.assertEqual(self.errors, [])

    def test_arena_drops_you_into_the_game_and_exits_clean(self):
        import re
        # The walkthrough screen is no longer a panel of tour controls: it is the game,
        # edge to edge, with only an exit affordance over the play area.
        self.page.goto(BASE + "/projects/rocks/walk")
        expect(self.page.get_by_role("heading", name="Arena", exact=True)).to_be_visible(timeout=30000)
        expect(self.page.get_by_role("button", name="Enter arena", exact=True)).to_be_enabled(timeout=30000)
        for stale in ("View 1", "Enter walkthrough", "Free flight", "Overview", "Start bot session", "Exit game session"):
            self.assertEqual(self.page.get_by_role("button", name=stale, exact=True).count(), 0, f"{stale} tour control came back")
        self.assertEqual(self.page.locator(".tour-map, .walk-keyboard, .nav-verdict").count(), 0)
        # The honest scale note survives, quietly, on the start surface only.
        expect(self.page.locator(".arena-scale-note")).to_contain_text("not survey-verified")
        self.page.get_by_role("button", name="5 Challenge", exact=True).click()
        self.page.get_by_role("button", name="Enter arena", exact=True).click()
        game = self.page.locator('iframe[title="Local game session"]')
        expect(game).to_have_attribute("src", re.compile(r"combat=1&bots=5$"), timeout=20000)
        handle = game.element_handle().content_frame()
        handle.wait_for_function("window.__combat || window.__combatError || window.__loadError", timeout=90000)
        self.assertIsNone(handle.evaluate("window.__combatError || window.__loadError || null"))
        self.assertTrue(handle.evaluate("!!window.__combat"))
        # Nothing is stacked over the live play area except the exit.
        self.assertEqual(self.page.locator(".viewport-message, .viewer-toolbar, .frame-strip").count(), 0)
        expect(self.page.get_by_role("button", name="Exit", exact=True)).to_be_visible()
        self.page.screenshot(path=str(Path(__file__).parents[1] / "scratch/game-tested.png"))
        self.page.get_by_role("button", name="Exit", exact=True).click()
        expect(self.page.get_by_role("heading", name="Arena", exact=True)).to_be_visible(timeout=30000)
        self.assertEqual(self.errors, [])

    def test_each_workspace_has_direct_entry_and_mobile_layout(self):
        for mode, heading in (("measure", "Measurement desk"), ("place", "Placement editor"), ("walk", "Arena")):
            with self.subTest(mode=mode):
                self.page.goto(BASE + f"/projects/test2train/{mode}")
                expect(self.page.get_by_role("heading", name=heading, exact=True)).to_be_visible(timeout=30000)
                self.page.set_viewport_size({"width": 390, "height": 844})
                self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"))
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
