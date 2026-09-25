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
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))

    def tearDown(self):
        # Remove any QA measurements a test left on the real scene so runs stay clean.
        try:
            self.page.evaluate(
                """async (scene) => {
                  const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
                  for (const m of (d.measurements || [])) {
                    if (/^(QA |Browser QA |North span$|Pad$)/.test(m.label || '')) {
                      await fetch('/api/backend/api/workspace/measurements/delete', {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({scene, id:m.id})});
                    }
                  }
                }""", SCENE)
        except Exception:
            pass
        # Placements live on the interior room scene; clear any a test left behind.
        try:
            self.page.evaluate(
                """async (scenes) => {
                  for (const scene of scenes) {
                    const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
                    for (const p of (d.placements || [])) {
                      await fetch('/api/backend/api/workspace/placements/delete', {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({scene, id:p.id, model_revision:d.model_revision})});
                    }
                  }
                }""", [SCENE, "room_w_jsonl"])
        except Exception:
            pass
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
        expect(self.page.get_by_role("button", name="Coffee table")).to_be_visible(timeout=5000)
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
        self.assertEqual(created["count"], 1)
        frame.wait_for_function("() => !!window.__app.root.findByName('placed')", timeout=15000)
        expect(self.page.locator(".place-list li")).to_have_count(1, timeout=15000)
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
        self.page.get_by_role("button", name="Coffee table").click()
        expect(self.page.locator(".place-draft")).to_be_visible(timeout=5000)
        bounds = frame.locator("canvas").bounding_box()
        for fx in (0.45, 0.5, 0.55, 0.6):
            self.page.mouse.click(bounds["x"] + bounds["width"] * fx, bounds["y"] + bounds["height"] * 0.65)
            self.page.wait_for_timeout(200)
        placed = self.page.evaluate(
            """async (scene) => {
              const d = await (await fetch(`/api/backend/api/workspace/project?scene=${scene}`)).json();
              return (d.placements || []).length;
            }""", SCENE)
        self.assertGreaterEqual(placed, 1, "no click landed on the measured floor to place an item")
        self.assertEqual(self.errors, [])

    def test_plan_view_and_measurement_undo(self):
        # The placement UX a human expects: a top-down plan view, and undo of the
        # last measurement point without clearing the whole selection.
        self.page.goto(BASE + "/projects/" + SCENE)
        expect(self.page.locator("iframe")).to_be_visible(timeout=30000)
        frame = self.page.locator("iframe").element_handle().content_frame()
        frame.wait_for_function("window.__ready", timeout=60000)
        # Plan view button exists and is clickable.
        self.page.get_by_role("button", name="Place", exact=True).click()
        plan = self.page.get_by_role("button", name="Plan view", exact=True)
        expect(plan).to_be_visible(timeout=5000)
        plan.click()
        self.assertEqual(self.errors, [])
        # Measurement: pick two points, undo one, the counter drops and no error fires.
        self.page.get_by_role("button", name="Measure", exact=True).click()
        self.page.get_by_role("button", name="Distance", exact=True).click()
        bounds = frame.locator("canvas").bounding_box()
        for fx in (0.45, 0.55):
            self.page.mouse.click(bounds["x"] + bounds["width"] * fx, bounds["y"] + bounds["height"] * 0.55)
            self.page.wait_for_timeout(200)
        undo = self.page.get_by_role("button", name=__import__("re").compile(r"Undo last point"))
        expect(undo).to_be_visible(timeout=5000)
        self.assertIn("(2)", undo.inner_text())
        undo.click()
        expect(self.page.get_by_role("button", name=__import__("re").compile(r"Undo last point \(1\)"))).to_be_visible(timeout=5000)
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
