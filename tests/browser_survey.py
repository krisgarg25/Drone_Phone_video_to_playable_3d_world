import functools
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from _serve import H
from playwright.sync_api import sync_playwright, expect
from survey_georef import normalize_telemetry


def main():
    output = ROOT / "scratch/survey_qa"
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix="survey-browser-") as directory:
        fixture = Path(directory)
        (fixture / "viewer").mkdir()
        for name in ("pipeline_gui.html", "survey_dashboard.js"):
            shutil.copyfile(ROOT / "viewer" / name, fixture / "viewer" / name)
        source = fixture / "videos/cpu_fixture"
        source.mkdir(parents=True)
        (source / "video.mp4").write_bytes(b"synthetic fixture; no decoding or GPU")
        metadata = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
                    "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
                    "single_pass": True, "video_duration_s": 600}
        csv_text = ("t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
                    "0,28,77,100,1,2\n1,28.0001,77,100,1,2\n"
                    "2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n")
        files = fixture / "input_files"
        files.mkdir()
        csv_path, meta_path = files / "telemetry.csv", files / "flight_metadata.json"
        csv_path.write_text(csv_text)
        meta_path.write_text(json.dumps(metadata))
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(fixture)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome", headless=True,
                    args=["--disable-gpu", "--disable-webgl", "--disable-extensions"])
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                page = context.new_page()
                errors, run_requests, model_requests = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: run_requests.append(request.url) if "/api/run" in request.url else None)
                page.on("request", lambda request: model_requests.append(request.url) if "/viewer/pc.html" in request.url else None)
                page.goto(f"http://127.0.0.1:{server.server_port}/viewer/pipeline_gui.html?scene=cpu_fixture")
                expect(page.locator("#survey-status")).to_have_text("Not prepared")
                expect(page.locator(".survey-criterion")).to_have_count(6)
                expect(page.locator("#view-frame")).to_have_attribute("src", "about:blank")
                checks.append("Six criteria visible, no hidden model or GPU request")
                page.screenshot(path=str(output / "desktop_unprepared.png"), full_page=True)

                def fill_inputs():
                    page.locator("#survey-csv").set_input_files(str(csv_path))
                    page.locator("#survey-metadata").set_input_files(str(meta_path))

                bad = dict(metadata, altitude_datum="unknown")
                meta_path.write_text(json.dumps(bad))
                fill_inputs()
                page.locator("#survey-save").click()
                expect(page.locator("#survey-error")).to_contain_text("400")
                page.locator("#survey-refresh").click()
                expect(page.locator("#survey-status")).to_have_text("Not prepared")
                meta_path.write_text(json.dumps(metadata))
                fill_inputs()
                page.locator("#survey-save").click()
                expect(page.locator("#survey-message")).to_contain_text("CPU operation complete")
                fill_inputs()
                page.locator("#survey-save").click()
                expect(page.locator("#survey-error")).to_contain_text("409")
                page.locator("#survey-refresh").click()
                expect(page.locator("#survey-status")).to_have_text("Not prepared")
                page.locator("#survey-prepare").click()
                expect(page.locator("#survey-status")).to_have_text("Prepared")
                page.locator("#survey-align").click()
                expect(page.locator("#survey-error")).to_contain_text("camera")
                checks.append("Invalid metadata, overwrite conflict and missing cameras produce actionable errors")

                work = fixture / "work/cpu_fixture"
                telemetry = normalize_telemetry(csv_path, metadata)
                camera_rows = [{"file": f"frame_{i}.jpg", "t_sec": sample["t_sec"],
                    "camera": {"R_rowmajor": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               "t": [-v for v in sample["position"]]}}
                    for i, sample in enumerate(telemetry["samples"])]
                (work / "keyframes_poses.jsonl").write_text("\n".join(json.dumps(row) for row in camera_rows))
                page.locator("#survey-refresh").click()
                expect(page.locator("#survey-status")).to_have_text("Prepared")
                page.locator("#survey-align").click()
                expect(page.locator("#survey-status")).to_have_text("Aligned")
                page.locator("#survey-evaluate").click()
                expect(page.locator("#survey-status")).to_have_text("Evaluated")
                expect(page.locator("#survey-criteria")).not_to_contain_text("Meets target")
                alignment = json.loads((work / "survey/georeference.json").read_text())
                (work / "survey/checkpoints.json").write_text(json.dumps({"independent": True,
                    "alignment": "none", "coordinate_frame": alignment["coordinate_frame"],
                    "checkpoints": [{"reconstructed": [0.1, 0, 0], "reference": [0, 0, 0]},
                                    {"reconstructed": [1, 1.2, 1], "reference": [1, 1, 1]}]}))
                page.locator("#survey-evaluate").click()
                expect(page.locator(".survey-criterion").first).to_contain_text("Measured")
                expect(page.locator(".survey-criterion").first).to_contain_text("0.16 m")
                with page.expect_download() as download_event:
                    page.get_by_role("link", name="evaluation.json", exact=True).click()
                downloaded = json.loads(Path(download_event.value.path()).read_text())
                assert downloaded["criteria"][0]["status"] == "measured"
                checks.append("Synthetic alignment, independent checkpoint display and downloadable report work")
                page.screenshot(path=str(output / "desktop_evaluated.png"), full_page=True)

                page.locator("#lane-btn-run").click()
                page.once("dialog", lambda dialog: dialog.dismiss())
                page.locator("#btn-start").click()
                assert not run_requests
                page.locator("#lane-btn-capture").click()
                expect(page.locator("#lane-capture")).to_be_visible()
                page.locator("#lane-btn-survey").click()
                checks.append("Run and Capture remain accessible; declining GPU confirmation sends no run request")

                page.set_viewport_size({"width": 390, "height": 844})
                expect(page.locator("#survey-title")).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                page.screenshot(path=str(output / "mobile.png"), full_page=True)
                # :focus-visible only matches keyboard focus, so drive it with Tab.
                page.locator("#survey-scene").focus()
                page.keyboard.press("Tab")
                focused = page.evaluate("() => document.activeElement.id")
                assert focused, "keyboard focus did not land on a control"
                outline = page.evaluate("() => getComputedStyle(document.activeElement).outlineStyle")
                assert outline != "none", f"{focused} has no keyboard focus indicator"
                checks.append(f"390px layout has no horizontal overflow; Tab focus on #{focused} is visible")

                context.set_offline(True)
                page.locator("#survey-refresh").click()
                expect(page.locator("#survey-error")).to_be_visible()
                context.set_offline(False)
                page.locator("#survey-refresh").click()
                expect(page.locator("#survey-status")).to_have_text("Evaluated")
                checks.append("Offline failure and recovery work")
                assert not errors, errors
                assert not model_requests, model_requests
                assert not run_requests, run_requests
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
        result = {"status": "passed", "gpu": "disabled; WebGL disabled", "fixture": "synthetic CPU-only, not accuracy evidence",
                  "checks": checks, "page_errors": errors, "gpu_run_requests": run_requests, "model_requests": model_requests}
        (output / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
