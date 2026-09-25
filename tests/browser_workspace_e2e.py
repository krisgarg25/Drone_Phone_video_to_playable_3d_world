import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "videos/rocks.mp4")
    parser.add_argument("--run-gpu", action="store_true")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:3000", wait_until="domcontentloaded")
        expect(page.locator(".project-grid")).to_be_visible(timeout=30000)
        page.get_by_role("button", name="New reconstruction", exact=True).click()
        dialog = page.get_by_role("dialog")
        dialog.get_by_role("textbox", name="Project name", exact=True).fill("Rocks · integration check")
        dialog.get_by_label("Captured with", exact=True).select_option("drone")
        dialog.get_by_label("Your workflow", exact=True).select_option("inspection")
        dialog.get_by_label("Video and optional sensor files").set_input_files(str(args.video.resolve()))
        dialog.get_by_role("button", name="Create project", exact=True).click()
        page.wait_for_url("**/projects/*", timeout=60000)
        scene = page.url.rsplit("/", 1)[-1]
        expect(page.get_by_role("heading", name="Rocks · integration check", exact=True)).to_be_visible(timeout=30000)
        page.get_by_role("button", name="Details", exact=True).click()
        page.get_by_label("Project notes", exact=True).fill("Browser-verified import of the repository rocks sample; not survey ground truth.")
        page.get_by_role("button", name="Save details", exact=True).click()
        expect(page.get_by_text("Project details saved.", exact=True)).to_be_visible()
        page.reload()
        page.get_by_role("button", name="Details", exact=True).click()
        expect(page.get_by_label("Project notes", exact=True)).to_have_value("Browser-verified import of the repository rocks sample; not survey ground truth.")
        if args.run_gpu:
            page.get_by_role("button", name="Reconstruct", exact=True).click()
            dialog = page.get_by_role("dialog")
            dialog.get_by_label("Scene type", exact=True).select_option("drone")
            dialog.get_by_label("Processing quality", exact=True).select_option("smoke")
            dialog.get_by_role("checkbox").check()
            dialog.get_by_role("button", name="Start reconstruction", exact=True).click()
            expect(page.locator(".job-banner")).to_contain_text("Processing this capture", timeout=30000)
        result = {"scene": scene, "url": page.url, "started_gpu": args.run_gpu, "errors": errors, "time": time.time()}
        (ROOT / "scratch/workspace-e2e.json").write_text(json.dumps(result, indent=2))
        page.screenshot(path=str(ROOT / "scratch/workspace-e2e-start.png"), full_page=True)
        print(json.dumps(result))
        browser.close()
        assert not errors, errors


if __name__ == "__main__":
    main()
