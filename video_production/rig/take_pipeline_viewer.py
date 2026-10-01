"""The pipeline chapter, viewer side: the project the pipeline take built, opened when Ready.

Loaded after demo.py (TAKEFILE=take_pipeline_viewer.py, SCENE=<project id>). A new camera
solve has its own coordinates, so nothing here uses rocks' world points: every move is an
orbit relative to the view the model opens on.
"""
exec(open(RIG / "take.py", encoding="utf-8").read())

SCENE = os.environ["SCENE"]


def setup():
    cdp("Page.navigate", url=f"{APP}/")
    time.sleep(2.5)
    install_cursor()
    move((960, 400), dur=0.1)


def run(t):
    t.chapter("Ready")
    with t.step("Projects: the new scan is ready"):
        hover({"text": "Boulder field"}, dwell=1.0)
    with t.step("Open it"):
        click({"text": "Boulder field"}, after=0.2)
        wait_ready()
        time.sleep(0.6)
    with t.step("Photo 41: fly to where it was taken (the quiz view)"):
        click({"sel": 'button[aria-label="Inspect frame 41"]'}, after=3.0)
    with t.step("Ease round"):
        drag_orbit(0.16, 0.0, 1.0, dur=3.0)

    t.chapter("Camera solve")
    with t.step("Feature points and the camera path"):
        click({"sel": 'button[aria-label="Layers"]'}, after=0.4)
        layer("Feature points", dwell=0.4)
        layer("Camera path", dwell=0.4)
        layer("Photo-real model", dwell=0.8)
        drag_orbit(-0.45, 0.3, 2.8, dur=4.2)
        time.sleep(0.6)

    t.chapter("Physics")
    with t.step("The solid surface"):
        layer("Feature points", dwell=0.2)
        layer("Camera path", dwell=0.2)
        layer("Solid surface", dwell=1.0)
        drag_orbit(0.45, -0.08, 0.85, dur=3.6)
        time.sleep(0.6)

    t.chapter("Photo-real")
    with t.step("Photo-real model back"):
        layer("Solid surface", dwell=0.2)
        layer("Photo-real model", dwell=1.0)
        click({"sel": 'button[aria-label="Close layers"]'}, after=0.3)
        click({"sel": 'button[aria-label="Inspect frame 41"]'}, after=3.2)
