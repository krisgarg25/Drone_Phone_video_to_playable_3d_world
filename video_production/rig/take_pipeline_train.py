"""The pipeline chapter, mid-run: the project's live processing log while the splat trains.

Loaded after demo.py (TAKEFILE=take_pipeline_train.py, SCENE=<project id>). Run it during
the train step of the run take_pipeline.py started; it only watches.
"""
exec(open(RIG / "take.py", encoding="utf-8").read())

SCENE = os.environ["SCENE"]


def setup():
    cdp("Page.navigate", url=f"{APP}/projects/{SCENE}")
    time.sleep(3.0)
    install_cursor()
    move((1180, 520), dur=0.1)
    if not js("!!document.querySelector('[aria-label=\"Processing log\"][aria-pressed=\"true\"]')"):
        click({"sel": '[aria-label="Processing log"]'}, after=1.0)
    js("(() => { const p = document.querySelector('pre.job-terminal'); if (p) p.scrollTop = p.scrollHeight; })()")
    time.sleep(2.0)


def run(t):
    t.chapter("Training")
    with t.step("Job banner: train"):
        hover({"sel": ".job-banner"}, dwell=2.5)
    with t.step("Training log streams"):
        move((420, 950), dur=1.2)
        # The log follows the new lines as they land.
        end = time.time() + 11.0
        while time.time() < end:
            js("(() => { const p = document.querySelector('pre.job-terminal'); if (p) p.scrollTop = p.scrollHeight; })()")
            time.sleep(0.25)
