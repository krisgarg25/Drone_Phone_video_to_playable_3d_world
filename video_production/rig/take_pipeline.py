"""The pipeline chapter, app side: a new project from the drone clip, started from the UI.

Loaded after demo.py (TAKEFILE=take_pipeline.py). It reuses take.py's helpers and
replaces setup/run. The take ends a few seconds into the camera solve; the run itself
keeps going for about 18 minutes, and the backend must be `backend-demo` (it sets
PIPELINE_TRAIN_PREVIEW so the trainer writes the fixed-camera timelapse).

STOP_BEFORE=create (rehearsal) stops with the dialog filled in, before a project exists;
STOP_BEFORE=start stops in the filled-in reconstruction dialog (a project now exists).
"""
exec(open(RIG / "take.py", encoding="utf-8").read())

VIDEO = ROOT / "videos" / "boulder-field-0aa9375f" / "rocks.mp4"
PROJECT_NAME = "Boulder field"
STOP = os.environ.get("STOP_BEFORE", "")


def set_select(label, value):
    """Choose an option of a React-controlled <select> (no native popup on screen)."""
    ok = js("""(() => {
      const s = [...document.querySelectorAll('select')].find(s => s.getAttribute('aria-label') === %s
                 || s.closest('label')?.innerText.trim().startsWith(%s));
      if (!s) return false;
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, %s);
      s.dispatchEvent(new Event('change', { bubbles: true }));
      return s.value === %s; })()""" % (json.dumps(label), json.dumps(label), json.dumps(value), json.dumps(value)))
    if not ok:
        raise LookupError(f"select {label} -> {value}")


def select_xy(label):
    return js("""(() => { const s = [...document.querySelectorAll('select')].find(s => s.getAttribute('aria-label') === %s
                 || s.closest('label')?.innerText.trim().startsWith(%s));
      const r = s.getBoundingClientRect(); return [r.left + r.width * 0.3, r.top + r.height / 2]; })()"""
              % (json.dumps(label), json.dumps(label)))


def attach(path):
    """The file the person would pick in the file chooser, set straight on the input."""
    root = fcdp("DOM.getDocument", depth=0)["root"]["nodeId"]
    node = fcdp("DOM.querySelector", nodeId=root, selector="dialog.import-dialog input[type=file]")["nodeId"]
    fcdp("DOM.setFileInputFiles", files=[str(path)], nodeId=node)


def job_step():
    return js("document.querySelector('.job-banner small')?.innerText || ''") or ""


def setup():
    cdp("Page.navigate", url=APP + "/")
    time.sleep(2.5)
    install_cursor()
    move((960, 330), dur=0.1)


def run(t):
    t.chapter("New project")
    with t.step("Projects page"):
        time.sleep(1.2)
    with t.step("New reconstruction"):
        click({"text": "New reconstruction", "exact": True}, after=0.9)
    with t.step("Name the project"):
        click({"sel": "dialog.import-dialog input[maxlength]"}, after=0.2)
        type_slow(PROJECT_NAME)
    with t.step("Add the drone video"):
        hover({"text": "browse files", "within": "dialog.import-dialog", "exact": True}, dwell=0.5)
        attach(VIDEO)
        wait_text("rocks.mp4", 5)
        time.sleep(0.8)
    with t.step("Captured with: drone"):
        move(select_xy("Captured with"))
        time.sleep(0.3)
        set_select("Captured with", "drone")
        time.sleep(0.8)
    if STOP == "create":
        return
    with t.step("Create project"):
        click({"text": "Create project", "within": "dialog.import-dialog"}, after=0.3)
        end = time.time() + 30
        while "/projects/" not in (js("location.pathname") or "") and time.time() < end:
            time.sleep(0.2)
        time.sleep(2.0)
        install_cursor()

    t.chapter("Start the reconstruction")
    with t.step("Open reconstruction"):
        click({"text": "Set up reconstruction", "exact": True}, after=1.0)
    with t.step("Quality: high"):
        move(select_xy("Processing quality"))
        time.sleep(0.3)
        set_select("Processing quality", "high")
        time.sleep(1.0)
    if STOP == "start":
        return
    with t.step("Confirm and start"):
        click({"sel": ".run-confirm input"}, after=0.5)
        click({"text": "Start reconstruction", "exact": True}, after=1.0)

    t.chapter("Processing")
    with t.step("Keyframes, then the camera solve begins"):
        end = time.time() + 45
        while "colmap" not in job_step() and time.time() < end:
            time.sleep(0.25)
        time.sleep(8.0)
    with t.step("Camera solve: logs stream"):
        move((1400, 640), dur=1.2)
        time.sleep(6.0)
