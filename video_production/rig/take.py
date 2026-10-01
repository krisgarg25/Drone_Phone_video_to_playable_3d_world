"""The whole demo as ONE continuous take (video_production/STORYBOARD.md). Loaded after demo.py.

No reloads between features: the app is opened once from the projects page, every
workspace is reached from the header, the camera glides between views, and the only
page changes are the ones a person makes (Export page, back to projects, Open ground).
Each chapter becomes a section of TIMESTAMPS.md; each step a row.
"""

ROCKS, FLAT = "rocks_quality", "flat_ground"
# Rocks is only clean from the side the drone flew (yaw about -0.5..0.3).
HERO = ([4, -8, -37], 42, -0.2, 0.42)
BOULDER = ([2.5, -7.5, -38.5], 24, -0.15, 0.32)
MEADOW = ([6, -9, -33], 50, -0.2, 0.5)
FLIGHT = ([2, -6, -22], 72, -0.15, 0.55)
BOULDER_TOP = (0.44, -3.45, -40.35)
BOULDER_FOOT = (1.2, -10.9, -33.9)
BOULDER_FACE = (1.1, -7.7, -37.7)
FLOOD_SEED = (9.0, -11.4, -29.8)
FENCE = [(-14.7, -11.1, -37.7), (-2.8, -10.8, -33.8), (16.8, -11.3, -29.8), (28.7, -11.6, -29.8)]
POST = (24.7, -11.5, -25.9)
OBJECTIVE = (24.7, -11.5, -40.0)
ROUTE = [(-12, -11.1, -24), (-2, -11.3, -27), (10, -11.5, -30), OBJECTIVE]
FLAT_WIDE = ([0, 0, 4], 72, 0.3, 0.6)
BUILDING_A = [(0, 0, -6), (18, 0, -6), (18, 0, 6), (0, 0, 6)]
BUILDING_B = [(-20, 0, -12), (-8, 0, -12), (-8, 0, -1), (-20, 0, -1)]
ROAD = [(-44, 0, 20), (0, 0, 16), (44, 0, 22)]
PATIO_VIEW = ([-21.4, 0.3, 17.6], 5.2, 0.35, 0.42)


def mode(label, after=1.1):
    """Switch workspace from the header (a short hover shows its tooltip)."""
    x, y = find(text=label, within="nav.workspace-modes", exact=True)
    move((x, y))
    time.sleep(0.4)
    click((x, y), dwell=0.05, after=after)


def pick(text, after=0.5, **kw):
    click({"text": text, "within": "aside.inspector", "exact": True, **kw}, after=after)


def tab(text, after=0.5):
    click({"sel": "aside.inspector [role=tab], aside.inspector .insp-tasks button", "text": text}, after=after)


def layer(label, dwell=0.8):
    click({"sel": f'input[aria-label="{label}"]'}, after=dwell)


def field(label, value, within="aside.inspector"):
    """Type into the number field whose label contains `label`, then commit."""
    xy = js("""(() => { const l = [...document.querySelectorAll('%s label, %s .insp-setting')].find(l => l.innerText.includes(%s));
      const r = l.querySelector('input').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()"""
            % (within, within, json.dumps(label)))
    click(xy, after=0.15)
    select_all()
    type_slow(str(value), cps=12)
    tap("Enter", after=0.6)


def name_it(sel, text, button):
    click({"sel": sel}, after=0.2)
    type_slow(text)
    click({"text": button, "exact": True}, after=1.1)


def setup():
    reset(ROCKS)
    reset(FLAT)
    (OUT / "raw" / "downloads").mkdir(parents=True, exist_ok=True)
    cdp("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(OUT / "raw" / "downloads"))
    cdp("Page.navigate", url=APP + "/")
    time.sleep(2.5)
    install_cursor()
    move((960, 330), dur=0.1)


def run(t):
    # ------------------------------------------------------------ rocks
    t.chapter("Open a scan")
    with t.step("Projects page"):
        time.sleep(1.0)
    with t.step("Open the rocks scan"):
        hover({"text": "Rocks Quality"}, dwell=0.7)
        click({"text": "Rocks Quality"}, after=0.2)
        wait_ready()
        time.sleep(0.4)
    with t.step("Swing round to the model"):
        cam_glide(*HERO, dur=2.6)
        drag_orbit(0.28, 0.0, 0.92, dur=2.4)

    t.chapter("Explore")
    with t.step("Panel tabs: Quality, Project, Files"):
        for label in ["Quality", "Project", "Files", "Overview"]:
            pick(label, after=0.6)
    with t.step("Layers: the drone's flight path"):
        click({"sel": 'button[aria-label="Layers"]'}, after=0.5)
        layer("Camera path", dwell=0.3)
        cam_glide(*FLIGHT, dur=2.4)
        time.sleep(0.8)
        layer("Camera path", dwell=0.3)
    with t.step("Collision mesh on"):
        layer("Solid surface", dwell=1.0)
        layer("Photo-real model", dwell=0.4)
        cam_glide(*HERO, dur=2.2)
        drag_orbit(-0.3, 0.06, 1.0, dur=2.4)
    with t.step("Photo-real model back"):
        layer("Photo-real model", dwell=0.8)
        layer("Solid surface", dwell=0.4)
        click({"sel": 'button[aria-label="Close layers"]'}, after=0.3)
    with t.step("Click a photo: fly to where it was taken"):
        click({"sel": 'button[aria-label="Inspect frame 9"]'}, after=2.4)

    t.chapter("Measure")
    with t.step("Open Measure"):
        mode("Measure")
        cam_glide(*BOULDER, dur=2.0)
    with t.step("Tools: Distance, Area, Volume, Note"):
        for label in ["Distance", "Area", "Volume", "Note"]:
            hover({"text": label, "within": "aside.inspector"}, dwell=0.35)
    with t.step("Height: foot, then top"):
        pick("Height", after=0.5, exact=False)
        click(world(*BOULDER_FOOT), after=0.4)
        x, y = world(*BOULDER_TOP)
        move((x, y + 60), dur=0.8)
        click((x, y), after=0.9)
    with t.step("Name and save"):
        name_it('input[placeholder^="Height"]', "Boulder height", "Save")

    t.chapter("Inspect")
    with t.step("Open Inspect"):
        mode("Inspect")
    with t.step("Tasks: Defect list, Pole tilt, Cable sag, What moved"):
        for label in ["Defect list", "Pole tilt", "Cable sag", "What moved", "Log a defect"]:
            pick(label, after=0.45)
    with t.step("Click a spot on the boulder"):
        pick("Click the defect on the model", after=0.4, exact=False)
        click(world(*BOULDER_FACE), after=0.3)
    with t.step("Every photo that saw that spot"):
        wait_text("photos saw this spot", timeout=20)
        time.sleep(1.8)
    with t.step("View from the photo"):
        click({"text": "View from here"}, after=2.4)
    with t.step("Name it, save to the list"):
        click({"sel": "aside.inspector .insp-form input"}, after=0.2)
        type_slow("Weathered face")
        pick("Save to the defect list", after=1.4, exact=False)
    with t.step("Heritage: local relief draped on the model"):
        pick("Heritage", after=0.6)
        pick("Relief maps", after=0.5)
        cam_glide(*MEADOW, dur=2.0, wait=False)
        pick("Local relief", after=2.4)
        drag_orbit(0.25, 0.0, 1.0, dur=2.2)
        pick("Photo map", after=0.8)
        pick("Structures", after=0.4)

    t.chapter("Operations")
    with t.step("Open Operations"):
        mode("Operations")
        cam_glide(*MEADOW, dur=1.6)
    with t.step("Disaster tasks"):
        for label in ["Building damage", "Debris volume", "People & vehicles", "Road access"]:
            pick(label, after=0.4)
    with t.step("Flood: click where the water comes in"):
        pick("Flood", after=0.5)
        pick("Click where the water comes in", after=0.4, exact=False)
        click(world(*FLOOD_SEED), after=0.3)
    with t.step("The valley fills"):
        wait_text("under water", timeout=30)
        time.sleep(1.4)
        drag_orbit(0.22, 0.04, 0.95, dur=2.2)
    with t.step("Construction tasks"):
        pick("Construction", after=0.6)
        for label in ["Cut & fill", "Built vs design", "What changed", "Site logistics"]:
            pick(label, after=0.4)
    with t.step("Border: line coverage"):
        pick("Border", after=0.6)
        pick("Line coverage", after=0.5)
    with t.step("Draw the fence line"):
        pick("Draw the border or fence line", after=0.4, exact=False)
        for pt in FENCE:
            click(world(*pt), after=0.2)
        click({"text": "Finish", "exact": True}, after=0.5)
    with t.step("Place an observation post"):
        pick("Place observation posts", after=0.4, exact=False)
        click(world(*POST), after=0.5)
    with t.step("What the post can see"):
        pick("Check what the posts can see", after=0.3, exact=False)
        wait_text("of the line seen", timeout=30)
        time.sleep(2.2)

    t.chapter("Twin")
    with t.step("Open Twin"):
        mode("Twin")
        cam_glide(*HERO, dur=1.6)
    with t.step("As measured"):
        pick("As measured", after=1.8)
    with t.step("How sure"):
        pick("How sure", after=1.8)
    with t.step("As it looks"):
        pick("As it looks", after=1.0)
    with t.step("Versions, game engine export"):
        pick("Versions", after=0.8, exact=False)
        pick("Game engine", after=0.9, exact=False)

    t.chapter("Mission")
    with t.step("Open Mission, name it"):
        mode("Mission")
        cam_glide(*MEADOW, dur=1.6, wait=False)
        name_it('input[aria-label="Mission name"]', "Op Ridge", "Create mission")
    with t.step("Symbol: enemy post on the boulders"):
        pick("Symbol", after=0.5)
        pick("Enemy post", after=0.4)
        click(world(*BOULDER_TOP), after=1.0)
    with t.step("Turn it to face the meadow"):
        field("Facing", 180)
        field("Arc it covers", 120)
    with t.step("Symbol: the objective"):
        tab("Mark", after=0.4)
        pick("Objective", after=0.4)
        click(world(*OBJECTIVE), after=1.0)
    with t.step("Route to the objective"):
        tab("Mark", after=0.4)
        pick("Route", after=0.4)
        for pt in ROUTE:
            click(world(*pt), after=0.25)
        click({"text": "Finish", "exact": True}, after=1.0)
    with t.step("Exposure: what the enemy sees of the route"):
        tab("Exposure", after=0.5)
        pick("Check what can see each route", after=0.3, exact=False)
        wait_text("seen by the enemy", timeout=30)
        time.sleep(1.2)
        drag_orbit(-0.22, 0.05, 0.95, dur=2.2)
    with t.step("Terrain: where vehicles can move"):
        tab("Terrain", after=0.5)
        pick("Wheels", after=0.4)
        pick("Map where we can move", after=0.3, exact=False)
        wait_text("free to move", timeout=30)
        time.sleep(2.0)

    t.chapter("Walk")
    with t.step("Open Walk"):
        mode("Walk", after=1.0)
    with t.step("Start walking"):
        click({"sel": '[aria-label="Bot count"] button', "nth": 0}, after=0.5)
        click({"text": "Start walking"}, after=0.5)
        wait_text("Game session", timeout=60)
        time.sleep(2.0)
        click((960, 540), after=0.4)
        cursor_visible(False)
    with t.step("Look over the meadow"):
        look(420, 0, dur=1.8, start=(760, 540))
    with t.step("Walk"):
        key_down("w")
        time.sleep(2.2)
        key_up("w")
    with t.step("Look back at the boulders"):
        look(-640, 20, dur=2.0, start=(1180, 540))
    with t.step("Free the mouse"):
        tap("Escape", after=0.6)
        cursor_visible(True)

    t.chapter("Export")
    with t.step("Open Export"):
        click({"text": "Export", "within": "header", "exact": True}, after=1.6)
        install_cursor()
    with t.step("Pick files"):
        click({"sel": 'input[aria-label="Include Point cloud"]'}, after=0.35)
        click({"sel": 'input[aria-label="Include Labelled point cloud"]'}, after=0.35)
    with t.step("Rename one"):
        click({"sel": 'input[aria-label="File name for Point cloud"]'}, after=0.2)
        select_all()
        type_slow("site_A_pointcloud")
        time.sleep(0.3)
    with t.step("One .zip, download"):
        click({"text": "One .zip", "exact": True}, after=0.4)
        click({"text": "Review & download", "exact": True}, after=1.0)
        click({"text": "Download", "exact": True}, after=1.6)
        click({"text": "Close", "exact": True}, after=0.5)

    # ------------------------------------------------------------ flat ground
    t.chapter("Open ground")
    with t.step("Back to projects"):
        click({"sel": 'a[href="/"]', "text": "Projects"}, after=1.4)
        install_cursor()
    with t.step("Open the flat ground"):
        hover({"text": "Open ground"}, dwell=0.5)
        click({"text": "Open ground"}, after=0.2)
        wait_ready()
        time.sleep(0.3)
        cam_glide(*FLAT_WIDE, dur=2.4)

    t.chapter("Plan")
    with t.step("Open Plan, name the scheme"):
        mode("Plan")
        name_it('input[aria-label="Scheme name"]', "Scheme A", "Create scheme")
        pick("Proposed", after=0.4)
    with t.step("Tasks: Rules, Impact, Existing"):
        for label in ["Rules", "Impact", "Existing", "Draw"]:
            tab(label, after=0.4)
    with t.step("Draw a building"):
        click({"sel": "aside.inspector .plan-tool", "text": "Building"}, after=0.4)
        for pt in BUILDING_A:
            click(world(*pt), after=0.2)
        click({"text": "Finish", "exact": True}, after=1.0)
    with t.step("Raise it: more floors"):
        for _ in range(4):
            click({"sel": 'aside.inspector button[aria-label="One floor more"]'}, after=0.45)
        time.sleep(0.6)
    with t.step("A second building"):
        tab("Draw", after=0.4)
        click({"sel": "aside.inspector .plan-tool", "text": "Building"}, after=0.4)
        for pt in BUILDING_B:
            click(world(*pt), after=0.2)
        click({"text": "Finish", "exact": True}, after=1.0)
    with t.step("A road past them"):
        tab("Draw", after=0.4)
        click({"sel": "aside.inspector .plan-tool", "text": "Road"}, after=0.4)
        for pt in ROAD:
            click(world(*pt), after=0.25)
        click({"text": "Finish", "exact": True}, after=1.0)
        drag_orbit(0.3, 0.0, 0.9, dur=2.4)
    with t.step("Swipe: existing against proposed"):
        pick("Swipe", after=2.6)
        drag((960, 540), (1330, 540), dur=1.2)
        drag((1330, 540), (240, 540), dur=2.2)
        drag((240, 540), (820, 540), dur=1.2)
        time.sleep(0.4)
        pick("Proposed", after=1.0)
    with t.step("Sunlight: shadows through the day"):
        tab("Sunlight", after=0.5)
        pick("Show shadows on the model", after=2.4, exact=False)
        r = js("(() => { const i = document.querySelector('input[aria-label=\"Time of day\"]'); const b = i.getBoundingClientRect(); return [b.left, b.top + b.height / 2, b.width, +i.value, +i.min, +i.max]; })()")
        x0 = r[0] + r[2] * (r[3] - r[4]) / (r[5] - r[4])
        drag((x0, r[1]), (r[0] + r[2] * 0.12, r[1]), dur=1.6)      # morning
        drag((r[0] + r[2] * 0.12, r[1]), (r[0] + r[2] * 0.72, r[1]), dur=4.0)   # to late afternoon
        time.sleep(1.5)

    t.chapter("Place")
    with t.step("Open Place"):
        mode("Place")
        cam_glide(*PATIO_VIEW, dur=2.2)
    with t.step("Coffee table on the patio"):
        click({"text": "Coffee table"}, after=0.4)
        click(world(-22, 0, 18.2), after=0.8)
    with t.step("A sofa behind it"):
        click({"text": "Add", "exact": True}, after=0.4)
        click({"text": "Sofa (3-seat)"}, after=0.4)
        click(world(-22, 0, 16.6), after=0.8)
    with t.step("An armchair to the side"):
        click({"text": "Add", "exact": True}, after=0.4)
        click({"text": "Armchair"}, after=0.4)
        click(world(-20.3, 0, 18.3), after=0.8)
    with t.step("Look around the layout"):
        drag_orbit(-0.5, 0.05, 1.0, dur=3.0, at=(1000, 600))
