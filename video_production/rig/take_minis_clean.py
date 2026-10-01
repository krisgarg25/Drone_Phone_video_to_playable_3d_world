"""Feature minis, each on a clean scene (TAKEFILE=take_minis_clean.py, TAKE=minis-clean).

take_minis.py records the minis one after another in one session, so every mini shows
what the ones before it left behind (measurements, marks, tents). Here each mini starts
from the scene's clean state: its workspace is reset, the page reloaded, and the camera
set to the same default view, then only that mini's own set-up runs ("prep:" chapter)
before the "mini:" chapter the film uses.

ONLY=<comma list of mini names> runs just those.
"""
exec(open(RIG / "take_minis.py", encoding="utf-8").read())

HOLD = 1.2          # every mini ends on its result for this long


def fresh(scene, view):
    """Back to the scene's clean state: saved work archived, page reloaded, default view."""
    reset(scene)
    cdp("Page.navigate", url=f"{APP}/projects/{scene}")
    time.sleep(2.0)
    install_cursor()
    wait_ready()
    set_orbit(*view)
    move((960, 540), dur=0.1)
    time.sleep(0.8)


def setup():
    reset(ROCKS)
    reset(FLAT)
    (OUT / "raw" / "downloads").mkdir(parents=True, exist_ok=True)
    cdp("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(OUT / "raw" / "downloads"))
    cdp("Page.navigate", url=f"{APP}/projects/{ROCKS}")
    time.sleep(2.0)
    install_cursor()
    wait_ready()


def mission_prep(t, name):
    mode("Mission")
    cam_glide(*MEADOW, dur=1.0, wait=False)
    name_it('input[aria-label="Mission name"]', "Op Ridge", "Create mission")
    pick("Symbol", after=0.4)
    pick("Enemy post", after=0.3)
    click(world(*BOULDER_TOP), after=0.8)
    field("Facing", 180)
    tab("Mark", after=0.3)
    pick("Objective", after=0.3)
    click(world(*OBJECTIVE), after=0.8)
    tab("Mark", after=0.3)
    pick("Route", after=0.3)
    for pt in ROUTE:
        click(world(*pt), after=0.2)
    click({"text": "Finish", "exact": True}, after=0.8)


def scheme_prep():
    mode("Plan")
    name_it('input[aria-label="Scheme name"]', "Relief site", "Create scheme")
    pick("Proposed", after=0.4)


def plot_and_building():
    cam_glide(*FLAT_WIDE, dur=1.0)
    tab("Draw", after=0.3)
    click({"sel": "aside.inspector .plan-tool", "text": "Plot"}, after=0.4)
    for pt in [(-30, 0, -24), (0, 0, -24), (0, 0, -2), (-30, 0, -2)]:
        click(world(*pt), after=0.2)
    click({"text": "Finish", "exact": True}, after=1.0)
    field("Tallest building", 12)
    tab("Draw", after=0.3)
    click({"sel": "aside.inspector .plan-tool", "text": "Building"}, after=0.4)
    for pt in [(-24, 0, -18), (-8, 0, -18), (-8, 0, -8), (-24, 0, -8)]:
        click(world(*pt), after=0.2)
    click({"text": "Finish", "exact": True}, after=1.0)


def place(item, at):
    tab("Draw", after=0.3)
    if not js("[...document.querySelectorAll('aside.inspector select')].some(s => s.closest('label, .field')?.innerText.trim().startsWith('What to place'))"):
        click({"sel": "aside.inspector .plan-tool", "text": "Object"}, after=0.4)
    choose("What to place", item)
    time.sleep(0.3)
    click(world(*at), after=0.8)


# ---------------------------------------------------------------- one function per mini
def m_object_types(t):
    cam_glide(*HERO, dur=0.6)
    click({"sel": 'button[aria-label="Layers"]'}, after=0.4)
    mini(t, "Object types")
    with t.step("Object types: ground, rock, trees"):
        layer("Photo-real model", dwell=0.2)
        layer("Object types", dwell=1.4)
        drag_orbit(0.2, 0.0, 0.95, dur=2.6)


def _measure(name, pts, label):
    def f(t):
        mode("Measure")
        cam_glide(*HERO, dur=0.6)
        mini(t, name)
        with t.step(f"{name}: click on the model"):
            pick(name, after=0.4, exact=False)
            for p in pts:
                click(world(*p), after=0.35)
            time.sleep(1.0)
        with t.step(f"{name}: save"):
            name_it(f'input[placeholder^="{name}"]', label, "Save")
    return f


def m_note(t):
    mode("Measure")
    cam_glide(*BOULDER, dur=0.8)
    mini(t, "Note")
    with t.step("Note: one click, a line of text"):
        pick("Note", after=0.4, exact=False)
        click(world(*BOULDER_FACE), after=0.6)
        name_it('input[placeholder^="e.g. Crack"]', "Lichen on the north face", "Save")


def _heritage():
    mode("Inspect")
    pick("Heritage", after=0.5)
    cam_glide(*MEADOW, dur=0.8)


def m_section(t):
    _heritage()
    mini(t, "Cut a section")
    with t.step("Two ends across the boulders"):
        pick("Cut a section", after=0.5)
        pick("Click the two ends of the cut", after=0.4, exact=False)
        for p in SECTION:
            click(world(*p), after=0.5)
        time.sleep(2.0)


def m_archive(t):
    _heritage()
    mini(t, "Archive record")
    with t.step("Build the archive record"):
        pick("Archive record", after=0.5)
        pick("Build the archive record", after=0.3, exact=False)
        time.sleep(3.0)


def m_road(t):
    mode("Operations")
    cam_glide(*MEADOW, dur=0.8)
    mini(t, "Road access")
    with t.step("Start and goal: a route a vehicle can drive"):
        pick("Disaster", after=0.4)
        pick("Road access", after=0.5)
        choose("Compare with an earlier flight", "None")
        time.sleep(0.4)
        pick("Click where the vehicle starts", after=0.4, exact=False)
        click(world(*ROAD_FROM), after=0.5)
        click(world(*ROAD_TO), after=0.5)
        time.sleep(2.6)


def m_tiles(t):
    mode("Operations")
    cam_glide(*MEADOW, dur=0.8)
    mini(t, "Map tiles")
    with t.step("Make and download tiles"):
        pick("Border", after=0.4)
        pick("Map tiles", after=0.5)
        pick("Make and download tiles", after=0.3, exact=False)
        time.sleep(3.0)


def m_enemy(t):
    mission_prep(t, "Enemy guess")
    mini(t, "Enemy guess")
    with t.step("Where would they watch from?"):
        tab("Enemy guess", after=0.5)
        choose("Your approach route")
        time.sleep(0.5)
        pick("Find likely enemy positions", after=0.3, exact=False)
        time.sleep(3.2)


def m_brief(t):
    mission_prep(t, "Brief")
    mini(t, "Brief")
    with t.step("Sand table"):
        tab("Brief", after=0.5)
        pick("Sand table", after=0.3, exact=False)
        time.sleep(4.0)
    prep(t, "Brief: leave the sand table")
    click({"sel": 'button[aria-label="Close sand table"]'}, after=0.8)


def m_rehearse(t):
    mission_prep(t, "Rehearse")
    mini(t, "Rehearse")
    with t.step("Night, goggles, walk the route"):
        tab("Rehearse", after=0.5)
        pick("Night", after=0.3)
        pick("Night-vision goggles", after=0.3, exact=False)
        pick("Start the rehearsal", after=0.3, exact=False)
        time.sleep(3.0)
        click((960, 540), after=0.3)
        cursor_visible(False)
        key_down("w")
        time.sleep(2.4)
        key_up("w")
        look(300, 0, dur=1.4, start=(760, 540))
    prep(t, "Rehearse: leave")
    tap("Escape", after=0.6)
    cursor_visible(True)


def m_landing(t):
    mode("Mission")
    name_it('input[aria-label="Mission name"]', "Op Meadow", "Create mission")
    mini(t, "Landing zones")
    with t.step("Find flat, clear circles"):
        tab("Landing zones", after=0.5)
        pick("Find landing zones", after=0.3, exact=False)
        time.sleep(2.5)
        pick("Add to plan", after=1.4, exact=False)
        drag_orbit(0.25, 0.0, 0.95, dur=2.4)


def m_relief(t):
    scheme_prep()
    cam_glide([10, 0, 2], 46, 0.3, 0.62, dur=1.0)
    mini(t, "Relief camp")
    with t.step("Tents, a medical tent, a helipad"):
        for item, at in (("relief tent", (4, 0, 0)), ("relief tent", (10, 0, 0)), ("medical tent", (16, 0, 0)),
                         ("water bladder", (4, 0, 8)), ("helipad", (18, 0, 12))):
            place(item, at)
        drag_orbit(0.3, 0.0, 0.95, dur=2.4)


def m_logistics(t):
    scheme_prep()
    cam_glide([26, 0, -18], 150, 0.3, 1.05, dur=1.0)
    mini(t, "Site logistics")
    with t.step("A tower crane and its jib circle"):
        place("tower crane", (28, 0, -18))
        place("site office", (20, 0, -28))
        time.sleep(0.8)
        drag_orbit(0.3, 0.0, 0.95, dur=2.6)


def m_rules(t):
    scheme_prep()
    plot_and_building()
    mini(t, "Rules")
    with t.step("Raise it past the plot's 12 m limit"):
        for _ in range(4):
            click({"sel": 'aside.inspector button[aria-label="One floor more"]'}, after=0.45)
        tab("Rules", after=1.6)
        drag_orbit(0.2, 0.0, 0.95, dur=2.0)


def m_impact(t):
    scheme_prep()
    plot_and_building()
    for _ in range(2):
        click({"sel": 'aside.inspector button[aria-label="One floor more"]'}, after=0.4)
    mini(t, "Impact")
    with t.step("What the scheme changes"):
        tab("Impact", after=3.4)


def m_split(t):
    scheme_prep()
    plot_and_building()
    mini(t, "Split")
    with t.step("Existing and proposed side by side"):
        pick("Split", after=2.4)
        drag_orbit(0.25, 0.0, 0.95, dur=2.4)
    prep(t, "Split off")
    pick("Proposed", after=0.6)


def m_share(t):
    scheme_prep()
    plot_and_building()
    mini(t, "Share")
    with t.step("GIS and CAD exports"):
        tab("Share", after=0.6)
        for fmt in ("GeoJSON", "DXF", "3D Tiles", "CityJSON 2.0"):
            pick(fmt, after=0.45, exact=False)
        pick("Download CityJSON", after=1.6, exact=False)


ROCK_MINIS = [
    ("Object types", m_object_types, HERO),
    ("Distance", _measure("Distance", [BOULDER_FOOT, STONE], "Boulder to stone"), HERO),
    ("Area", _measure("Area", [MEADOW_A, MEADOW_B, MEADOW_C, MEADOW_D], "Meadow patch"), HERO),
    ("Volume", _measure("Volume", CLUSTER, "Boulder pile"), HERO),
    ("Note", m_note, HERO),
    ("Cut a section", m_section, HERO),
    ("Archive record", m_archive, HERO),
    ("Road access", m_road, HERO),
    ("Map tiles", m_tiles, HERO),
    ("Enemy guess", m_enemy, HERO),
    ("Brief", m_brief, HERO),
    ("Rehearse", m_rehearse, HERO),
]
FLAT_MINIS = [
    ("Landing zones", m_landing, FLAT_WIDE),
    ("Relief camp", m_relief, FLAT_WIDE),
    ("Site logistics", m_logistics, FLAT_WIDE),
    ("Rules", m_rules, FLAT_WIDE),
    ("Impact", m_impact, FLAT_WIDE),
    ("Split", m_split, FLAT_WIDE),
    ("Share", m_share, FLAT_WIDE),
]


def run(t):
    for scene, items in ((ROCKS, ROCK_MINIS), (FLAT, FLAT_MINIS)):
        for name, fn, view in items:
            if not want(name):
                continue
            prep(t, f"{name}: clean scene")
            fresh(scene, view)
            fn(t)
            time.sleep(HOLD)
