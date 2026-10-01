"""Feature minis: short results of the features the main take does not use.

Loaded after demo.py (TAKEFILE=take_minis.py). One recording, one chapter per mini
("mini: <name>"); the moves in between are "prep" chapters the edit leaves out. The film
shows each mini as a small window, so every one ends on a readable result.

ONLY=<comma list of mini names> runs just those (for rehearsal); prep still runs.
"""
exec(open(RIG / "take.py", encoding="utf-8").read())

ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}


def choose(label, text=None):
    """Pick an option of the <select> whose label starts with `label`: the one whose text
    starts with `text`, or the first real option. React sees a normal change event."""
    ok = js("""(() => {
      const s = [...document.querySelectorAll('aside.inspector select')].find(s => s.getAttribute('aria-label')?.startsWith(%s)
                 || s.closest('label, .insp-setting, .field')?.innerText.trim().startsWith(%s));
      if (!s) return 'no select';
      const want = %s;
      const o = [...s.options].find(o => want ? o.text.trim().startsWith(want) : o.value && !o.disabled);
      if (!o) return 'no option: ' + [...s.options].map(o => o.text).join(' / ');
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, o.value);
      s.dispatchEvent(new Event('change', { bubbles: true }));
      return true; })()""" % (json.dumps(label), json.dumps(label), json.dumps(text)))
    if ok is not True:
        raise LookupError(f"choose {label} / {text}: {ok}")


def want(name):
    return not ONLY or name in ONLY


def mini(t, name):
    t.chapter(f"mini: {name}")


def prep(t, what):
    t.chapter(f"prep: {what}")


# Meadow points in front of the boulders (rocks scene units, as in take.py).
MEADOW_A = (-6.0, -11.2, -26.0)
MEADOW_B = (6.0, -11.4, -28.0)
MEADOW_C = (9.0, -11.5, -22.0)
MEADOW_D = (-3.0, -11.2, -21.0)
STONE = (9.0, -11.4, -29.8)
CLUSTER = [(-10.0, -10.9, -33.0), (9.0, -11.2, -30.5), (15.0, -11.0, -37.0), (-8.0, -10.6, -41.0)]
SECTION = [(-14.0, -11.0, -36.0), (18.0, -11.4, -38.0)]
ROAD_FROM = (-12.0, -11.1, -24.0)
ROAD_TO = (24.7, -11.5, -40.0)


def setup():
    # reset() archives what earlier takes saved (the person's measurements included);
    # run.py puts rocks' own measurements and plan back after the take.
    reset(ROCKS)
    reset(FLAT)
    (OUT / "raw" / "downloads").mkdir(parents=True, exist_ok=True)
    cdp("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(OUT / "raw" / "downloads"))
    cdp("Page.navigate", url=f"{APP}/projects/{ROCKS}")
    time.sleep(2.0)
    install_cursor()
    wait_ready()
    set_orbit(*HERO)
    move((960, 540), dur=0.1)


def run(t):
    # ------------------------------------------------------------ Explore layers
    # Feature points are the pipeline chapter's shot; Camera coverage draws nothing
    # readable on rocks, so it is a card. Object types reads only without the splat.
    if want("Object types"):
        prep(t, "Object types")
        cam_glide(*HERO, dur=1.0)
        click({"sel": 'button[aria-label="Layers"]'}, after=0.4)
        mini(t, "Object types")
        with t.step("Object types: ground, rock, trees"):
            layer("Photo-real model", dwell=0.2)
            layer("Object types", dwell=1.4)
            drag_orbit(0.2, 0.0, 0.95, dur=2.6)
            time.sleep(0.6)
        prep(t, "Object types off")
        layer("Object types", dwell=0.2)
        layer("Photo-real model", dwell=0.3)
        click({"sel": 'button[aria-label="Close layers"]'}, after=0.3)

    # ------------------------------------------------------------ Measure
    if any(want(n) for n in ("Distance", "Area", "Volume", "Note")):
        prep(t, "Measure")
        mode("Measure")
        cam_glide(*HERO, dur=1.0)
    for name, pts in (("Distance", [BOULDER_FOOT, STONE]),
                      ("Area", [MEADOW_A, MEADOW_B, MEADOW_C, MEADOW_D]),
                      ("Volume", CLUSTER)):
        if not want(name):
            continue
        mini(t, name)
        with t.step(f"{name}: click on the model"):
            pick(name, after=0.4, exact=False)
            for p in pts:
                click(world(*p), after=0.35)
            time.sleep(1.4)
        with t.step(f"{name}: save"):
            name_it(f'input[placeholder^="{name}"]',
                    {"Distance": "Boulder to stone", "Area": "Meadow patch", "Volume": "Boulder pile"}[name], "Save")
    if want("Note"):
        mini(t, "Note")
        with t.step("Note: one click, a line of text"):
            cam_glide(*BOULDER, dur=1.0)
            pick("Note", after=0.4, exact=False)
            click(world(*BOULDER_FACE), after=0.6)
            name_it('input[placeholder^="e.g. Crack"]', "Lichen on the north face", "Save")

    # ------------------------------------------------------------ Inspect / Heritage
    if want("Cut a section") or want("Archive record"):
        prep(t, "Heritage")
        mode("Inspect")
        pick("Heritage", after=0.5)
        cam_glide(*MEADOW, dur=1.0)
    if want("Cut a section"):
        mini(t, "Cut a section")
        with t.step("Two ends across the boulders"):
            pick("Cut a section", after=0.5)
            pick("Click the two ends of the cut", after=0.4, exact=False)
            for p in SECTION:
                click(world(*p), after=0.5)
            time.sleep(2.4)
    if want("Archive record"):
        mini(t, "Archive record")
        with t.step("Build the archive record"):
            pick("Archive record", after=0.5)
            pick("Build the archive record", after=0.3, exact=False)
            time.sleep(3.0)

    # ------------------------------------------------------------ Operations
    if want("Road access") or want("Map tiles"):
        prep(t, "Operations")
        mode("Operations")
        cam_glide(*MEADOW, dur=1.0)
    if want("Road access"):
        mini(t, "Road access")
        with t.step("Start and goal: a route a vehicle can drive"):
            pick("Disaster", after=0.4)
            pick("Road access", after=0.5)
            choose("Compare with an earlier flight", "None")
            time.sleep(0.4)
            pick("Click where the vehicle starts", after=0.4, exact=False)
            click(world(*ROAD_FROM), after=0.5)
            click(world(*ROAD_TO), after=0.5)
            time.sleep(3.0)
    if want("Map tiles"):
        mini(t, "Map tiles")
        with t.step("Make and download tiles"):
            pick("Border", after=0.4)
            pick("Map tiles", after=0.5)
            pick("Make and download tiles", after=0.3, exact=False)
            time.sleep(3.0)

    # ------------------------------------------------------------ Mission
    if any(want(n) for n in ("Enemy guess", "Rehearse", "Brief")):
        prep(t, "Mission: post, objective, route")
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
    if want("Enemy guess"):
        mini(t, "Enemy guess")
        with t.step("Where would they watch from?"):
            tab("Enemy guess", after=0.5)
            choose("Your approach route")
            time.sleep(0.5)
            pick("Find likely enemy positions", after=0.3, exact=False)
            time.sleep(3.5)
    if want("Brief"):
        mini(t, "Brief")
        with t.step("Sand table"):
            tab("Brief", after=0.5)
            pick("Sand table", after=0.3, exact=False)
            time.sleep(4.0)
        prep(t, "Brief: leave the sand table")
        click({"sel": 'button[aria-label="Close sand table"]'}, after=0.8)
    if want("Rehearse"):
        mini(t, "Rehearse")
        with t.step("Night, goggles, walk the route"):
            tab("Rehearse", after=0.5)
            pick("Night", after=0.3)
            pick("Night-vision goggles", after=0.3, exact=False)
            pick("Start the rehearsal", after=0.3, exact=False)
            time.sleep(4.0)
            click((960, 540), after=0.3)
            cursor_visible(False)
            key_down("w")
            time.sleep(2.4)
            key_up("w")
            look(300, 0, dur=1.4, start=(760, 540))
        prep(t, "Rehearse: leave")
        tap("Escape", after=0.6)
        cursor_visible(True)

    # ------------------------------------------------------------ Open ground
    flat = [n for n in ("Landing zones", "Relief camp", "Site logistics", "Rules", "Impact",
                        "Split", "Share") if want(n)]
    if not flat:
        return
    prep(t, "Open ground")
    cdp("Page.navigate", url=f"{APP}/projects/{FLAT}")
    time.sleep(2.0)
    install_cursor()
    wait_ready()
    set_orbit(*FLAT_WIDE)
    time.sleep(0.6)

    if want("Landing zones"):
        prep(t, "Landing zones: a mission")
        mode("Mission")
        name_it('input[aria-label="Mission name"]', "Op Meadow", "Create mission")
        mini(t, "Landing zones")
        with t.step("Find flat, clear circles"):
            tab("Landing zones", after=0.5)
            pick("Find landing zones", after=0.3, exact=False)
            time.sleep(2.5)
            pick("Add to plan", after=1.4, exact=False)
            drag_orbit(0.25, 0.0, 0.95, dur=2.4)

    if any(want(n) for n in ("Relief camp", "Site logistics", "Rules", "Impact", "Split", "Share")):
        prep(t, "Plan: a scheme")
        mode("Plan")
        name_it('input[aria-label="Scheme name"]', "Relief site", "Create scheme")
        pick("Proposed", after=0.4)

    def place(item, at):
        # "What to place" is a <select>; the Object tool stays armed between drops,
        # and clicking it again would disarm it.
        tab("Draw", after=0.3)
        if not js("[...document.querySelectorAll('aside.inspector select')].some(s => s.closest('label, .field')?.innerText.trim().startsWith('What to place'))"):
            click({"sel": "aside.inspector .plan-tool", "text": "Object"}, after=0.4)
        choose("What to place", item)
        time.sleep(0.3)
        click(world(*at), after=0.8)

    if want("Relief camp"):
        mini(t, "Relief camp")
        with t.step("Tents, a medical tent, a helipad"):
            cam_glide([10, 0, 2], 46, 0.3, 0.62, dur=1.2)
            for item, at in (("relief tent", (4, 0, 0)), ("relief tent", (10, 0, 0)), ("medical tent", (16, 0, 0)),
                             ("water bladder", (4, 0, 8)), ("helipad", (18, 0, 12))):
                place(item, at)
            drag_orbit(0.3, 0.0, 0.95, dur=2.4)

    if want("Site logistics"):
        mini(t, "Site logistics")
        with t.step("A tower crane and its jib circle"):
            cam_glide([26, 0, -18], 150, 0.3, 1.05, dur=1.2)
            place("tower crane", (28, 0, -18))
            place("site office", (20, 0, -28))
            time.sleep(0.8)
            drag_orbit(0.3, 0.0, 0.95, dur=2.6)

    if want("Rules") or want("Impact"):
        prep(t, "Rules: a plot and a building")
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
    if want("Rules"):
        mini(t, "Rules")
        with t.step("Raise it past the plot's 12 m limit"):
            for _ in range(4):
                click({"sel": 'aside.inspector button[aria-label="One floor more"]'}, after=0.45)
            tab("Rules", after=1.6)
            drag_orbit(0.2, 0.0, 0.95, dur=2.0)
    if want("Impact"):
        mini(t, "Impact")
        with t.step("What the scheme changes"):
            tab("Impact", after=3.0)
    if want("Split"):
        mini(t, "Split")
        with t.step("Existing and proposed side by side"):
            pick("Split", after=2.4)
            drag_orbit(0.25, 0.0, 0.95, dur=2.4)
        prep(t, "Split off")
        pick("Proposed", after=0.6)
    if want("Share"):
        mini(t, "Share")
        with t.step("GIS and CAD exports"):
            tab("Share", after=0.6)
            for fmt in ("GeoJSON", "DXF", "3D Tiles", "CityJSON 2.0"):
                pick(fmt, after=0.45, exact=False)
            pick("Download CityJSON", after=1.6, exact=False)
