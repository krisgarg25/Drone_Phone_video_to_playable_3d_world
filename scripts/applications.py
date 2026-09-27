"""Mission profiles: one registry entry per SIH26158 application (playbook §4).

An application selects existing knobs and nothing else: which pipeline lanes, which dense
profile, which capture presets suit it, which official products it needs, which analyses
and workspace tabs it uses, and what its acceptance demo checks. It is recorded in
``scenario.json`` (pipeline) and the survey run record with who set it, so a finished run
can say what it was built for.

``suggest(pattern)`` turns the GPS flight pattern (``pipeline.gps_flight_pattern``: orbit /
grid / corridor) into ranked suggestions - a suggestion, never a silent choice.
"""

OFFICIAL = ("obj", "ply", "las", "geotiff", "glb/gltf", "fbx")

APPLICATIONS = {
    "military": {
        "label": "Military reconnaissance and mission planning",
        "lanes": ["survey:P-R", "survey:P-S", "pipeline:P-V"], "dense_profile": "budget",
        "preset_candidates": ["corridor", "drone", "drone_mapping"],
        "required_products": ["las", "geotiff", "glb/gltf", "ply"],
        "analyses": ["MGRS coordinates", "line of sight", "viewsheds", "route exposure", "HLZ", "threat heatmap",
                     "obstacles", "trafficability", "detections", "mission pack"],
        "workspace_tabs": ["layers", "measure", "mission", "ops", "walk"],
        "acceptance_checks": ["time to first target coordinate", "CE90 vs checkpoints", "planted person detected",
                              "route exposure computed", "rehearsal + AAR"],
        "not_needed": ["CityJSON", "shadow study"]},
    "urban": {
        "label": "Urban planning and smart cities",
        "lanes": ["survey:P-S", "pipeline:P-V"], "dense_profile": "fast",
        "preset_candidates": ["outdoor_building", "drone"],
        "required_products": list(OFFICIAL),
        "analyses": ["building inventory", "facade completeness", "roads", "buildings", "rule checks", "shadow study",
                     "before/after", "CityJSON / 3D Tiles"],
        "workspace_tabs": ["layers", "measure", "plan", "twin", "walk"],
        "acceptance_checks": ["5 building heights within 0.5 m", "road impact table", "rule violation fixed by edit",
                              "swipe before/after", "completeness per class"],
        "not_needed": ["mission rehearsal"]},
    "disaster": {
        "label": "Disaster damage assessment",
        "lanes": ["survey:P-R --progressive", "survey:P-S"], "dense_profile": "budget",
        "preset_candidates": ["drone_mapping", "drone"],
        "required_products": ["geotiff", "las", "ply"],
        "analyses": ["first map", "change", "damage grade", "debris volume", "detections", "blocked roads",
                     "vehicle route", "flood", "field pack"],
        "workspace_tabs": ["layers", "measure", "ops", "plan"],
        "acceptance_checks": ["time to first map", "debris volume within 10%", "people absent from mesh, present as points"],
        "not_needed": ["FBX", "CityJSON"]},
    "construction": {
        "label": "Construction progress monitoring",
        "lanes": ["survey:P-T"], "dense_profile": "fast",
        "preset_candidates": ["drone_mapping", "drone"],
        "required_products": ["las", "geotiff", "ply", "obj"],
        "analyses": ["stockpile volume", "cut/fill vs design", "epoch change", "zone progress", "design deviation",
                     "crane clearance", "report"],
        "workspace_tabs": ["layers", "measure", "ops", "plan", "twin"],
        "acceptance_checks": ["change above LoD found, unchanged ground clean", "pile volume within 5%"],
        "not_needed": ["mission rehearsal"]},
    "border": {
        "label": "Border and strategic area mapping",
        "lanes": ["survey:P-T corridor", "survey:P-R"], "dense_profile": "budget",
        "preset_candidates": ["corridor", "drone_mapping"],
        "required_products": ["geotiff", "las"],
        "analyses": ["straight-track alignment", "tiles", "change with MGRS", "profile", "blind spots",
                     "surveillance posts", "KLV ingest"],
        "workspace_tabs": ["layers", "measure", "ops", "mission"],
        "acceptance_checks": [">= 1 km corridor aligned by M11", "CE90 vs checkpoints", "planted change found",
                              "blind-spot map from two posts"],
        "not_needed": ["FBX", "CityJSON"]},
    "inspection": {
        "label": "Infrastructure inspection",
        "lanes": ["survey:P-D", "survey:P-S"], "dense_profile": "survey",
        "preset_candidates": ["outdoor_building", "corridor", "drone"],
        "required_products": ["obj", "ply", "las", "glb/gltf"],
        "analyses": ["frames that saw a point", "defect register", "crack candidates", "tilt", "wire sag + clearance",
                     "M3C2 deformation", "inspection report"],
        "workspace_tabs": ["layers", "measure", "inspect", "plan", "twin"],
        "acceptance_checks": ["5 tape lengths within 2%", "defect traced to its frames", "report generated"],
        "not_needed": ["mission rehearsal", "flood"]},
    "archaeology": {
        "label": "Archaeological documentation",
        "lanes": ["survey:P-D", "survey:P-T", "pipeline:P-V"], "dense_profile": "survey",
        "preset_candidates": ["outdoor_building", "drone"],
        "required_products": list(OFFICIAL),
        "analyses": ["textured mesh", "ortho", "hillshade", "local relief model", "sections", "annotations",
                     "observed/unobserved layers", "hypothesis reconstruction", "archive record", "seasonal M3C2"],
        "workspace_tabs": ["layers", "measure", "inspect", "plan", "twin", "walk"],
        "acceptance_checks": ["textured mesh, hillshade, a section, unobserved regions marked", "hold-out texture check"],
        "not_needed": ["mission rehearsal", "vehicle routing"]},
    "twin": {
        "label": "Digital twin generation",
        "lanes": ["survey:P-S", "pipeline:P-V"], "dense_profile": "fast",
        "preset_candidates": ["outdoor_building", "drone", "drone_mapping"],
        "required_products": list(OFFICIAL),
        "analyses": ["visual/measured/evidence views", "asset inventory + attributes", "epochs", "3D Tiles LOD",
                     "engine package", "proposals"],
        "workspace_tabs": ["layers", "measure", "twin", "plan", "place", "walk"],
        "acceptance_checks": ["splat and mesh in one frame", "FBX/glTF/3D Tiles open in an engine"],
        "not_needed": []},
}

PATTERN_SUGGESTIONS = {
    "orbit": ["urban", "inspection", "twin", "archaeology"],
    "grid": ["construction", "disaster", "archaeology"],
    "corridor": ["border", "military", "inspection"],
}


def get(app_id):
    if app_id not in APPLICATIONS:
        raise ValueError(f"unknown application {app_id!r}; one of {', '.join(APPLICATIONS)}")
    return {"id": app_id, **APPLICATIONS[app_id]}


def listing():
    return [{"id": k, "label": v["label"], "workspace_tabs": v["workspace_tabs"]} for k, v in APPLICATIONS.items()]


def suggest(pattern):
    """Ranked suggestions for a flight pattern, with the reason; [] when unknown."""
    kind = pattern.get("pattern") if isinstance(pattern, dict) else pattern
    return [{"id": a, "label": APPLICATIONS[a]["label"],
             "because": f"a {kind} flight is how {APPLICATIONS[a]['label'].lower()} is usually captured"}
            for a in PATTERN_SUGGESTIONS.get(kind, [])]


def preset_for(app_id, detected_preset):
    """Keep the detected preset when the application accepts it; else the application's first choice.

    Returns (preset, reason). The capture probe measured the footage; the application only
    breaks a tie it could not, or overrides a preset that application never uses.
    """
    app = get(app_id)
    if detected_preset in app["preset_candidates"]:
        return detected_preset, f"application:{app_id} accepts the detected preset {detected_preset}"
    return app["preset_candidates"][0], (f"application:{app_id} uses {', '.join(app['preset_candidates'])}; "
                                         f"the detected {detected_preset} is not one of them")


def ledger_for(ledger, app_id):
    """Mark official formats the application did not ask for as not requested (never missing)."""
    if not app_id:
        return ledger
    wanted = set(get(app_id)["required_products"])
    rows = []
    for row in ledger["rows"]:
        row = dict(row)
        row["requested"] = row["format"] in wanted
        if not row["requested"] and row["status"] != "delivered":
            row["status"] = "not_requested"
            row["reason"] = f"not requested by the {app_id} profile"
        rows.append(row)
    return {**ledger, "rows": rows, "application": app_id,
            "requested": sorted(wanted), "delivered_of_requested": sum(1 for r in rows if r["requested"] and r["status"] == "delivered")}
