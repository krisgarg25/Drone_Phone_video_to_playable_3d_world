"""What the run can say about itself without being handed surveyed truth.

Three questions the brief asks, and the honest CPU-only answer each one gets:

* **how complete is the model?** ``survey_observability`` measures which regions a
  single flight path could triangulate at all - the geometric ceiling on coverage.
  That is not the same as measured completeness, which needs an independent surface
  reference, and the payload says so in ``not_measured``.
* **will this finish in time?** ``survey_streaming`` multiplies per-image rates that
  were actually measured on this machine by the frame budget this run used, and
  reports the result as a prediction with its provenance attached.
* **what control does this scene actually need?** ``survey_gnss`` plus
  ``survey_accuracy.gcp_requirement`` turn the telemetry's own quality figures into
  a statement about identifiability: what GPS alone can and cannot pin down.

Everything here runs at reconstruction time, never on a dashboard poll: the cloud
subsample and the SVDs are minutes of work, not microseconds.
"""
import numpy as np

try:  # imported as scripts.survey_assess by the tests
    from scripts import survey_accuracy as accuracy
    from scripts import survey_gnss as gnss
    from scripts import survey_observability as observability
    from scripts import survey_occlusion as occlusion
    from scripts import survey_streaming as streaming
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_accuracy as accuracy
    import survey_gnss as gnss
    import survey_observability as observability
    import survey_occlusion as occlusion
    import survey_streaming as streaming

SUBSAMPLE_POINTS = 20_000
"""Region observability is a spatial statistic: 20k points resolves a metre grid over
a hectare, and the difference between 20k and 2M points is noise at that cell size."""
DEFAULT_CELL_SIZE_M = 2.0
MAX_CANDIDATE_FRAMES = 900


def _sample(points, limit=SUBSAMPLE_POINTS, seed=0):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points must be a non-empty finite Nx3 array")
    if not np.isfinite(points).all():
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError("no finite points remain to assess")
    if len(points) <= limit:
        return points, False
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=limit, replace=False)], True


def completeness(camera_centers_enu, points_enu, *, cell_size_m=DEFAULT_CELL_SIZE_M,
                 support=None, visible_support=None, min_views=3, min_visible=2,
                 max_points=SUBSAMPLE_POINTS):
    """Region observability of the measured cloud, plus layer counts when they exist.

    ``support``/``visible_support`` come from the evidence products: claimed view
    count and the subset that survives the depth-map occlusion check. With them the
    cloud is sorted into measured / weak / unobserved; without them only the
    observability half is reported, and the payload says which half is missing.
    """
    centers = np.asarray(camera_centers_enu, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 3:
        raise ValueError("camera_centers_enu needs at least three cameras as Nx3 ENU metres")
    sampled, decimated = _sample(points_enu, max_points)
    centroid = np.median(sampled, axis=0)
    summary = observability.summarise(centers, centers, sampled, cell_size_m=cell_size_m,
                                      target=centroid)
    result = {"schema_version": 1, "cell_size_m": float(cell_size_m),
              "point_count_assessed": int(len(sampled)),
              "point_count_total": int(len(np.atleast_2d(points_enu))),
              "decimated": bool(decimated),
              "trajectory": summary["trajectory"], "regions": summary["regions"],
              "baselines": summary["baselines"], "thresholds": summary["thresholds"],
              "limits": summary["limits"], "not_measured": summary["not_measured"],
              "layers": None,
              "interpretation": ("region observability: what one flight path could "
                                 "triangulate. It is the geometric ceiling on coverage, "
                                 "not measured completeness against a reference surface")}
    if support is not None and visible_support is not None:
        classified = occlusion.classify(sampled, support=support, visible_support=visible_support,
                                        min_views=min_views, min_visible=min_visible)
        result["layers"] = {"point_count": classified["point_count"],
                            "counts": classified["counts"], "fractions": classified["fractions"],
                            "support_basis": classified["support_basis"],
                            "visibility_basis": classified["visibility_basis"],
                            "note": classified["note"]}
    return result


def throughput(*, frame_count, image_px, video_duration_s, rates=None, windows=4,
               frame_options=None, resolution_options=None):
    """Predicted timing for the budget this run actually used, never a measurement."""
    rates = {name: dict(spec) for name, spec in (rates or streaming.MEASURED_RATES).items()}
    frame_count = max(1, int(frame_count))
    image_px = max(1, int(image_px))
    report = streaming.report(rates, frame_count, image_px,
                              video_duration_s=float(video_duration_s), windows=windows,
                              frame_options=frame_options,
                              resolution_options=resolution_options or (image_px,))
    report["frame_count_used"] = frame_count
    report["image_px_used"] = image_px
    return report


def control_requirement(*, telemetry, camera_centers_enu, samples, max_speed_m_s,
                        clock_bounds=None):
    """What this scene's GPS alone can pin down, and what it cannot.

    Inputs are the reports ``survey_gnss`` already produced at preparation time, so
    the answer is bound to the telemetry that built the model rather than to a
    generic statement about drones.
    """
    quality = gnss.quality_report(samples, max_speed_m_s=max_speed_m_s)
    observed = gnss.observability(camera_centers_enu, samples)
    vertical = gnss.vertical_reference_check(samples, {"altitude_datum": "ellipsoidal"})
    fix_quality = gnss.fix_quality_weights(["unknown"] * len(samples),
                                           allow_unknown_fallback=True)
    bounds = clock_bounds or gnss.time_offset_bounds(
        [0.0, float(telemetry["video_duration_s"])] if telemetry.get("video_duration_s")
        else [0.0, 1.0], [sample["t_sec"] for sample in samples], max_speed_m_s=max_speed_m_s)
    return {"schema_version": 1, "telemetry_quality": quality, "observability": observed,
            "vertical_reference": vertical, "fix_quality": fix_quality, "clock_bounds": bounds,
            "requirement": accuracy.gcp_requirement(telemetry_quality=quality,
                                                    trajectory_observability=observed,
                                                    fix_quality=fix_quality,
                                                    vertical_reference=vertical,
                                                    clock_bounds=bounds)}


def accuracy_from_checkpoints(rows, *, crs, vertical_datum="ellipsoidal",
                              withheld_from_reconstruction=False, fit_fraction=0.6, seed=0):
    """Split surveyed checkpoints into fit and hold-out sets, then measure the hold-out.

    Returns ``None`` when the scene has too few checkpoints to split: the caller
    then reports the plain paired error and says the split was impossible, rather
    than quietly presenting a fitted residual as an independent one.
    """
    if len(rows) < 8:
        return None
    plain = [{"id": str(row["id"]), "surveyed": [float(v) for v in row["surveyed"]],
              "model": [float(v) for v in row["model"]]} for row in rows]
    return accuracy.report(rows=plain, crs=crs, vertical_datum=vertical_datum,
                           fit_fraction=fit_fraction, seed=seed,
                           withheld_from_reconstruction=withheld_from_reconstruction)
