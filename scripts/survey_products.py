"""Georeferenced evidence products: a cloud plus per-point support reporting.

Combines multi-view support (survey_evidence) with the ENU transform
(survey_georef) so a reviewer can see which parts of the model are well
observed, not only how the scene looks. Support is internal consistency, not
measured accuracy against surveyed truth.

COLMAP track length proves a feature was matched in N views, not that those
views could see the surface. When the caller supplies dense stereo depth maps,
the product also records the occlusion-checked support (survey_visibility) next
to the claimed one, and says plainly when it has not been checked.
"""
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

try:  # imported as scripts.survey_products by the tests
    from scripts import survey_evidence as evidence
    from scripts import survey_georef as georef
    from scripts import survey_visibility as visibility
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_evidence as evidence
    import survey_georef as georef
    import survey_visibility as visibility


def read_evidence(path):
    """Read back an evidence PLY (XYZ/RGB/support/error/confidence/visible_support)."""
    return evidence.read_ply(path)


def _source_frames(views):
    """Record which source each view came from; unnamed views get a positional id."""
    names = []
    for index, view in enumerate(views):
        name = view.get("name") if isinstance(view, dict) else getattr(view, "name", None)
        names.append(str(name) if name else f"view_{index}")
    return names


def evidence_for_sparse(points3d_path, alignment, output_dir, *,
                        min_views=3, max_error_px=1.0, views=None,
                        relative_tolerance=0.02, min_visible_views=2):
    """Write an ENU evidence cloud and its support summary; returns both paths.

    views is an optional sequence of {"K", "viewmat", "depth"} depth-map views.
    When supplied, each point gets a second, measured column (visible_support)
    beside the claimed COLMAP track support, and the JSON gains a visibility
    summary. The check runs on the sparse positions before the georeference, since
    the depth maps are in that frame's metric units. When omitted, the product
    claims no occlusion checking at all.
    """
    checked = views is not None
    if checked and not len(views):
        raise ValueError("views must be a non-empty sequence of depth-map views")
    model = evidence.parse_points3d(points3d_path)
    visible = None
    if checked:
        visible = visibility.visible_support(model.xyz, views,
                                             relative_tolerance=relative_tolerance)
    confidence = evidence.support_confidence(model, min_views=min_views, max_error_px=max_error_px)
    enu = georef.transform_points(model.xyz, alignment)
    frame = alignment["coordinate_frame"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud = output_dir / "evidence_points.ply"
    evidence.export_evidence_ply(replace(model, xyz=enu, visible_support=visible),
                                 confidence, cloud)
    summary = evidence.support_summary(replace(model, xyz=enu), min_views=min_views,
                                       cell_size_m=1.0)
    summary.update(coordinate_frame=frame, source="colmap-points3d",
                   reprojection_error_unit="px", min_views=min_views,
                   max_error_px=max_error_px,
                   support_basis="colmap_track_length",
                   occlusion_checked=bool(checked),
                   visibility_source_frames=_source_frames(views) if checked else [])
    if checked:
        summary.update(
            visibility=visibility.visibility_summary(model.support, visible,
                                                     min_views=min_visible_views),
            visibility_basis="depth_map_ray_consistency",
            relative_tolerance=float(relative_tolerance),
            min_visible_views=int(min_visible_views),
            visibility_note=("occlusion checked against the supplied depth maps: "
                             "visible_support counts the views whose recorded surface "
                             "agrees with the geometry along the viewing ray, while "
                             "support is the COLMAP track length it is compared with"),
            interpretation=("support counts observing views and visible_support counts the "
                            "views whose recorded surface agrees; neither is surveyed accuracy"))
    else:
        summary.update(
            visibility=None,
            visibility_basis=None,
            visibility_note=("no depth-map views supplied, so occlusion was not checked: "
                             "support is COLMAP track length only and does not show that an "
                             "observing camera could see the surface"),
            interpretation=("support counts observing views; occlusion was not checked "
                            "against depth maps, so it is not surveyed accuracy"))
    report = output_dir / "evidence_summary.json"
    temporary = report.with_name(report.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, report)
    finally:
        temporary.unlink(missing_ok=True)
    return cloud, report
