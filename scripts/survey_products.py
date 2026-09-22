"""Georeferenced evidence products: a cloud plus per-point support reporting.

Combines multi-view support (survey_evidence) with the ENU transform
(survey_georef) so a reviewer can see which parts of the model are well
observed, not only how the scene looks. Support is internal consistency, not
measured accuracy against surveyed truth.
"""
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

try:  # imported as scripts.survey_products by the tests
    from scripts import survey_evidence as evidence
    from scripts import survey_georef as georef
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_evidence as evidence
    import survey_georef as georef


def read_evidence(path):
    """Read back an evidence PLY (XYZ/RGB/support/error/confidence)."""
    return evidence.read_ply(path)


def evidence_for_sparse(points3d_path, alignment, output_dir, *,
                        min_views=3, max_error_px=1.0):
    """Write an ENU evidence cloud and its support summary; returns both paths."""
    model = evidence.parse_points3d(points3d_path)
    confidence = evidence.support_confidence(model, min_views=min_views, max_error_px=max_error_px)
    enu = georef.transform_points(model.xyz, alignment)
    frame = alignment["coordinate_frame"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud = output_dir / "evidence_points.ply"
    evidence.export_evidence_ply(replace(model, xyz=enu), confidence, cloud)
    summary = evidence.support_summary(replace(model, xyz=enu), min_views=min_views, cell_size_m=1.0)
    summary.update(coordinate_frame=frame, source="colmap-points3d",
                   reprojection_error_unit="px", min_views=min_views,
                   max_error_px=max_error_px,
                   interpretation="support counts observing views; it is not surveyed accuracy")
    report = output_dir / "evidence_summary.json"
    temporary = report.with_name(report.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, report)
    finally:
        temporary.unlink(missing_ok=True)
    return cloud, report
