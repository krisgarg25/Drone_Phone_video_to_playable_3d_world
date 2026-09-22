"""CPU-only COLMAP intrinsics conversion for OpenCV undistortion."""

import numpy as np


_PARAMETER_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "FULL_OPENCV": 12,
}


def _finite_vector(value, name, length):
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex values are not supported")
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric vector") from exc
    if vector.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values in a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def camera_matrix_and_distortion(model, params, source_size, image_size):
    """Return float64 K and OpenCV distortion for the actual image pixel grid.

    source_size is the calibrated COLMAP (width, height); image_size is the
    decoded image (width, height). Distortion is dimensionless and unscaled.
    Unsupported models and malformed calibration raise ValueError.
    """
    if not isinstance(model, str) or model not in _PARAMETER_COUNTS:
        raise ValueError(f"Unsupported camera model {model!r}; supported: {', '.join(_PARAMETER_COUNTS)}")
    params = _finite_vector(params, f"{model} params", _PARAMETER_COUNTS[model])
    source_size = _finite_vector(source_size, "source_size", 2)
    image_size = _finite_vector(image_size, "image_size", 2)
    for name, size in (("source_size", source_size), ("image_size", image_size)):
        if np.any(size <= 0) or np.any(size != np.floor(size)):
            raise ValueError(f"{name} must contain positive integer pixel dimensions (width, height)")

    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx, fy, cx, cy = params[:4]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{model} focal lengths must be positive")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            sx, sy = image_size / source_size
            K[0, :] *= sx
            K[1, :] *= sy
    except FloatingPointError as exc:
        raise ValueError("Scaled camera matrix K is not representable in float64") from exc
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Scaled camera matrix K must be finite with positive focal lengths")

    dist = np.zeros(4, dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        dist[0] = params[3]
    elif model == "RADIAL":
        dist[:2] = params[3:5]
    elif model in ("OPENCV", "FULL_OPENCV"):
        # COLMAP's rational model matches OpenCV's k1,k2,p1,p2,k3,k4,k5,k6 order.
        dist = params[4:].copy()
    return K, dist
