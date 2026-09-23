"""Dynamic-object evidence: what moved, as opposed to what the camera did.

Challenge (iv) of the brief is that vehicles, people and animals ride along into a
static reconstruction and inflate it into a ghost cloud. This module does not name
anything - it has no segmentation network, no downloaded weights and no GPU, and it
never imports torch. What it does have is motion evidence, which is the honest
signal available offline:

* ``background_model`` + ``motion_mask`` - where does this frame disagree with a
  per-pixel median of its neighbours? That alone cannot tell a moving car from a
  moving drone, so on a flying camera it lights up whole frames.
* ``epipolar_inconsistency`` - does the observed pixel correspondence agree with the
  camera's own epipolar geometry? A panning camera produces parallax that is exactly
  consistent with its essential matrix; an obstacle sliding across the scene does
  not. This is the part that separates "camera moved" from "object moved".
* ``dynamic_masks_for_sequence`` combines them, so appearance evidence is only kept
  where the geometry cannot explain it.

Two deliberate refusals. Stationary obstacles (a parked vehicle, a tree, a scaffold)
are *retained* - see ``static_obstacle_note`` - because the brief asks for them, and
``apply_to_weights`` discounts observations instead of deleting them, because a
wrong mask must cost weight and not geometry. Where the geometry cannot be trusted
(low parallax, too few correspondences) the result says ``'unreliable'`` and hands
back an empty mask rather than a confident guess.

Conventions match the rest of the survey scripts: frames are uint8 (H, W) or
(H, W, 3) BGR (float frames must be normalised to 0..1), ``camera_centers`` are
ordered metric camera positions (2 or 3 columns; only their differences are used),
``K`` is a 3x3 pinhole matrix, and every mask is a boolean array in the pixel grid it
describes. OpenCV is used for CPU image processing only - no cv2.cuda, no UMat.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

# Per-pixel median over a bounded window. The stack and np.partition's copy are the
# only two large buffers, so peak memory is 2 * window * H * W * channels bytes:
# 22 MiB at 1280x720 with the window below. Windows are forced ODD so the median is
# an actual observed pixel value and needs no second (wider) buffer for averaging.
MAX_BACKGROUND_WINDOW = 7

# A median is only a static background if the window is short enough that no object
# parks inside it for half the frames, and long enough to survive exposure noise.
DEFAULT_MIN_AREA_PX = 8
DEFAULT_AUTO_FLOOR = 6.0        # 0..255 units; below this Otsu is splitting noise.
DEFAULT_TOLERANCE_PX = 2.0      # epipolar/flow tolerance in pixels.
DEFAULT_LK_LEVELS = 3           # pyramid depth used when no motion budget is derivable.
PYRAMID_LEVEL_CAP = 5           # never pyramid past this, whatever the budget says.
VETO_RADIUS_STEPS = 8           # veto support as a multiple of the feature spacing.
MOVING_RADIUS_PX = 4            # disk drawn around an epipolar-inconsistent track.
PARALLAX_BAND_SLACK = 0.5       # extra width on the geometric parallax band.
PARALLAX_BAND_WIDTH = 1.0       # the band is this multiple of the prior's parallax span.
MIN_PARALLAX_POPULATION = 8     # least correspondences needed to centre the band.
FAST_CORNER_THRESHOLD = 12      # absolute FAST contrast, in 0..255 grey levels.
# How much of a tracked window has to still look like itself before the correspondence is
# allowed to accuse a pixel. Measured on the crossing-obstacle scenes: correspondences the
# pyramid placed on their true surface score 0.92-1.00 and ones it dropped 12-37 px away -
# inside the obstacle's capture radius, which is also where the accusations were false -
# score 0.14-0.84. Raising this above ~0.9 starts deleting the obstacle's own tracks, which
# is the wrong direction: a missed mover puts a ghost into the model, a dropped track does
# not.
MATCH_CORRELATION_FLOOR = 0.85

_OPTION_NAMES = ("method", "window", "threshold", "min_area_px", "morph_open",
                 "morph_close", "grow_px", "auto_floor", "max_area_fraction",
                 "tolerance_px", "min_tracks", "min_inlier_ratio", "min_parallax_px",
                 "scene_depth_range", "max_corners", "quality_level", "min_distance",
                 "moving_radius_px", "veto_radius_px", "min_explained_ratio",
                 "fb_tolerance_px", "lk_window_px", "lk_levels", "rng_seed")


@dataclass(frozen=True)
class BackgroundModel:
    """A coarse static background plus the arithmetic that made it reproducible."""

    image: np.ndarray
    method: str
    window: int
    frames_used: int
    peak_bytes: int

    def __array__(self, dtype=None, copy=None):
        array = self.image
        return array.astype(dtype, copy=True) if dtype is not None else array


# --------------------------------------------------------------------------- helpers


def _positive_int(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from exc
    if isinstance(value, float) and value != number:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if number < 1:
        raise ValueError(f"{name} must be a positive whole number, got {number}")
    return number


def _non_negative_int(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from exc
    if isinstance(value, float) and value != number or number < 0:
        raise ValueError(f"{name} must be a whole number >= 0, got {value!r}")
    return number


def _finite_float(value, name, *, low=0.0, high=math.inf, exclusive_low=True):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < low or (exclusive_low and number == low) or number > high:
        bounds = f">{low:g}" if exclusive_low else f">={low:g}"
        raise ValueError(f"{name} must be {bounds} and <= {high:g}, got {number:g}")
    return number


def _as_image(value, name):
    """Validate and normalise an image to contiguous uint8 with a channel axis."""
    if isinstance(value, BackgroundModel):
        value = value.image
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise ValueError(f"{name} must be an image array, got {type(value).__name__}")
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must be a numeric image array, not ragged or object data")
    if array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        pass
    else:
        raise ValueError(f"{name} must be a 2-D grey image or a 3-D image with 1 or 3 "
                         f"channels, got shape {array.shape}")
    if (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.complexfloating)
            or not np.issubdtype(array.dtype, np.number)):
        raise ValueError(f"{name} must hold an integer image dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite - NaN or inf pixels cannot be compared")
    if array.dtype == np.uint8:
        out = array
    elif np.issubdtype(array.dtype, np.integer):
        if array.min() < 0 or array.max() > 255:
            raise ValueError(f"{name} with dtype {array.dtype} must stay inside 0..255")
        out = array.astype(np.uint8)
    else:
        if array.min() < 0.0 or array.max() > 1.0:
            raise ValueError(f"{name} float images must be normalised to 0..1 "
                             "(uint8 frames in 0..255 are also accepted)")
        out = np.rint(array * 255.0).astype(np.uint8)
    if 0 in out.shape[:2]:
        raise ValueError(f"{name} must not be empty, got shape {out.shape}")
    return np.ascontiguousarray(out)


def _as_gray(value, name):
    image = _as_image(value, name)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def _as_frame_list(frames, name="frames"):
    """Validate an ordered sequence of same-shaped frames, without copying them twice."""
    if isinstance(frames, np.ndarray):
        if frames.ndim not in (3, 4):
            raise ValueError(f"{name} must be a sequence of 2-D or 3-D frames, "
                             f"got an array of shape {frames.shape}")
        source = [frames[index] for index in range(len(frames))]
    elif isinstance(frames, (list, tuple)):
        source = list(frames)
    else:
        raise ValueError(f"{name} must be a sequence of frames, got {type(frames).__name__}")
    if not source:
        raise ValueError(f"{name}: no frames given")
    images = [_as_image(frame, f"{name}[{index}]") for index, frame in enumerate(source)]
    shape = images[0].shape
    for index, image in enumerate(images):
        if image.shape != shape:
            raise ValueError(f"{name} must all have the same shape; frame {index} is "
                             f"{image.shape}, not {shape}")
    return images


def _as_intrinsics(matrix):
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"K must be a 3x3 camera matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("K must be finite")
    if array[0, 0] <= 0 or array[1, 1] <= 0:
        raise ValueError("K must carry positive focal lengths")
    return array


def _as_centers(value, name="camera_centers"):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array of camera positions") from exc
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{name} must be an ordered (frames, 2 or 3 columns) array of "
                         f"camera positions, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_mask(value, name="mask"):
    if isinstance(value, dict):
        if "mask" not in value:
            raise ValueError(f"{name} dicts must carry a 'mask' key")
        value = value["mask"]
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array, got dtype {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D boolean array, got shape {array.shape}")
    return array


def _odd_window(value, available, name="window"):
    window = _positive_int(value, name)
    window = min(window, max(1, int(available)))
    return window if window % 2 else max(1, window - 1)


def _median_uint8(stack):
    """Per-pixel median of a uint8 stack without promoting anything to float64."""
    count = len(stack)
    if count == 1:
        return stack[0].copy()
    middle = count // 2
    partitioned = np.partition(stack, middle, axis=0)  # one copy, same dtype
    if count % 2:
        return partitioned[middle].copy()
    low, high = partitioned[middle - 1], partitioned[middle]
    return (low + ((high.astype(np.uint16) - low.astype(np.uint16) + 1) // 2)).astype(np.uint8)


def _disk(shape, points, radius):
    """Boolean coverage of the frame by radius-px disks centred on integer points."""
    mask = np.zeros(shape, np.uint8)
    for column, row in points:
        cv2.circle(mask, (int(column), int(row)), int(radius), 1, -1, cv2.LINE_8)
    return mask.astype(bool)


def _nearest_class_grid(shape, points, classes, *, subsample=4):
    """Voronoi class and squared distance per pixel, computed on a coarse grid.

    ``points`` are (col, row) float positions and ``classes`` int8 labels; callers pass
    them in priority order so that ties go to the first class listed.
    """
    height, width = shape
    grid_y, grid_x = np.meshgrid(np.arange(0.5, height, subsample),
                                 np.arange(0.5, width, subsample), indexing="ij")
    cells = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
    best_class = np.zeros(len(cells), np.int8)
    best_distance = np.full(len(cells), np.inf)
    for start in range(0, len(points), 64):
        chunk = np.asarray(points[start:start + 64], dtype=np.float64)
        labels = np.asarray(classes[start:start + 64], dtype=np.int8)
        delta = cells[:, None, :] - chunk[None, :, :]
        squared = np.einsum("cij,cij->ci", delta, delta)
        winner = np.argmin(squared, axis=1)
        distance = squared[np.arange(len(cells)), winner]
        better = distance < best_distance
        best_distance[better] = distance[better]
        best_class[better] = labels[winner[better]]
    class_grid = best_class.reshape(grid_x.shape)
    distance_grid = best_distance.reshape(grid_x.shape)
    resize = (width, height)
    class_full = cv2.resize(class_grid.astype(np.int16), resize, interpolation=cv2.INTER_NEAREST)
    distance_full = cv2.resize(distance_grid.astype(np.float32), resize,
                              interpolation=cv2.INTER_NEAREST)
    return class_full.astype(np.int8), distance_full.astype(np.float64)


# --------------------------------------------------------------- background + motion


def background_model(frames, *, method="median", window=None):
    """Coarse static background of a short ordered sequence: the per-pixel median.

    ``window`` frames are used, capped at ``MAX_BACKGROUND_WINDOW`` (7) and forced odd;
    for a longer sequence the window is centred on the sequence. Peak memory is
    ``2 * frames_used * H * W * channels`` bytes (the stack plus np.partition's copy) -
    22 MiB for a 1280x720 colour window at the cap - which is why the window is bounded
    rather than "use every keyframe".

    The median only removes an object that occupies any given pixel in fewer than half
    the frames: a drone that hovers behind a parked van, or a person standing still for
    the whole window, stays in the background and is therefore never masked. That is a
    limitation of the estimator, not a decision to delete geometry.
    """
    if method != "median":
        raise ValueError(f"background_model only estimates a median background, "
                         f"got method={method!r}")
    images = _as_frame_list(frames)
    wanted = MAX_BACKGROUND_WINDOW if window is None else window
    size = _odd_window(wanted, len(images))
    start = max(0, (len(images) - size) // 2)
    stack = np.stack(images[start:start + size])
    return BackgroundModel(_median_uint8(stack), "median", size, len(stack),
                           2 * int(stack.nbytes))


def motion_mask(frame, background, *, threshold=None, min_area_px=DEFAULT_MIN_AREA_PX,
                morph_open=1, morph_close=None, grow_px=0, auto_floor=DEFAULT_AUTO_FLOOR,
                max_area_fraction=0.6):
    """Boolean mask of pixels that moved, plus the threshold that produced it.

    The difference is taken per pixel across channels, then thresholded with Otsu
    (``cv2.THRESH_OTSU``) rather than a magic constant, cleaned by an elliptical
    open/close and filtered by connected-component area. Otsu's own value is clamped up
    to ``auto_floor``: on a static frame the difference histogram is one noise peak, and
    a method designed to split two peaks will happily split the noise and mask the lot.

    The chosen threshold, the raw pixel count and the shape of every knob are returned
    so a run can be reproduced exactly.
    """
    image = _as_image(frame, "frame")
    model = _as_image(background, "background")
    if image.shape != model.shape:
        raise ValueError("frame and background must have the same shape, got "
                         f"{image.shape} and {model.shape}")
    min_area_px = _positive_int(min_area_px, "min_area_px")
    morph_open = _non_negative_int(morph_open, "morph_open")
    morph_close = morph_open if morph_close is None else _non_negative_int(morph_close, "morph_close")
    grow_px = _non_negative_int(grow_px, "grow_px")
    auto_floor = _finite_float(auto_floor, "auto_floor", high=255.0, exclusive_low=False)
    max_area_fraction = _finite_float(max_area_fraction, "max_area_fraction", high=1.0)
    if threshold is None:
        method = "otsu"
    else:
        method = "fixed"
        _finite_float(threshold, "threshold", high=255.0)

    difference = cv2.absdiff(image, model)
    flattened = difference if difference.ndim == 2 else difference.max(axis=2)
    gray = np.ascontiguousarray(flattened)
    if method == "otsu":
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        chosen = max(float(otsu), auto_floor)
    else:
        chosen = float(threshold)
    binary = (gray > chosen).astype(np.uint8)
    raw_pixels = int(binary.sum())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_open + 1,) * 2)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) if morph_open else binary
    if morph_close:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_close + 1,) * 2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    components = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    count, _, stats = components[0], components[1], components[2]
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    mask = keep[components[1]]
    if grow_px:
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2)
        mask = cv2.dilate(mask.astype(np.uint8), grow_kernel).astype(bool)
    mask = np.ascontiguousarray(mask.reshape(image.shape[:2]).astype(bool))

    warnings = []
    fraction = float(mask.mean())
    if fraction > max_area_fraction:
        warnings.append(f"motion-mask-covers-{fraction:.2f}-of-the-frame: a moving camera, "
                        "an exposure jump or a bad background window looks like this; do not "
                        "trust it without the epipolar check")
    return {"mask": mask, "threshold": chosen, "threshold_method": method,
            "auto_floor": auto_floor,
            "otsu_threshold": None if method == "fixed" else float(otsu),
            "raw_pixels": raw_pixels, "pixels": int(mask.sum()), "area_fraction": fraction,
            "min_area_px": min_area_px, "morph_open": morph_open, "morph_close": morph_close,
            "grow_px": grow_px, "shape": tuple(image.shape[:2]),
            "difference_median": float(np.median(gray)), "difference_p99": float(np.percentile(gray, 99)),
            "warnings": warnings}


# ------------------------------------------------------------------ epipolar geometry


def epipolar_inconsistency(prev_gray, cur_gray, camera_centers, K, *, tolerance_px,
                           min_tracks=24, min_inlier_ratio=0.25, min_parallax_px=None,
                           scene_depth_range=None, max_corners=1200, quality_level=0.01,
                           min_distance=5, moving_radius_px=MOVING_RADIUS_PX,
                           veto_radius_px=None, min_explained_ratio=0.4,
                           fb_tolerance_px=1.5, lk_window_px=None, lk_levels=None,
                           rng_seed=0):
    """Which correspondences cannot be explained by the camera's own motion.

    Sparse features are tracked between the two frames (Shi-Tomasi + pyramidal
    Lucas-Kanade), an essential matrix is fitted robustly (``cv2.findEssentialMat``,
    five-point + RANSAC, pose by ``cv2.recoverPose``), and each track is first judged on
    the Sampson distance to the fundamental matrix ``F = K^-T E K^-1`` in pixels. That is
    the cross-epipolar-line test: exact for any scene depth, and blind to motion along
    the epipolar line.

    An accusation is only as good as the correspondence underneath it, so a track has to
    survive two gates before it can accuse anything: the pyramid is no deeper than the
    motion budget the camera geometry allows (see ``_pyramid_depth``), and the two windows
    have to still look like the same surface (see ``_correspondences``). A fast obstacle
    sitting inside a tracker's capture radius otherwise drafts the background behind it
    into its own motion, and the residual test then reports a static wall as a moving van.
    ``lk_levels`` overrides the depth rule for callers that know better.

    The blind direction is exactly where a drone flies, so a second test uses the pose and
    the metric baseline from ``camera_centers``: a static point must land on the segment
    between where the camera motion predicts it at the near and the far end of
    ``scene_depth_range``, so its displacement along that segment implies a depth. Faster
    than the near end means it moved; slower than the far end is left unjudged; a
    correspondence off the segment is inconsistent. Without a depth prior only the exact
    cross-line test runs, no pixel is vetoed as static, and the result says so in
    ``reasons`` rather than pretending the residual test sees everything.

    Returns a dict with ``status`` ('ok' or 'unreliable'), ``usable``, ``reasons``,
    ``mask`` (inconsistent pixels, in the current frame's grid), ``static_veto`` (pixels
    whose nearest correspondence is proven static parallax), and the measurements behind
    both: track counts, inlier ratio, residuals, rotation angle, measured parallax, the
    parallax spread the geometry actually has, and the implied depth per track. An
    'unreliable' result carries empty masks: no veto and no accusation.
    """
    prev = _as_gray(prev_gray, "prev_gray")
    cur = _as_gray(cur_gray, "cur_gray")
    if prev.shape != cur.shape:
        raise ValueError("prev_gray and cur_gray must have the same shape, got "
                         f"{prev.shape} and {cur.shape}")
    height, width = prev.shape
    if min(height, width) < 24:
        raise ValueError(f"frames are too small to judge epipolar geometry: {prev.shape} "
                         "needs at least 24 px on both axes")
    matrix = _as_intrinsics(K)
    centers = _as_centers(camera_centers)
    if centers.shape[0] != 2:
        raise ValueError("camera_centers must hold exactly the two camera centers for this "
                         f"frame pair, got {centers.shape[0]}")
    tolerance = _finite_float(tolerance_px, "tolerance_px")
    depth_prior = None if scene_depth_range is None else _depth_prior(scene_depth_range)
    tracks_min = _positive_int(min_tracks, "min_tracks")
    if tracks_min < 8:
        raise ValueError(f"min_tracks must be at least 8 for a five-point fit, got {tracks_min}")
    ratio_min = _finite_float(min_inlier_ratio, "min_inlier_ratio", high=1.0)
    explained_min = _finite_float(min_explained_ratio, "min_explained_ratio", high=1.0)
    _positive_int(max_corners, "max_corners")
    _finite_float(quality_level, "quality_level", high=1.0)
    _positive_int(min_distance, "min_distance")
    moving_radius = _non_negative_int(moving_radius_px, "moving_radius_px")
    veto_radius = (VETO_RADIUS_STEPS * min_distance if veto_radius_px is None
                   else _non_negative_int(veto_radius_px, "veto_radius_px"))
    fb_tolerance = _finite_float(fb_tolerance_px, "fb_tolerance_px")
    if lk_window_px is None:
        # Trackers want a window that scales with the image and with the inter-frame motion,
        # so it is derived here and capped. Too small and it sees no structure; too large and
        # it averages the moving object's edge with the static background behind it, which
        # produces correspondences that slide along that edge and accuse the whole frame.
        window_px = int(min(21, max(5, round(min(height, width) / 16.0) | 1)))
    else:
        window_px = _positive_int(lk_window_px, "lk_window_px")
    wanted_levels = None if lk_levels is None else _non_negative_int(lk_levels, "lk_levels")
    _non_negative_int(rng_seed, "rng_seed")
    parallax_floor = (max(1.0, 1.5 * tolerance) if min_parallax_px is None
                      else _finite_float(min_parallax_px, "min_parallax_px"))

    empty = {"status": "unreliable", "usable": False, "mask": np.zeros((height, width), bool),
             "static_veto": np.zeros((height, width), bool)}
    result = {**empty, "reasons": [], "tolerance_px": tolerance, "shape": (height, width),
              "essential": None, "fundamental": None, "rotation": None,
              "translation_direction": None, "rotation_angle_rad": None, "rotation_px": None,
              "parallax_px": None, "parallax_spread_px": None, "parallax_band_px": None,
              "parallax_bias_px": None,
              "baseline_units": None, "veto_radius_px": float(veto_radius),
              "moving_radius_px": float(moving_radius),
              "tracks": {"count": 0, "inlier_count": 0, "inlier_ratio": 0.0,
                         "inconsistent_count": 0, "unknown_count": 0,
                         "explained_ratio": 0.0, "ransac_inlier_ratio": 0.0,
                         "median_residual_px": None,
                         "residual_px": np.zeros(0), "displacement_px": np.zeros(0),
                         "along_epipolar_px": None, "pixels": np.zeros((0, 2), int),
                         "inconsistent_pixels": np.zeros((0, 2), int)}}

    baseline = float(np.linalg.norm(centers[1] - centers[0]))
    result["baseline_units"] = baseline
    if baseline <= 0.0:
        result["reasons"].append("epipolar-unreliable: no-camera-baseline, a rotating or "
                                 "hovering camera produces no parallax, so epipolar "
                                 "consistency proves nothing")
        return result

    focal = float(np.hypot(matrix[0, 0], matrix[1, 1]) / math.sqrt(2.0))
    if wanted_levels is not None:
        levels = wanted_levels
    elif depth_prior is None:
        # No depth prior means no bound on how far a static point may travel, so there is
        # no motion budget to size the pyramid by and the historical depth is kept.
        levels = DEFAULT_LK_LEVELS
        result["reasons"].append("epipolar note: no depth prior, so the tracker's pyramid "
                                 f"depth stays at {levels}; without a bound on static "
                                 "parallax the capture range cannot be sized to the scene")
    else:
        # The farthest displacement this module is willing to explain as camera motion is
        # the near-end parallax plus the geometric parallax span plus the band's own
        # allowance; that is exactly how far the tracker has to be able to see, and every
        # level beyond it adds only capture radius for the failure described above.
        near, far = depth_prior
        spread = focal * baseline * (1.0 / near - 1.0 / far)
        budget = (focal * baseline / near + spread
                  + max(tolerance, PARALLAX_BAND_SLACK * spread))
        levels = _pyramid_depth(window_px, budget)

    first, second = _correspondences(prev, cur, max_corners=int(max_corners),
                                     quality_level=float(quality_level),
                                     min_distance=int(min_distance),
                                     fb_tolerance_px=fb_tolerance,
                                     window_px=window_px, levels=levels)
    if len(second):
        inside = ((second[:, 0] >= 0) & (second[:, 0] <= width - 1)
                  & (second[:, 1] >= 0) & (second[:, 1] <= height - 1))
        first, second = first[inside], second[inside]
    result["tracks"]["count"] = int(len(first))
    if len(first) < tracks_min:
        result["reasons"].append(f"epipolar-unreliable: only {len(first)} usable "
                                 "correspondences between the frames, too few to judge "
                                 "the camera's own geometry")
        return result

    cv2.setRNGSeed(int(rng_seed))  # RANSAC draws from a global RNG; seed it for reproducibility
    # findEssentialMat applies its threshold in the normalised image plane, not in pixels
    # (measured: a "2 px" value lets an 800 px error pass as an inlier, so the fit and the
    # pose decomposition that follows both drift). Convert the pixel tolerance to that plane.
    try:
        essential, inlier_mask = cv2.findEssentialMat(first, second, matrix, cv2.RANSAC,
                                                     0.995, tolerance / focal)
    except cv2.error as exc:
        result["reasons"].append(f"epipolar-unreliable: essential matrix fit failed ({exc})")
        return result
    if essential is None or not np.isfinite(essential).all() or np.abs(essential).max() <= 1e-12:
        result["reasons"].append("epipolar-unreliable: degenerate essential matrix")
        return result
    essential = np.asarray(essential, dtype=np.float64)
    inliers = np.asarray(inlier_mask).ravel().astype(bool) if inlier_mask is not None \
        else np.ones(len(first), bool)
    result["essential"] = essential
    result["tracks"]["ransac_inlier_count"] = int(inliers.sum())
    result["tracks"]["ransac_inlier_ratio"] = float(inliers.mean())

    inverse = np.linalg.inv(matrix)
    fundamental = inverse.T @ essential @ inverse
    result["fundamental"] = fundamental
    residual = _sampson_distance(first, second, fundamental)
    # Fit support is measured here, in pixels, rather than taken from RANSAC's own flag:
    # OpenCV scores the essential matrix with its algebraic error, which is several times the
    # geometric distance and would quietly reject most of a perfectly good background.
    consistent = residual <= tolerance
    support = float(consistent.mean())
    result["tracks"]["inlier_count"] = int(consistent.sum())
    result["tracks"]["inlier_ratio"] = support
    if support < ratio_min:
        result["reasons"].append(f"epipolar-unreliable: only {support:.2f} of the "
                                 f"correspondences agree with a single camera motion to within "
                                 f"{tolerance:.2f} px, so the fit itself cannot be trusted")
        return result
    displacement = second - first
    speed = np.linalg.norm(displacement, axis=1)
    result["tracks"]["residual_px"] = residual
    result["tracks"]["displacement_px"] = speed
    result["tracks"]["median_residual_px"] = float(np.median(residual))
    result["tracks"]["pixels"] = np.rint(second).astype(int)
    result["parallax_px"] = float(np.median(speed[consistent])) if consistent.any() else 0.0

    angle, rotation_px, _pose = _recover_pose(essential, first, second, matrix, inliers, result)
    result["rotation_angle_rad"] = angle
    result["rotation_px"] = rotation_px

    if result["parallax_px"] < parallax_floor:
        result["reasons"].append(f"epipolar-unreliable: low parallax, the static scene moved "
                                 f"{result['parallax_px']:.2f} px between frames, below the "
                                 f"{parallax_floor:.2f} px needed to separate camera motion "
                                 "from object motion")
        return result

    # Travel *along* the epipolar line, which is the direction the epipolar residual is
    # blind to. The line direction comes from F, so this needs no camera pose at all - and
    # deliberately so: on a drone's short inter-frame baseline the four decompositions of E
    # are separated by less than a pixel of reprojection, so a pose-derived prediction of
    # parallax is a number computed from the very thing being checked.
    # parallax is a number computed from the very thing being checked. Only its magnitude is
    # used: the sign of an epipolar line's coefficients is a matter of convention, and two
    # points either side of the epipole travel in opposite directions along one line.
    along = np.abs(_along_epipolar(first, second, fundamental))
    band = None
    if scene_depth_range is not None:
        near, far = depth_prior
        # The whole span of parallax this baseline can produce across the plausible scene.
        spread = focal * baseline * (1.0 / near - 1.0 / far)
        result["parallax_spread_px"] = float(spread)
        reference = along[consistent & np.isfinite(along)]
        if spread < parallax_floor:
            result["reasons"].append(f"epipolar-unreliable: no measurable parallax, the "
                                     f"{baseline:.5f} unit camera baseline can move a static "
                                     f"point at most {spread:.2f} px along its epipolar line "
                                     "across the stated depth prior, which is too little to "
                                     "tell camera motion from object motion")
            return result
        if len(reference) < MIN_PARALLAX_POPULATION:
            result["reasons"].append(f"epipolar-unreliable: only {len(reference)} "
                                     "correspondences agree with the fitted camera motion, "
                                     "too few to locate the static population's own parallax, "
                                     "so nothing is vetoed")
            return result
        # Centre the band on the static population's typical travel, which absorbs whatever
        # the camera's own rotation contributes, and take its width from the geometry: the
        # parallax the depth prior and the metric baseline allow, plus an allowance.
        centre = float(np.median(reference))
        half = (PARALLAX_BAND_WIDTH * spread
                + max(tolerance, PARALLAX_BAND_SLACK * spread))
        band = (0.0, centre + half)
        result["parallax_band_px"] = band
        result["parallax_bias_px"] = centre
        result["reasons"].append(
            f"along-epipolar parallax-band test active: the {near:g}..{far:g} depth prior and "
            f"the {baseline:.5f} unit baseline span {spread:.2f} px of parallax, so travel "
            f"past {band[1]:.2f} px along the epipolar line (the static median {centre:.2f} px "
            "widened by the geometric span and its allowance) is motion, not camera motion")
    else:
        result["reasons"].append("epipolar note: no depth prior given, so along-epipolar "
                                 "motion cannot be bounded by parallax; only the "
                                 "cross-epipolar residual is judged and nothing is vetoed")

    if band is None:
        broken = ~consistent
        inconsistent = broken
        static = np.zeros(len(first), bool)
        unknown = ~inconsistent
    else:
        strayed = along > band[1]
        inconsistent = ~consistent | strayed
        static = consistent & ~strayed
        unknown = ~(inconsistent | static)
    result["tracks"]["along_epipolar_px"] = along
    result["tracks"]["inconsistent_count"] = int(inconsistent.sum())
    result["tracks"]["unknown_count"] = int(unknown.sum())
    result["tracks"]["inconsistent_pixels"] = np.rint(second[inconsistent]).astype(int)
    result["mask"] = _disk((height, width), second[inconsistent], moving_radius)

    # Only tracks with an opinion seed the veto grid: 'unknown' means the geometry had
    # nothing to say about that point, so it must neither grant nor block a veto.
    order = np.concatenate([np.flatnonzero(inconsistent), np.flatnonzero(static)])
    labels = np.concatenate([np.full(inconsistent.sum(), 2, np.int8),
                             np.full(static.sum(), 1, np.int8)])
    if static.any():
        classes, distance = _nearest_class_grid((height, width), second[order], labels)
        veto = (classes == 1) & (distance <= float(veto_radius) ** 2)
        result["static_veto"] = np.ascontiguousarray(veto)
    if inconsistent.any():
        result["reasons"].append("epipolar: some correspondences are inconsistent with the "
                                 "camera's own motion and are not vetoed as parallax")
    if unknown.any():
        result["reasons"].append("epipolar: correspondences slower than the parallax band are "
                                 "left unvetoed rather than called static")
    explained = float(static.mean())
    result["tracks"]["explained_ratio"] = explained
    if band is not None and explained < min_explained_ratio:
        result["reasons"].append(f"epipolar-unreliable: the camera's own motion explains only "
                                 f"{explained:.2f} of the correspondences, so this frame pair "
                                 "is not trusted to veto anything")
        return {**result, "status": "unreliable", "usable": False,
                "mask": np.zeros((height, width), bool),
                "static_veto": np.zeros((height, width), bool)}
    result["status"] = "ok"
    result["usable"] = True
    return result


def _correspondences(prev, cur, *, max_corners, quality_level, min_distance,
                     fb_tolerance_px, window_px, levels):
    """Feature pairs tracked between two frames, and why this detector was chosen.

    Shi-Tomasi's ``qualityLevel`` is relative to the strongest corner in the frame, so a
    single sunlit van or bright tarpaulin can suppress every background corner and leave the
    geometry test with nothing to say. FAST's threshold is absolute, so it cannot be
    outshouted; its corners are then thinned to the requested spacing and tracked with
    pyramidal Lucas-Kanade, kept only if the round trip prev -> cur -> prev returns to the
    same pixel. Without that check a few pixels of drift per track is normal, and drift is
    indistinguishable from motion in the residual test, so it invents moving objects.

    The round trip is necessary and not sufficient: it proves the two positions are
    *mutually reachable*, and a correspondence that the pyramid dumped into a neighbouring
    structure satisfies that just as neatly as a correct one. Measured on the crossing
    obstacle fixture, correspondences 20-30 px away from their true match closed the round
    trip to 0.03-1.5 px, because going there and coming back both fall in the same wrong
    basin. So each surviving pair is also checked for being the *same surface*, by the
    zero-mean correlation of the two windows the tracker itself used. A pair below
    ``MATCH_CORRELATION_FLOOR`` is not evidence about anything: the geometry test cannot
    tell a 20 px tracking lie from a moving object, and it is the accusation, not the
    silence, that costs static geometry. A flat window scores 0 there, which is correct - a
    patch with no structure carries no evidence to check.
    """
    detector = cv2.FastFeatureDetector_create(threshold=max(4, FAST_CORNER_THRESHOLD),
                                             nonmaxSuppression=True)
    found = detector.detect(prev, None)
    points = [(key.pt[0], key.pt[1]) for key in found]
    responses = [max(float(key.response), 1.0) for key in found]
    if len(points) < min_distance:
        corners = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners,
                                         qualityLevel=quality_level,
                                         minDistance=min_distance, blockSize=5)
        if corners is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        points = [tuple(corner) for corner in corners.reshape(-1, 2)]
        responses = [1.0] * len(points)
    kept = _thin_by_distance(points, responses, min_distance, max_corners)
    if not len(kept):
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    forward, status, _err = cv2.calcOpticalFlowPyrLK(
        prev, cur, kept.reshape(-1, 1, 2).astype(np.float32), None,
        winSize=(window_px, window_px), maxLevel=levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01))
    if forward is None or status is None:
        return kept[:0], kept[:0]
    alive = status.reshape(-1) == 1
    tracked = forward.reshape(-1, 2)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        cur, prev, tracked[alive].reshape(-1, 1, 2), None,
        winSize=(window_px, window_px), maxLevel=levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01))
    if backward is None:
        return kept[alive], tracked[alive]
    returned = backward.reshape(-1, 2)
    good = alive.copy()
    good[alive] = ((back_status.reshape(-1) == 1)
                   & (np.linalg.norm(returned - kept[alive], axis=1) <= fb_tolerance_px))
    if not good.any():
        return kept[:0], kept[:0]
    start, end = kept[good], tracked[good]
    same_surface = (_patch_correlation(prev, cur, start, end, window_px // 2)
                    >= MATCH_CORRELATION_FLOOR)
    return start[same_surface], end[same_surface]


def _patch_correlation(first_image, second_image, first_points, second_points, radius):
    """Zero-mean normalised cross-correlation of coincident windows in two frames.

    Windows are the tracker's own size, clamped inside the frame (a point whose window does
    not fit is a borderline case that loses a little of its score rather than being allowed
    to accuse a pixel). The result is in -1..1: 1 is the same patch, 0 is a patch with no
    structure or no relationship, and it is scale invariant, so it measures correspondence
    and not brightness - which matters because an obstacle keeps its appearance while a
    mis-tracked background point loses it.
    """
    size = 2 * int(radius) + 1
    count = len(first_points)
    if min(first_image.shape) < size or min(second_image.shape) < size or not count:
        return np.zeros(count, np.float32)

    def windows(image, points):
        rows, columns = image.shape
        top = np.clip(np.rint(points[:, 1]).astype(int) - radius, 0, rows - size)
        left = np.clip(np.rint(points[:, 0]).astype(int) - radius, 0, columns - size)
        view = np.lib.stride_tricks.sliding_window_view(image, (size, size))
        return view[top, left].astype(np.float32)

    a = windows(first_image, np.asarray(first_points, dtype=np.float64))
    b = windows(second_image, np.asarray(second_points, dtype=np.float64))
    a -= a.mean(axis=(1, 2), keepdims=True)
    b -= b.mean(axis=(1, 2), keepdims=True)
    numerator = np.einsum("ijk,ijk->i", a, b)
    denominator = (np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(b, axis=(1, 2)))
    return np.divide(numerator, denominator, out=np.zeros(count, np.float32),
                     where=denominator > 1e-6).astype(np.float32)


def _thin_by_distance(points, responses, min_distance, max_corners):
    """Greedy spatial thinning, strongest corner first, on a coarse position grid."""
    cell = max(1, int(min_distance))
    radius_squared = float(min_distance) ** 2
    occupied = {}
    kept = []
    order = sorted(range(len(points)), key=lambda index: -responses[index])
    for index in order:
        x, y = points[index]
        gx, gy = int(x // cell), int(y // cell)
        far = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_x, other_y in occupied.get((gx + dx, gy + dy), ()):
                    if (other_x - x) ** 2 + (other_y - y) ** 2 < radius_squared:
                        far = False
                        break
                if not far:
                    break
            if not far:
                break
        if not far:
            continue
        occupied.setdefault((gx, gy), []).append((x, y))
        kept.append((x, y))
        if len(kept) >= max_corners:
            break
    return np.asarray(kept, np.float32).reshape(-1, 2)


def _along_epipolar(first, second, fundamental):
    """Signed travel of each correspondence along its own epipolar line, in pixels.

    The epipolar line of ``first`` in the second image is ``F p1``; its direction is the
    line's normal rotated by 90 degrees. Projecting the measured displacement onto that
    direction is exactly the component the epipolar residual cannot see, which is the
    component a translating drone lives on.
    """
    lines = np.column_stack([first, np.ones(len(first))]) @ fundamental.T
    norm = np.hypot(lines[:, 0], lines[:, 1])
    usable = norm > 1e-12
    direction = np.divide(np.column_stack([-lines[:, 1], lines[:, 0]]), norm[:, None],
                          out=np.zeros((len(first), 2)), where=usable[:, None])
    along = np.einsum("ij,ij->i", second - first, direction)
    return np.where(usable, along, np.nan)


def _recover_pose(essential, first, second, matrix, inliers, result):
    """Relative rotation, unit translation direction, and the largest shift either implies."""
    try:
        pose_mask = np.asarray(_inliers_as_mask(inliers, len(first)))
        _, rotation, translation, _chirality = cv2.recoverPose(essential, first, second,
                                                              matrix, mask=pose_mask)
    except cv2.error as exc:
        result["reasons"].append(f"epipolar note: pose recovery failed ({exc}); only the "
                                 "epipolar residual is measured")
        return None, None, None
    if rotation is None or translation is None:
        return None, None, None
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(translation))
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or norm <= 0.0:
        return None, None, None
    result["rotation"] = rotation
    result["translation_direction"] = translation / norm
    angle = float(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    height, width = result["shape"]
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    corner_radius = max(math.hypot(dx, dy) for dx in (cx, width - cx) for dy in (cy, height - cy))
    return angle, angle * corner_radius, (rotation, translation / norm)


def _pyramid_depth(window_px, motion_budget_px, cap=PYRAMID_LEVEL_CAP):
    """Smallest Lucas-Kanade pyramid depth that can still reach the expected motion.

    Depth buys capture range and costs accuracy. ``calcOpticalFlowPyrLK`` searches a
    ``window_px`` square at every level, so the coarsest level alone can carry a track by
    about (window_px / 2) * 2**levels pixels, and a level whose own pixel is comparable to
    the real displacement resolves it as a coin toss. The levels below it can then only
    crawl +-window_px/2 from that coin toss, so a track that was never lost by the scene is
    lost by the tracker: it settles on the strongest structure inside its new reach, which
    in a survey frame is whatever silhouette is nearby (a vehicle, a person, a fence rail),
    and it reports that silhouette's motion as its own.

    That is not caught by the forward-backward check, because such a pair is a mutual
    attractor: going there and coming back both land inside the same wrong basin, so the
    round trip closes to well under a pixel while the correspondence is 20-30 px off - and
    8 px of that error across the epipolar lines is four times the tolerance, i.e. an
    accusation that a static pixel is a moving object. So the depth is the shallowest one
    that can still measure the motion budget the caller derived from the geometry, and
    never one level more.
    """
    half = max(1.0, window_px / 2.0)
    levels = 0
    while half * (2.0 ** levels) < motion_budget_px and levels < int(cap):
        levels += 1
    return levels


def _sampson_distance(first, second, fundamental):
    """Symmetric epipolar residual in pixels for paired (N, 2) correspondence sets."""
    x1 = np.column_stack([first, np.ones(len(first))])
    x2 = np.column_stack([second, np.ones(len(second))])
    line_in_2 = x1 @ fundamental.T
    line_in_1 = x2 @ fundamental
    numerator = np.einsum("ij,ij->i", x2, line_in_2)
    denominator = (line_in_2[:, 0] ** 2 + line_in_2[:, 1] ** 2
                   + line_in_1[:, 0] ** 2 + line_in_1[:, 1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.abs(numerator) / np.sqrt(denominator)
    return np.where(denominator > 1e-12, distance, np.inf)



def _inliers_as_mask(inliers, count):
    mask = np.zeros((count, 1), np.uint8)
    mask.ravel()[inliers] = 1
    return mask


def _depth_prior(value):
    try:
        near, far = value
    except (TypeError, ValueError) as exc:
        raise ValueError("scene_depth_range must be a (near, far) pair of positive depths "
                         "in the same units as camera_centers") from exc
    near = _finite_float(near, "scene_depth_range near depth")
    far = _finite_float(far, "scene_depth_range far depth")
    if far <= near:
        raise ValueError(f"scene_depth_range must have far > near, got {near} and {far}")
    return near, far


# ------------------------------------------------------------------ the combined pass


def dynamic_masks_for_sequence(frames, camera_centers, K, **opts):
    """Per-frame dynamic masks, with the confidence and reasons that earned them.

    For each frame the appearance evidence is the difference from a local median
    background; the geometric evidence is ``epipolar_inconsistency`` against the
    preceding frame. Appearance is kept only where the geometry cannot explain it as
    parallax, and geometrically inconsistent tracks are added on top:

        mask = (appearance & ~static_veto) | epipolar_inconsistent

    So a panning drone produces an almost empty mask (its parallax is vetoed) while a
    crossing vehicle produces a tight one. Where the geometry is unreliable - hovering,
    no texture, too few correspondences - the mask falls back to appearance alone with
    a lower confidence and a reason naming the missing evidence.

    Options (all keyword, all validated): the ``background_model``, ``motion_mask`` and
    ``epipolar_inconsistency`` arguments above. Unrecognised names raise rather than
    being silently ignored, because a misspelled knob here changes what gets deleted.
    """
    images = _as_frame_list(frames)
    centers = _as_centers(camera_centers)
    if len(centers) != len(images):
        raise ValueError(f"camera_centers must supply one camera center per frame: "
                         f"{len(centers)} centers for {len(images)} frames")
    matrix = _as_intrinsics(K)
    unknown = sorted(set(opts) - set(_OPTION_NAMES))
    if unknown:
        raise ValueError(f"unknown option(s) for dynamic_masks_for_sequence: "
                         f"{', '.join(unknown)}. Accepted: {', '.join(_OPTION_NAMES)}")
    shared = dict(opts)
    tolerance = shared.pop("tolerance_px", DEFAULT_TOLERANCE_PX)
    _finite_float(tolerance, "tolerance_px")
    window = shared.pop("window", MAX_BACKGROUND_WINDOW)
    _positive_int(window, "window")
    mask_options = {key: value for key, value in shared.items()
                    if key in ("threshold", "min_area_px", "morph_open", "morph_close",
                               "grow_px", "auto_floor", "max_area_fraction")}
    geometry_options = {key: value for key, value in shared.items()
                        if key in ("min_tracks", "min_inlier_ratio", "min_parallax_px",
                                   "scene_depth_range", "max_corners", "quality_level",
                                   "min_distance", "moving_radius_px", "veto_radius_px",
                                   "min_explained_ratio", "fb_tolerance_px",
                                   "lk_window_px", "lk_levels", "rng_seed")}
    method = opts.get("method", "median")
    if method != "median":
        raise ValueError(f"dynamic_masks_for_sequence only estimates a median background, "
                         f"got method={method!r}")

    size = _odd_window(window, len(images))
    half = size // 2
    gray = [_as_gray(image, "frame") for image in images]
    results = []
    for index, image in enumerate(images):
        low = max(0, min(index - half, len(images) - size))
        window_frames = images[low:low + size]
        background = background_model(window_frames, method="median", window=size)
        appearance = motion_mask(image, background, **mask_options)
        entry = {"index": index, "status": "appearance-only", "epipolar_status": "absent",
                 "confidence": 0.45, "reasons": [], "epipolar": None,
                 "background_threshold": appearance["threshold"], "vetoed_fraction": 0.0,
                 "mask": appearance["mask"], "area_fraction": appearance["area_fraction"]}
        entry["reasons"].extend(appearance["warnings"])
        if index == 0:
            entry["reasons"].append("no-preceding-frame: appearance evidence only")
            results.append(entry)
            continue
        geometry = epipolar_inconsistency(gray[index - 1], gray[index],
                                          centers[index - 1:index + 1], matrix,
                                          tolerance_px=tolerance, **geometry_options)
        entry["epipolar"] = {key: value for key, value in geometry.items()
                             if key not in ("mask", "static_veto", "essential", "fundamental",
                                            "rotation", "tracks")}
        entry["epipolar"]["tracks"] = {key: value for key, value in geometry["tracks"].items()
                                       if key in ("count", "inlier_count", "inlier_ratio",
                                                  "inconsistent_count", "unknown_count",
                                                  "median_residual_px")}
        entry["epipolar_status"] = geometry["status"]
        if not geometry["usable"]:
            entry["reasons"].extend(geometry["reasons"])
            entry["reasons"].append("appearance-only-mask: the geometry could not tell "
                                    "parallax from motion, so this mask is not corroborated")
            results.append(entry)
            continue
        veto = geometry["static_veto"]
        moving = geometry["mask"]
        appearance_pixels = int(entry["mask"].sum())
        vetoed = np.logical_and(entry["mask"], veto)
        entry["mask"] = np.ascontiguousarray((entry["mask"] & ~veto) | moving)
        entry["area_fraction"] = float(entry["mask"].mean())
        entry["vetoed_fraction"] = (float(vetoed.sum() / appearance_pixels)
                                    if appearance_pixels else 0.0)
        entry["status"] = "ok"
        entry["confidence"] = 0.85
        if entry["vetoed_fraction"] >= 0.5:
            entry["reasons"].append("camera-motion-explained")
            entry["reasons"].append(f"{entry['vetoed_fraction']:.2f} of the appearance "
                                    "difference was accounted for as parallax by the camera's "
                                    "own epipolar geometry")
        if not entry["mask"].any():
            entry["reasons"].append("no-motion-evidence: neither appearance nor geometry "
                                    "found anything moving")
        else:
            overlap = float(np.logical_and(entry["mask"], appearance["mask"]).sum()
                            / entry["mask"].sum())
            if overlap < 0.5:
                entry["confidence"] -= 0.25
                entry["reasons"].append("epipolar-motion-without-appearance-change: geometry "
                                        "flagged pixels the median background did not")
            else:
                entry["confidence"] += 0.05
                entry["reasons"].append("epipolar-and-appearance-agree")
        if geometry["tracks"]["inlier_ratio"] < 0.6:
            entry["confidence"] -= 0.1
            entry["reasons"].append("sparse-epipolar-inliers")
        if entry["area_fraction"] > 0.6:
            entry["confidence"] *= 0.5
            entry["reasons"].append(f"mask-covers-{entry['area_fraction']:.2f}-of-the-frame: "
                                    "discounting this much area is a completeness cost, and a "
                                    "fitted camera motion this poor is the usual cause")
        entry["confidence"] = float(min(max(entry["confidence"], 0.0), 1.0))
        results.append(entry)
    return results


# ------------------------------------------------------------- effects on the model


def apply_to_weights(masks, weights, *, pixels=None, factor=0.25, floor=0.05):
    """Down-weight observations inside dynamic regions. Nothing is deleted.

    ``weights`` is either one value per mask (1-D) or one row of per-observation values
    per mask (2-D). With 1-D weights the discount is proportional to the discounted
    area of that frame - the honest aggregate when the observation-to-pixel mapping is
    not known - so a frame whose mask covers 40 % of the image keeps
    ``1 - 0.6 * (1 - factor)`` of its weight. With 2-D weights, pass ``pixels`` (one
    (row, col) per observation, shared or per view) to discount exactly the
    observations that project inside the mask, or size the row to the flattened mask
    grid. Every result is clamped to ``floor * original``, so a wrong mask costs weight
    and never geometry.
    """
    mask_list = _mask_list(masks)
    if not mask_list:
        raise ValueError("apply_to_weights: no masks given")
    factor = _finite_float(factor, "factor", high=1.0)
    floor = _finite_float(floor, "floor", low=0.0, exclusive_low=False, high=1.0)
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc
    if values.dtype == object or not np.isfinite(values).all():
        raise ValueError("weights must be finite numbers")
    if np.any(values < 0):
        raise ValueError("weights must be non-negative; this function discounts, it does not sign-flip")
    if pixels is not None:
        grid = np.asarray(pixels)
        if grid.dtype == object or not np.issubdtype(grid.dtype, np.integer):
            raise ValueError("pixels must be an integer array of (row, col) positions")
        if grid.ndim == 2 and grid.shape[1] == 2:
            grid = np.repeat(grid[None, :, :], len(mask_list), axis=0)
        if grid.ndim != 3 or grid.shape[0] != len(mask_list) or grid.shape[2] != 2:
            raise ValueError("pixels must be (observations, 2) or (masks, observations, 2) "
                             f"of integer row/column positions, got shape {grid.shape}")
        if values.ndim != 2 or values.shape[0] != len(mask_list) or values.shape[1] != grid.shape[1]:
            raise ValueError("with pixels, weights must be (masks, observations) matching "
                             f"{len(mask_list)} masks and {grid.shape[1]} observations, "
                             f"got shape {values.shape}")
        out = values.copy()
        for index, mask in enumerate(mask_list):
            rows, columns = grid[index][:, 0], grid[index][:, 1]
            if rows.size and (rows.min() < 0 or columns.min() < 0
                              or rows.max() >= mask.shape[0] or columns.max() >= mask.shape[1]):
                raise ValueError("pixels fall outside the mask frame, so the discount could "
                                 "not be applied honestly")
            inside = mask[rows, columns] if len(rows) else np.zeros(0, bool)
            multiplier = np.where(inside, factor, 1.0)
            out[index] = np.maximum(values[index] * multiplier, values[index] * floor)
        return out

    if values.ndim == 1:
        if len(values) != len(mask_list):
            raise ValueError("weights must hold one weight per mask (1-D) or one row per "
                             f"mask (2-D): got {len(values)} weights for {len(mask_list)} masks")
        fractions = np.array([float(mask.mean()) for mask in mask_list])
        multiplier = np.maximum(1.0 - (1.0 - factor) * fractions, floor)
        return np.maximum(values * multiplier, values * floor)
    if values.ndim == 2 and values.shape[0] == len(mask_list):
        out = values.copy()
        for index, mask in enumerate(mask_list):
            flat = mask.shape[0] * mask.shape[1]
            if values.shape[1] != flat:
                raise ValueError(f"weights row {index} has {values.shape[1]} observations but "
                                 f"mask {index} covers {flat} pixels: pass pixels=(observations, 2) "
                                 "to say where each observation lands")
            multiplier = np.where(mask.reshape(-1), factor, 1.0)
            out[index] = np.maximum(values[index] * multiplier, values[index] * floor)
        return out
    raise ValueError("weights must hold one weight per mask (1-D) or one row per mask (2-D), "
                     f"got shape {values.shape} for {len(mask_list)} masks")


def _mask_list(masks):
    if isinstance(masks, dict):
        masks = [masks]
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        masks = [masks]
    if not isinstance(masks, (list, tuple)):
        masks = list(masks) if hasattr(masks, "__iter__") else [masks]
    return [_as_mask(value, "mask") for value in masks]


def coverage_cost(masks, shape, *, budget=0.25):
    """How much image area was discounted - the completeness price of the masks.

    Throwing away half of every frame is not a free safety measure: it removes real
    coverage as well as moving junk, so the fraction is reported here, per frame and
    overall, and flagged when it exceeds ``budget``.
    """
    mask_list = _mask_list(masks)
    if not mask_list:
        raise ValueError("coverage_cost: no masks given")
    budget = _finite_float(budget, "budget", high=1.0)
    if shape is None:
        shape = mask_list[0].shape
    shape = tuple(int(value) for value in shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be a (height, width) pair, got {shape}")
    per_frame = []
    for index, mask in enumerate(mask_list):
        if mask.shape != shape:
            raise ValueError(f"mask {index} has shape {mask.shape}, not the reported shape {shape}")
        per_frame.append(int(mask.sum()))
    total = len(mask_list) * shape[0] * shape[1]
    discounted = int(sum(per_frame))
    fraction = discounted / total
    frames_with_area = [value / (shape[0] * shape[1]) for value in per_frame]
    return {"frames": len(mask_list), "shape": shape, "total_area_px": int(total),
            "discounted_area_px": discounted,
            "discounted_fraction": round(float(fraction), 6),
            "mean_frame_fraction": round(float(np.mean(frames_with_area)), 6),
            "max_frame_fraction": round(float(np.max(frames_with_area)), 6),
            "per_frame_area_px": per_frame, "budget": budget,
            "over_budget": bool(fraction > budget),
            "statement": (f"{fraction * 100:.1f}% of the image area across {len(mask_list)} "
                          f"frames was discounted as dynamic; that is a completeness cost, not "
                          f"a free win, because every one of those pixels may also have carried "
                          f"static geometry. "
                          + (f"This exceeds the {budget * 100:.0f}% budget and should be reviewed "
                             "before training." if fraction > budget else
                             "This stays inside the completeness budget."))}


def static_obstacle_note(masks, shape):
    """Say out loud what this module refuses to remove.

    Motion evidence cannot see a parked vehicle, a tree or a scaffold, and this module
    does not try: they are static, they are legitimate obstacles the brief asks to be
    reconstructed, so they keep full weight by design.
    """
    cost = coverage_cost(masks, shape)
    return {
        "statement": ("Only moving pixels were discounted here. A parked vehicle, standing "
                      "vegetation, scaffolding or any other stationary obstacle keeps full "
                      "weight by design, because those are exactly the obstacles the brief "
                      "asks the model to reconstruct; nothing is deleted at all, and "
                      "observations inside the masks are down-weighted instead."),
        "stationary_content_retained": True,
        "retained_by_design": ["parked vehicles", "static vegetation", "structures and scaffolds",
                               "unmoving people and animals"],
        "retained_but_suspect": ["a person or animal standing still for the whole background "
                                 "window is indistinguishable from the static scene here"],
        "discounted_fraction": cost["discounted_fraction"],
        "discounted_area_px": cost["discounted_area_px"],
        "frames": cost["frames"],
        "deleted_pixels": 0,
        "needs_a_model_for": ["class labels - a SAM2-class segmenter would name vehicles, humans "
                              "and animals and could remove an obstacle that never moves",
                              "amodal completion of the geometry behind a masked region",
                              "telling an unmoving person apart from a statue"],
    }
