"""
Motion-based insect detection.

Provides GMM background subtraction with shape and cohesiveness
filters to identify insects, plus path topology analysis.

Pixel-scale detection parameters are expressed as FRACTIONS of the input
image dimensions (not absolute pixels). This makes configs portable
across resolutions. At runtime, call ``resolve_detection_params`` with
the actual frame width and height to convert fractions into absolute
pixel values that the detector/tracker internals consume.

Reference dimensions used by ``resolve_detection_params``:
    * Lengths → fraction of image width W
    * Areas   → fraction of image area W * H

``morph_kernel_size`` is an exception: it stays in absolute NxN pixels
(default 3) since it is a local noise-removal kernel, not a scene-scale
quantity.

Dependencies: opencv, numpy (no ML frameworks)
"""

import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: float
    frame_number: int


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Pixel-scale keys (areas and lengths) are stored as FRACTIONS of image
# dimensions here and in YAML configs. They are converted to absolute
# pixels once the frame size is known via ``resolve_detection_params``.
DEFAULT_DETECTION_CONFIG = {
    # GMM Background Subtractor
    "gmm_history": 500,
    "gmm_var_threshold": 16,

    # Morphological filtering — absolute kernel size in pixels (NxN)
    "morph_kernel_size": 3,

    # Cohesiveness
    "min_largest_blob_ratio": 0.80,
    "max_num_blobs": 5,
    "min_motion_ratio": 0.15,

    # Shape — fractions of image area (min/max) and raw ratio (density, solidity)
    "min_area": 0.0002,
    "max_area": 0.035,
    "min_density": 3.0,
    "min_solidity": 0.55,

    # Tracking — fractions of image width (displacement, frame jump)
    "min_displacement": 0.05,
    "min_path_points": 10,
    "max_frame_jump": 0.1,
    "max_lost_frames": 45,
    "max_area_change_ratio": 3.0,

    # Tracker matching
    "tracker_w_dist": 0.6,
    "tracker_w_area": 0.4,
    "tracker_cost_threshold": 0.3,

    # Path topology — fraction of image width (revisit_radius)
    "max_revisit_ratio": 0.30,
    "min_progression_ratio": 0.70,
    "max_directional_variance": 0.90,
    "revisit_radius": 0.05,

    # Chronic-motion spatial prior (fixed-camera detection filter).
    # Accumulates a per-pixel motion-frequency map over the clip; pixels that
    # move in at least `chronic_motion_threshold` of frames are "chronic"
    # (wind-blown vegetation, rippling water, etc.). A detection whose bounding
    # box overlaps chronic pixels by more than `max_chronic_overlap` (fraction
    # of bbox area) is dropped BEFORE tracking, cutting clutter and speeding up
    # tracking. Insects are transient visitors, so they rarely sit on
    # chronically-moving pixels. Default off preserves behaviour.
    "chronic_motion_suppression": False,
    "chronic_motion_threshold": 0.30,
    "max_chronic_overlap": 0.50,
    # Frames to accumulate before the chronic map is trusted enough to drop
    # detections. During warmup the map still accumulates but no boxes are
    # dropped, avoiding cold-start over-suppression when frequencies are noisy.
    "chronic_motion_warmup_frames": 30,

    # Detection resolution — explicit (width, height) in pixels to run the
    # detector at. Detection runs on frames resized to this resolution for
    # speed while bounding boxes are scaled back to native resolution, so
    # tracking, crops, and composites stay full-res. None = native resolution.
    "detection_resolution": None,

    # Reference resolution — explicit (width, height) in pixels the
    # absolute-pixel params (``morph_kernel_size``, ``min_density``) were tuned
    # for. They are auto-scaled from this to the actual detection resolution,
    # so a config authored for, say, 4K behaves correctly at any native or
    # detection resolution. None = treat the native frame size as the reference.
    "reference_resolution": None,
}


# Keys whose config values are FRACTIONS that need resolving to pixels.
# ``morph_kernel_size`` is intentionally NOT a fraction — it is a local
# noise-removal kernel and is configured as an absolute NxN pixel size.
_AREA_FRACTION_KEYS = ("min_area", "max_area")
_LENGTH_FRACTION_KEYS = (
    "min_displacement",
    "max_frame_jump",
    "revisit_radius",
)


def get_default_config() -> Dict:
    """Return a copy of the default detection configuration (fractions)."""
    return DEFAULT_DETECTION_CONFIG.copy()


def build_detection_params(**kwargs) -> Dict:
    """Build detection parameters from defaults + overrides."""
    params = get_default_config()
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            raise ValueError(f"Unknown detection parameter: {key}")
    return params


def resolve_detection_params(
    params: Dict, image_width: int, image_height: int
) -> Dict:
    """
    Resolve fraction-based config values into absolute pixel values.

    Returns a new dict where the pixel-scale keys (``min_area``,
    ``max_area``, ``min_displacement``, ``max_frame_jump``,
    ``revisit_radius``) hold resolved pixel values, while
    ``{key}_frac`` companions preserve the original fractions for
    record/logging. The dict also records ``_image_width``,
    ``_image_height``, and ``_image_area`` for downstream use.

    Area keys multiply the fraction by image area (W * H).
    Length keys multiply the fraction by image width W.

    ``morph_kernel_size`` is intentionally *not* resolved — it is a
    local noise-removal kernel configured as an absolute NxN pixel
    size (default 3) and is passed through as-is.

    A WARNING is emitted for any fraction > 1.0 — this very likely
    indicates an old-style config with absolute pixels.
    """
    image_area = image_width * image_height

    resolved = dict(params)

    for key in _AREA_FRACTION_KEYS:
        if key in resolved and resolved[key] is not None:
            frac = float(resolved[key])
            if frac > 1.0:
                logger.warning(
                    "%s=%s looks like an absolute pixel value (>1.0); "
                    "expected a fraction of image area. Treating as fraction.",
                    key, frac,
                )
            resolved[f"{key}_frac"] = frac
            resolved[key] = frac * image_area

    for key in _LENGTH_FRACTION_KEYS:
        if key in resolved and resolved[key] is not None:
            frac = float(resolved[key])
            if frac > 1.0:
                logger.warning(
                    "%s=%s looks like an absolute pixel value (>1.0); "
                    "expected a fraction of image width. Treating as fraction.",
                    key, frac,
                )
            resolved[f"{key}_frac"] = frac
            resolved[key] = frac * image_width

    resolved["_image_width"] = image_width
    resolved["_image_height"] = image_height
    resolved["_image_area"] = image_area

    return resolved


# =============================================================================
# MOTION DETECTOR
# =============================================================================

class MotionDetector:
    """
    Motion-based detector using GMM background subtraction.

    Detects moving objects and filters by shape/cohesiveness
    to identify likely insects vs plants/noise.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=params.get("gmm_history", 500),
            varThreshold=params.get("gmm_var_threshold", 16),
            detectShadows=False,
        )
        kernel_size = params.get("morph_kernel_size", 3)
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> Tuple[List[Detection], np.ndarray]:
        """Detect insects in a single frame. Returns (detections, fg_mask)."""
        fg_mask = self.back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = frame.shape[:2]
        detections = []

        for contour in contours:
            if not passes_shape_filters(
                contour,
                self.params["min_area"],
                self.params["max_area"],
                self.params["min_density"],
                self.params["min_solidity"],
            ):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            region = fg_mask[y : y + h, x : x + w]
            cohesive, _ = is_cohesive_blob(
                region, w * h,
                self.params["min_largest_blob_ratio"],
                self.params["max_num_blobs"],
                self.params.get("min_motion_ratio", 0.15),
            )
            if not cohesive:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                area=cv2.contourArea(contour),
                frame_number=frame_number,
            ))

        return detections, fg_mask

    def reset(self) -> None:
        """Reset background model."""
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=self.params.get("gmm_history", 500),
            varThreshold=self.params.get("gmm_var_threshold", 16),
            detectShadows=False,
        )


# =============================================================================
# SCALED DETECTOR (optional downscaled detection)
# =============================================================================

class ScaledDetector:
    """
    A ``MotionDetector`` that optionally runs detection at a lower resolution
    for speed, returning bounding boxes mapped back to native pixels.

    This is the single place that owns the detection-resolution policy so that
    every consumer (the ``DetectionPipeline`` here, plus external callers that
    build their own frame loop, e.g. bplusplus) behaves identically.

    Behaviour:
        * If ``detection_resolution`` (a ``(width, height)`` pair) is set in the
          config, frames are resized to it before detection and detection
          bounding boxes are scaled back to native resolution.
        * Detector params are resolved at the DETECTION resolution so the
          fraction-based area/length thresholds match the frames it sees.
        * Two length-dimensioned absolute-pixel params that
          ``resolve_detection_params`` does not touch (``morph_kernel_size`` and
          ``min_density``) are scaled from the REFERENCE resolution to the
          DETECTION resolution by the geometric mean of the x/y factors:
            - ``morph_kernel_size`` — otherwise a fixed kernel is the wrong
              size for the frame and MORPH_CLOSE-merges scattered motion into
              compact blobs (false positives);
            - ``min_density`` (area / perimeter) — otherwise real objects,
              whose density scales ~linearly with resolution, get rejected.
          Dimensionless filters (``min_solidity``, ``min_largest_blob_ratio``,
          ``min_motion_ratio``, ``max_num_blobs``) are scale-invariant and are
          left unchanged.

        ``reference_resolution`` (a ``(width, height)`` pair) declares the
        resolution the absolute-pixel params were authored for. It defaults to
        the NATIVE frame size, so:
            - unset + no downscale  -> no change (params used as written);
            - unset + downscale     -> params scaled native -> detection;
            - set (e.g. 4K) + any native/detection -> params scaled
              reference -> detection, so a "4K config" works at any resolution.

    The detector is created at construction; pass the NATIVE frame size.
    """

    def __init__(self, config: Dict, native_width: int, native_height: int):
        det_resolution = config.get("detection_resolution")
        det_width, det_height = native_width, native_height
        if det_resolution:
            det_width = max(1, int(det_resolution[0]))
            det_height = max(1, int(det_resolution[1]))

        self.native_width = native_width
        self.native_height = native_height
        self.det_width = det_width
        self.det_height = det_height
        self.downscaled = (det_width, det_height) != (native_width, native_height)
        # Scale factors map detection-space coords back up to native pixels.
        # x and y are independent so non-matching aspect ratios are handled.
        self.scale_x = native_width / det_width
        self.scale_y = native_height / det_height

        # The reference resolution the absolute-pixel params were authored for.
        # Defaults to native, which preserves the no-reference behaviour.
        ref_resolution = config.get("reference_resolution")
        if ref_resolution:
            ref_width = max(1, int(ref_resolution[0]))
            ref_height = max(1, int(ref_resolution[1]))
        else:
            ref_width, ref_height = native_width, native_height
        self.reference_width = ref_width
        self.reference_height = ref_height

        params = resolve_detection_params(config, det_width, det_height)

        # Scale the length-dimensioned absolute-pixel params from the reference
        # resolution to the detection resolution (geometric mean of x/y handles
        # non-uniform aspect). When detection == reference this is a no-op.
        length_scale = (det_width / ref_width * det_height / ref_height) ** 0.5
        if length_scale != 1.0:
            kernel = params.get("morph_kernel_size", 3)
            params["morph_kernel_size"] = max(1, int(round(kernel * length_scale)))
            if params.get("min_density"):
                params["min_density"] = params["min_density"] * length_scale

        self.length_scale = length_scale
        self.params = params
        self.detector = MotionDetector(params)

        # Chronic-motion spatial prior (fixed-camera FP suppression). Lives
        # here so EVERY consumer that calls detect() — including callers that
        # run their own tracking loop (e.g. bplusplus) — gets chronic boxes
        # dropped BEFORE their tracker. Off by default = unchanged behaviour.
        self.chronic_enabled = bool(config.get("chronic_motion_suppression", False))
        self.chronic_threshold = float(config.get("chronic_motion_threshold", 0.30))
        self.max_chronic_overlap = float(config.get("max_chronic_overlap", 0.50))
        self.chronic_warmup = int(config.get("chronic_motion_warmup_frames", 30))
        self.chronic_map: Optional["ChronicMotionMap"] = (
            ChronicMotionMap(det_width, det_height) if self.chronic_enabled else None
        )

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect on ``frame`` (native resolution), optionally downscaling first.

        Returns ``(bboxes_native, fg_mask)`` where each bbox is an
        ``(x1, y1, x2, y2)`` tuple in NATIVE pixel coordinates. ``fg_mask`` is
        the detection-resolution foreground mask (diagnostic only).

        When chronic-motion suppression is enabled, the running motion map is
        updated from this frame and detections sitting on chronically-moving
        pixels are removed from the returned list (after a warmup period), so
        clutter never reaches the caller's tracker.
        """
        if self.downscaled:
            detect_frame = cv2.resize(
                frame, (self.det_width, self.det_height), interpolation=cv2.INTER_AREA
            )
        else:
            detect_frame = frame

        detections, fg_mask = self.detector.detect(detect_frame, frame_number)

        bboxes: List[Tuple[int, int, int, int]] = []
        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            if self.downscaled:
                nx1 = max(0, min(int(round(dx1 * self.scale_x)), self.native_width))
                ny1 = max(0, min(int(round(dy1 * self.scale_y)), self.native_height))
                nx2 = max(0, min(int(round(dx2 * self.scale_x)), self.native_width))
                ny2 = max(0, min(int(round(dy2 * self.scale_y)), self.native_height))
            else:
                nx1, ny1, nx2, ny2 = dx1, dy1, dx2, dy2
            bboxes.append((nx1, ny1, nx2, ny2))

        if self.chronic_map is not None:
            # Accumulate first so the current frame counts, then drop chronic
            # detections (skipped during warmup while frequencies are noisy).
            self.chronic_map.update(fg_mask)
            if self.chronic_map.frames >= self.chronic_warmup:
                bboxes = [
                    b for b in bboxes
                    if self.chronic_overlap_native(b) <= self.max_chronic_overlap
                ]

        return bboxes, fg_mask

    def chronic_overlap_native(self, bbox_native: Tuple[int, int, int, int]) -> float:
        """
        Fraction of a NATIVE-pixel bbox that sits on chronically-moving pixels.

        Maps the box into detection-resolution coordinates (where the chronic
        map lives) and queries it. Returns 0.0 when chronic tracking is off.
        """
        if self.chronic_map is None:
            return 0.0
        x1, y1, x2, y2 = bbox_native
        inv_x = 1.0 / self.scale_x  # native -> detection
        inv_y = 1.0 / self.scale_y
        det_box = (x1 * inv_x, y1 * inv_y, x2 * inv_x, y2 * inv_y)
        return self.chronic_map.overlap_ratio(det_box, self.chronic_threshold)

    def reset(self) -> None:
        """Reset the underlying background model (and chronic map if present)."""
        self.detector.reset()
        if self.chronic_map is not None:
            self.chronic_map = ChronicMotionMap(self.det_width, self.det_height)


# =============================================================================
# CHRONIC-MOTION SPATIAL PRIOR (fixed-camera false-positive suppression)
# =============================================================================

class ChronicMotionMap:
    """
    Accumulates a per-pixel motion-frequency map over a clip and answers
    "how chronically does this region move?" queries.

    Motivation: on a FIXED camera, wind-blown vegetation, rippling water and
    similar clutter move in the *same image regions* throughout the clip,
    whereas a real insect is a transient visitor that passes through a region
    once. Suppressing detections that sit on chronically-moving pixels removes
    a large class of false positives without any per-object tuning.

    The map is accumulated at whatever resolution the foreground masks are
    produced at (i.e. the DETECTION resolution when downscaling is used), so
    bounding boxes must be mapped into that space before querying. The map is
    resolution-agnostic otherwise.

    Usage:
        cmap = ChronicMotionMap(mask_width, mask_height)
        for frame:
            _, fg_mask = detector.detect(frame)
            cmap.update(fg_mask)
        ratio = cmap.overlap_ratio(bbox, threshold)  # 0..1
    """

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        # uint32 counts: number of frames each pixel was foreground.
        self._counts = np.zeros((self.height, self.width), dtype=np.uint32)
        self._frames = 0

    @property
    def frames(self) -> int:
        return self._frames

    def update(self, fg_mask: np.ndarray) -> None:
        """Accumulate one foreground mask (non-zero = motion)."""
        if fg_mask is None:
            return
        if fg_mask.shape[:2] != (self.height, self.width):
            fg_mask = cv2.resize(
                fg_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        self._counts += (fg_mask > 0).astype(np.uint32)
        self._frames += 1

    def frequency(self) -> np.ndarray:
        """Per-pixel motion frequency in [0, 1] (fraction of frames in motion)."""
        if self._frames == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return self._counts.astype(np.float32) / float(self._frames)

    def chronic_mask(self, threshold: float = 0.30) -> np.ndarray:
        """Boolean map of pixels that move in >= ``threshold`` of frames."""
        return self.frequency() >= float(threshold)

    def overlap_ratio(
        self, bbox: Tuple[int, int, int, int], threshold: float = 0.30
    ) -> float:
        """
        Fraction of ``bbox`` (in map coordinates) covered by chronic pixels.

        Returns 0.0 when there is no data or the box is empty/out of bounds.
        """
        if self._frames == 0:
            return 0.0
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region_freq = self.frequency()[y1:y2, x1:x2]
        if region_freq.size == 0:
            return 0.0
        return float(np.mean(region_freq >= float(threshold)))


# =============================================================================
# SHAPE AND COHESIVENESS FILTERS
# =============================================================================

def is_cohesive_blob(
    fg_mask_region: np.ndarray,
    bbox_area: int,
    min_largest_blob_ratio: float = 0.80,
    max_num_blobs: int = 5,
    min_motion_ratio: float = 0.15,
) -> Tuple[bool, Optional[Dict]]:
    """Check if motion is cohesive (insect) vs scattered (plant)."""
    motion_pixels = np.count_nonzero(fg_mask_region)
    if motion_pixels == 0:
        return False, None

    motion_ratio = motion_pixels / bbox_area
    if motion_ratio < min_motion_ratio:
        return False, None

    contours, _ = cv2.findContours(fg_mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, None

    if len(contours) > max_num_blobs:
        return False, None

    largest = max(contours, key=cv2.contourArea)
    largest_ratio = cv2.contourArea(largest) / motion_pixels
    if largest_ratio < min_largest_blob_ratio:
        return False, None

    return True, {
        "motion_ratio": motion_ratio,
        "num_blobs": len(contours),
        "largest_blob_ratio": largest_ratio,
    }


def passes_shape_filters(
    contour,
    min_area: int = 200,
    max_area: int = 40000,
    min_density: float = 3.0,
    min_solidity: float = 0.55,
) -> bool:
    """Check if contour passes size and shape filters."""
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    if area / perimeter < min_density:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area > 0 and (area / hull_area) < min_solidity:
        return False

    return True


# =============================================================================
# PATH TOPOLOGY ANALYSIS
# =============================================================================

def calculate_revisit_ratio(path: np.ndarray, revisit_radius: int = 50) -> float:
    """Low = exploring new areas (insect), High = oscillating (plant)."""
    revisit_count = 0
    for i in range(len(path)):
        for j in range(i):
            if np.linalg.norm(path[i] - path[j]) < revisit_radius:
                revisit_count += 1
    max_revisits = len(path) * (len(path) - 1) / 2
    return revisit_count / (max_revisits + 1e-6)


def calculate_progression_ratio(path: np.ndarray) -> float:
    """High = linear progression (insect), Low = backtracking (plant)."""
    if len(path) < 2:
        return 0
    net = np.linalg.norm(path[-1] - path[0])
    max_dist = max(np.linalg.norm(p - path[0]) for p in path)
    return net / (max_dist + 1e-6)


def calculate_directional_variance(path: np.ndarray) -> float:
    """Low = consistent direction (insect), High = random (plant)."""
    if len(path) < 2:
        return 1.0
    directions = []
    for i in range(1, len(path)):
        dx, dy = path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]
        if dx != 0 or dy != 0:
            directions.append(np.arctan2(dy, dx))
    if not directions:
        return 1.0
    return 1 - np.sqrt(np.mean(np.sin(directions)) ** 2 + np.mean(np.cos(directions)) ** 2)


def analyze_path_topology(path, params: Dict) -> Tuple[bool, Dict]:
    """Analyze path for insect-like movement. Returns (passes, metrics)."""
    if len(path) < params.get("min_path_points", 10):
        return False, {}

    path_arr = np.array(path)
    net_displacement = float(np.linalg.norm(path_arr[-1] - path_arr[0]))
    revisit_ratio = calculate_revisit_ratio(path_arr, params.get("revisit_radius", 50))
    progression_ratio = calculate_progression_ratio(path_arr)
    directional_variance = calculate_directional_variance(path_arr)

    metrics = {
        "net_displacement": net_displacement,
        "revisit_ratio": revisit_ratio,
        "progression_ratio": progression_ratio,
        "directional_variance": directional_variance,
    }

    passes = (
        net_displacement >= params["min_displacement"]
        and revisit_ratio <= params["max_revisit_ratio"]
        and progression_ratio >= params["min_progression_ratio"]
        and directional_variance <= params["max_directional_variance"]
    )
    return passes, metrics


# =============================================================================
# TRACK CONSISTENCY
# =============================================================================

def check_track_consistency(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
    prev_area: float,
    curr_area: float,
    max_frame_jump: int,
    max_area_change_ratio: float = 3.0,
) -> bool:
    """Check if track update is consistent (not a bad match)."""
    if np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) > max_frame_jump:
        return False
    ratio = max(curr_area, prev_area) / (min(curr_area, prev_area) + 1e-6)
    if ratio > max_area_change_ratio:
        return False
    return True

