"""
Motion-based insect detection.

Provides GMM background subtraction with shape and cohesiveness
filters to identify insects, plus path topology analysis.

Dependencies: opencv, numpy (no ML frameworks)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


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

DEFAULT_DETECTION_CONFIG = {
    # GMM Background Subtractor
    "gmm_history": 500,
    "gmm_var_threshold": 16,

    # Morphological filtering
    "morph_kernel_size": 3,

    # Cohesiveness
    "min_largest_blob_ratio": 0.80,
    "max_num_blobs": 5,
    "min_motion_ratio": 0.15,

    # Shape
    "min_area": 200,
    "max_area": 40000,
    "min_density": 3.0,
    "min_solidity": 0.55,

    # Tracking
    "min_displacement": 50,
    "min_path_points": 10,
    "max_frame_jump": 100,
    "max_lost_frames": 45,
    "max_area_change_ratio": 3.0,

    # Tracker matching
    "tracker_w_dist": 0.6,
    "tracker_w_area": 0.4,
    "tracker_cost_threshold": 0.3,

    # Path topology
    "max_revisit_ratio": 0.30,
    "min_progression_ratio": 0.70,
    "max_directional_variance": 0.90,
    "revisit_radius": 50,
}


def get_default_config() -> Dict:
    """Return a copy of the default detection configuration."""
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
    if len(path) < 3:
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

