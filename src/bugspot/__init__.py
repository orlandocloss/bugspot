"""
BugSpot: Lightweight insect detection and tracking core.

Provides motion-based insect detection, Hungarian tracking,
and path topology analysis. No ML framework dependencies.

Dependencies: opencv, numpy, scipy
"""

from .detector import (
    Detection,
    MotionDetector,
    DEFAULT_DETECTION_CONFIG,
    get_default_config,
    build_detection_params,
    is_cohesive_blob,
    passes_shape_filters,
    analyze_path_topology,
    check_track_consistency,
)
from .tracker import (
    BoundingBox,
    Track,
    InsectTracker,
)
from .pipeline import (
    TrackResult,
    PipelineResult,
    DetectionPipeline,
)

__all__ = [
    # Detector
    "Detection",
    "MotionDetector",
    "DEFAULT_DETECTION_CONFIG",
    "get_default_config",
    "build_detection_params",
    "is_cohesive_blob",
    "passes_shape_filters",
    "analyze_path_topology",
    "check_track_consistency",
    # Tracker
    "BoundingBox",
    "Track",
    "InsectTracker",
    # Pipeline
    "TrackResult",
    "PipelineResult",
    "DetectionPipeline",
]

