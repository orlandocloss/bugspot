"""
Detection pipeline: video → confirmed tracks + crops + composites.

This is the core pipeline shared by bplusplus (desktop) and edge26 (device).
Classification is NOT included — each consumer adds their own classifier.

Dependencies: opencv, numpy, scipy
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime

from .detector import (
    MotionDetector,
    Detection,
    analyze_path_topology,
    check_track_consistency,
    get_default_config,
)
from .tracker import InsectTracker


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class TrackResult:
    """Result for a single confirmed track."""
    track_id: str
    num_detections: int
    first_frame_time: float
    last_frame_time: float
    duration: float
    topology_metrics: Dict
    crops: List[Tuple[int, np.ndarray]] = field(default_factory=list, repr=False)
    composite: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PipelineResult:
    """Complete pipeline output for one video."""
    confirmed_tracks: Dict[str, TrackResult]
    all_detections: List[Dict]
    track_paths: Dict[str, List[Tuple[float, float]]]
    video_info: Dict


# =============================================================================
# DETECTION PIPELINE
# =============================================================================

class DetectionPipeline:
    """
    Stateful detection pipeline.

    Processes video files through:
        1. Detection & Tracking (frame loop)
        2. Topology Analysis (confirm tracks)
        3. Crop Extraction (re-read confirmed frames)
        4. Composite Rendering (lighten blend per track)

    Supports both single-video and continuous multi-chunk operation.

    Usage:
        pipeline = DetectionPipeline(config)
        result = pipeline.process_video("video.mp4")

        # result.confirmed_tracks has crops and composites
        for track_id, track in result.confirmed_tracks.items():
            for frame_num, crop in track.crops:
                classification = your_classifier(crop)

        # For continuous operation (edge26):
        pipeline.clear()  # keep tracker state, clear detections
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Detection parameters dict. None = use defaults.
        """
        self.config = config or get_default_config()

        # Components
        self._detector = MotionDetector(self.config)
        self._tracker: Optional[InsectTracker] = None

        # Per-video state
        self.all_detections: List[Dict] = []
        self.track_paths: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.track_areas: Dict[str, List[float]] = defaultdict(list)

    def process_video(
        self,
        video_path: str,
        extract_crops: bool = True,
        render_composites: bool = True,
        save_crops_dir: Optional[str] = None,
        save_composites_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full detection pipeline on a video.

        Args:
            video_path: Path to input video
            extract_crops: Whether to extract crop images for confirmed tracks
            render_composites: Whether to render composite images
            save_crops_dir: Optional directory to save crop images to disk
            save_composites_dir: Optional directory to save composite images to disk

        Returns:
            PipelineResult with confirmed tracks, crops, and composites
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_info = {
            "fps": input_fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": total_frames / input_fps if input_fps > 0 else 0,
        }

        # Initialise tracker on first call (or if frame size changed)
        if self._tracker is None:
            self._tracker = InsectTracker(
                image_height=height,
                image_width=width,
                max_lost_frames=self.config.get("max_lost_frames", 45),
                w_dist=self.config.get("tracker_w_dist", 0.6),
                w_area=self.config.get("tracker_w_area", 0.4),
                cost_threshold=self.config.get("tracker_cost_threshold", 0.3),
            )

        # =====================================================================
        # PHASE 1: Detection & Tracking
        # =====================================================================
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_num / input_fps if input_fps > 0 else 0

            detections, _ = self._detector.detect(frame, frame_num)
            bboxes = [d.bbox for d in detections]
            track_ids = self._tracker.update(bboxes, frame_num)

            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det.bbox
                track_id = track_ids[i] if i < len(track_ids) else None
                if not track_id:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                # Track consistency check
                if self.track_paths[track_id]:
                    prev_pos = self.track_paths[track_id][-1]
                    prev_area = self.track_areas[track_id][-1] if self.track_areas[track_id] else area
                    if not check_track_consistency(
                        prev_pos, (cx, cy), prev_area, area,
                        self.config["max_frame_jump"],
                        self.config.get("max_area_change_ratio", 3.0),
                    ):
                        self.track_paths[track_id] = []
                        self.track_areas[track_id] = []

                self.track_paths[track_id].append((cx, cy))
                self.track_areas[track_id].append(area)

                self.all_detections.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame_number": frame_num,
                    "frame_time_seconds": frame_time,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_normalized": [
                        (x1 + x2) / (2 * width),
                        (y1 + y2) / (2 * height),
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                })

            frame_num += 1

        cap.release()

        # =====================================================================
        # PHASE 2: Topology Analysis
        # =====================================================================
        confirmed_ids, all_track_info = self._analyze_tracks()

        # =====================================================================
        # PHASE 3: Crop Extraction
        # =====================================================================
        track_crops: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        if extract_crops and confirmed_ids:
            track_crops = self._extract_crops(video_path, confirmed_ids, save_crops_dir)

        # =====================================================================
        # PHASE 4: Composite Rendering
        # =====================================================================
        track_composites: Dict[str, np.ndarray] = {}
        if render_composites and confirmed_ids:
            track_composites = self._render_composites(video_path, confirmed_ids, save_composites_dir)

        # Build results
        confirmed_tracks: Dict[str, TrackResult] = {}
        for track_id in confirmed_ids:
            info = all_track_info[track_id]
            confirmed_tracks[track_id] = TrackResult(
                track_id=track_id,
                num_detections=info["num_detections"],
                first_frame_time=info["first_frame_time"],
                last_frame_time=info["last_frame_time"],
                duration=info["duration"],
                topology_metrics={
                    k: info[k]
                    for k in ["net_displacement", "revisit_ratio", "progression_ratio", "directional_variance"]
                    if k in info
                },
                crops=track_crops.get(track_id, []),
                composite=track_composites.get(track_id),
            )

        # Advance tracker for continuous operation
        self._tracker.finalize_video(frame_num)

        return PipelineResult(
            confirmed_tracks=confirmed_tracks,
            all_detections=self.all_detections.copy(),
            track_paths=dict(self.track_paths),
            video_info=video_info,
        )

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _analyze_tracks(self) -> Tuple[Set[str], Dict]:
        """Topology analysis on all tracks."""
        track_detections: Dict[str, List[Dict]] = defaultdict(list)
        for det in self.all_detections:
            if det["track_id"]:
                track_detections[det["track_id"]].append(det)

        confirmed: Set[str] = set()
        info: Dict = {}

        for track_id, dets in track_detections.items():
            path = self.track_paths.get(track_id, [])
            passes, metrics = analyze_path_topology(path, self.config)
            frame_times = [d["frame_time_seconds"] for d in dets]

            info[track_id] = {
                "track_id": track_id,
                "num_detections": len(dets),
                "first_frame_time": min(frame_times),
                "last_frame_time": max(frame_times),
                "duration": max(frame_times) - min(frame_times),
                "passes_topology": passes,
                **metrics,
            }
            if passes:
                confirmed.add(track_id)

        return confirmed, info

    def _extract_crops(
        self,
        video_path: str,
        confirmed_ids: Set[str],
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[Tuple[int, np.ndarray]]]:
        """Extract crop images for confirmed tracks."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for tid in confirmed_ids:
                os.makedirs(os.path.join(save_dir, str(tid)[:8]), exist_ok=True)

        frames_needed: Dict[int, List[Dict]] = defaultdict(list)
        for det in self.all_detections:
            if det["track_id"] in confirmed_ids:
                frames_needed[det["frame_number"]].append(det)

        if not frames_needed:
            return {}

        cap = cv2.VideoCapture(video_path)
        crops: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        current = 0

        for target in sorted(frames_needed.keys()):
            while current < target:
                cap.read()
                current += 1
            ret, frame = cap.read()
            if not ret:
                break
            current += 1

            for det in frames_needed[target]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops[det["track_id"]].append((target, crop.copy()))
                    if save_dir:
                        path = os.path.join(save_dir, str(det["track_id"])[:8], f"frame_{target:06d}.jpg")
                        cv2.imwrite(path, crop)

        cap.release()
        return dict(crops)

    def _render_composites(
        self,
        video_path: str,
        confirmed_ids: Set[str],
        save_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Render composite images (lighten blend on darkened background)."""
        track_frames: Dict[str, List[Dict]] = defaultdict(list)
        for det in self.all_detections:
            if det["track_id"] in confirmed_ids:
                track_frames[det["track_id"]].append(det)

        needed: Dict[int, Set[str]] = defaultdict(set)
        for tid, dets in track_frames.items():
            for det in dets:
                needed[det["frame_number"]].add(tid)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        composites: Dict[str, np.ndarray] = {}
        bg_set: Set[str] = set()
        BG_DARKEN = 0.35
        current = 0

        for target in sorted(needed.keys()):
            while current < target:
                cap.read()
                current += 1
            ret, frame = cap.read()
            if not ret:
                break
            current += 1

            for tid in needed[target]:
                if tid not in bg_set:
                    composites[tid] = frame.copy().astype(np.float64) * BG_DARKEN
                    bg_set.add(tid)

                for det in track_frames[tid]:
                    if det["frame_number"] == target:
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        if x2 > x1 and y2 > y1:
                            region = frame[y1:y2, x1:x2].astype(np.float64)
                            composites[tid][y1:y2, x1:x2] = np.maximum(
                                composites[tid][y1:y2, x1:x2], region
                            )
                        break

        cap.release()

        # Finalize and optionally save
        result: Dict[str, np.ndarray] = {}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for tid, comp in composites.items():
            img = np.clip(comp, 0, 255).astype(np.uint8)

            # Start/end markers
            path = self.track_paths.get(tid, [])
            if len(path) > 1:
                sx, sy = int(path[0][0]), int(path[0][1])
                ex, ey = int(path[-1][0]), int(path[-1][1])
                cv2.circle(img, (sx, sy), 6, (0, 255, 0), -1)
                cv2.circle(img, (ex, ey), 6, (0, 0, 255), -1)

            n_dets = len(track_frames[tid])
            cv2.putText(img, f"{n_dets} detections", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            result[tid] = img

            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"track_{str(tid)[:8]}.jpg"), img)

        return result

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def clear(self) -> None:
        """Clear per-video detections. Keeps tracker state for continuous operation."""
        self.all_detections = []
        self._detector.reset()

    def reset(self) -> None:
        """Full reset — clear everything including tracker."""
        self.all_detections = []
        self.track_paths = defaultdict(list)
        self.track_areas = defaultdict(list)
        self._detector.reset()
        self._tracker = None

