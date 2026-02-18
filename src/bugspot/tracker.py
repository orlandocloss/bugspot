"""
Insect tracking using Hungarian algorithm.

Supports both single-video and continuous multi-chunk operation.
Tracks persist across video boundaries when using finalize_video().

Dependencies: numpy, scipy
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box with tracking metadata."""
    x: float
    y: float
    width: float
    height: float
    frame_id: int
    track_id: Optional[str] = None

    @property
    def area(self) -> float:
        return self.width * self.height

    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, frame_id, track_id=None):
        """Create from x1,y1,x2,y2 coordinates."""
        return cls(x1, y1, x2 - x1, y2 - y1, frame_id, track_id)


@dataclass
class Track:
    """Track with metadata."""
    track_id: str
    first_frame: int = 0
    last_frame: int = 0
    is_active: bool = True


# =============================================================================
# TRACKER
# =============================================================================

class InsectTracker:
    """
    Insect tracker using Hungarian algorithm for optimal assignment.

    Features:
        - Lost track recovery within memory window
        - Weighted distance + area cost function
        - Continuous tracking across video chunks (finalize_video)
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        max_lost_frames: int = 45,
        w_dist: float = 0.6,
        w_area: float = 0.4,
        cost_threshold: float = 0.3,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.max_dist = np.sqrt(image_height ** 2 + image_width ** 2)
        self.max_lost_frames = max_lost_frames
        self.w_dist = w_dist
        self.w_area = w_area
        self.cost_threshold = cost_threshold

        # State
        self.current_tracks: List[BoundingBox] = []
        self.lost_tracks: Dict[str, Dict] = {}
        self.all_tracks: Dict[str, Track] = {}
        self.global_frame_count: int = 0

    def update(self, detections: List, frame_id: int) -> List[Optional[str]]:
        """
        Update tracking with new detections.

        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes
            frame_id: Current frame number

        Returns:
            List of track IDs corresponding to each detection
        """
        global_frame = self.global_frame_count + frame_id

        if not detections:
            self._move_all_to_lost()
            self._age_lost_tracks()
            return []

        new_boxes = [BoundingBox.from_xyxy(*det[:4], global_frame) for det in detections]

        if not self.current_tracks and not self.lost_tracks:
            track_ids = self._assign_new_ids(new_boxes, global_frame)
            self.current_tracks = new_boxes
            return track_ids

        all_previous = self.current_tracks.copy()
        for tid, info in self.lost_tracks.items():
            box = info["box"]
            box.track_id = tid
            all_previous.append(box)

        if not all_previous:
            track_ids = self._assign_new_ids(new_boxes, global_frame)
            self.current_tracks = new_boxes
            return track_ids

        # Hungarian assignment
        cost_matrix, n_prev, n_curr = self._build_cost_matrix(all_previous, new_boxes)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        track_ids: List[Optional[str]] = [None] * len(new_boxes)
        assigned: Set[int] = set()
        recovered: Set[str] = set()

        for i, j in zip(row_idx, col_idx):
            if i < n_prev and j < n_curr and cost_matrix[i, j] < self.cost_threshold:
                prev_id = all_previous[i].track_id
                new_boxes[j].track_id = prev_id
                track_ids[j] = prev_id
                assigned.add(j)
                if prev_id in self.lost_tracks:
                    recovered.add(prev_id)

        for tid in recovered:
            del self.lost_tracks[tid]

        for j in range(n_curr):
            if j not in assigned:
                new_id = str(uuid.uuid4())
                new_boxes[j].track_id = new_id
                track_ids[j] = new_id
                self._create_track(new_id, global_frame)

        matched_ids = {track_ids[j] for j in assigned if track_ids[j]}
        for track in self.current_tracks:
            if track.track_id not in matched_ids and track.track_id not in recovered:
                if track.track_id not in self.lost_tracks:
                    self.lost_tracks[track.track_id] = {"box": track, "frames_lost": 1}

        self._age_lost_tracks()
        self.current_tracks = new_boxes
        return track_ids

    def finalize_video(self, frames_in_video: int) -> None:
        """Advance global frame counter after processing a video chunk."""
        self.global_frame_count += frames_in_video

    def get_stats(self) -> Dict:
        """Current tracking statistics."""
        return {
            "active_tracks": len(self.current_tracks),
            "lost_tracks": len(self.lost_tracks),
            "total_tracks": len(self.all_tracks),
            "global_frame_count": self.global_frame_count,
        }

    # -- internals --

    def _build_cost_matrix(self, prev, curr):
        n_prev, n_curr = len(prev), len(curr)
        n = max(n_prev, n_curr)
        cost = np.ones((n, n)) * 999.0
        for i in range(n_prev):
            for j in range(n_curr):
                cost[i, j] = self._cost(prev[i], curr[j])
        return cost, n_prev, n_curr

    def _cost(self, b1: BoundingBox, b2: BoundingBox) -> float:
        cx1, cy1 = b1.center()
        cx2, cy2 = b2.center()
        norm_dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) / self.max_dist
        min_a, max_a = min(b1.area, b2.area), max(b1.area, b2.area)
        area_cost = 1 - (min_a / max_a if max_a > 0 else 1.0)
        return norm_dist * self.w_dist + area_cost * self.w_area

    def _assign_new_ids(self, boxes, global_frame):
        ids = []
        for box in boxes:
            new_id = str(uuid.uuid4())
            box.track_id = new_id
            ids.append(new_id)
            self._create_track(new_id, global_frame)
        return ids

    def _create_track(self, track_id, first_frame):
        if track_id not in self.all_tracks:
            self.all_tracks[track_id] = Track(
                track_id=track_id, first_frame=first_frame, last_frame=first_frame
            )

    def _move_all_to_lost(self):
        for t in self.current_tracks:
            if t.track_id not in self.lost_tracks:
                self.lost_tracks[t.track_id] = {"box": t, "frames_lost": 1}
        self.current_tracks = []

    def _age_lost_tracks(self):
        remove = [
            tid for tid, info in self.lost_tracks.items()
            if info.update({"frames_lost": info["frames_lost"] + 1}) is None
            and info["frames_lost"] > self.max_lost_frames
        ]
        for tid in remove:
            if tid in self.all_tracks:
                self.all_tracks[tid].is_active = False
            del self.lost_tracks[tid]

