# BugSpot

[![PyPI](https://img.shields.io/pypi/v/bugspot)](https://pypi.org/project/bugspot/)
[![Python](https://img.shields.io/pypi/pyversions/bugspot)](https://pypi.org/project/bugspot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight insect detection and tracking using motion-based GMM background subtraction, Hungarian algorithm tracking, and path topology analysis. Core library for [B++](https://github.com/Tvenver/Bplusplus) and [Sensing Garden](https://github.com/MIT-Senseable-City-Lab/sensing-garden).

**No ML framework dependencies** — only requires OpenCV, NumPy, and SciPy.

## Installation

```bash
pip install bugspot
```

## Quick Start

### Command Line

```bash
# Run with defaults
bugspot video.mp4

# Run with a custom config
bugspot video.mp4 --config detection_config.yaml --output results/
```

### Python API

```python
from bugspot import DetectionPipeline

pipeline = DetectionPipeline()
result = pipeline.process_video("video.mp4")

print(f"Confirmed: {len(result.confirmed_tracks)} tracks")

for track_id, track in result.confirmed_tracks.items():
    print(f"  Track {track_id[:8]}: {track.num_detections} detections, {track.duration:.1f}s")

    for frame_num, crop in track.crops:
        pass  # feed crop to your classifier

    if track.composite is not None:
        import cv2
        cv2.imwrite(f"track_{track_id[:8]}.jpg", track.composite)
```

### Save Outputs to Disk

```python
result = pipeline.process_video(
    "video.mp4",
    save_crops_dir="output/crops",
    save_composites_dir="output/composites",
)
```

### Continuous Operation (Multi-Chunk)

For processing video chunks where tracks persist across boundaries:

```python
pipeline = DetectionPipeline(config)

for video_chunk in video_queue:
    result = pipeline.process_video(video_chunk)

    # Process results...

    pipeline.clear()  # Keep tracker state, clear detections
```

### Single Video (Stateless)

For one-off processing without persistent state:

```python
pipeline = DetectionPipeline(config)
result = pipeline.process_video("video.mp4")

pipeline.reset()  # Full reset — clear everything including tracker
```

## Pipeline

1. **Detection** — GMM background subtraction → morphological filtering → shape filters → cohesiveness check
2. **Tracking** — Hungarian algorithm matching with lost track recovery
3. **Topology Analysis** — Path analysis confirms insect-like movement (vs plants/noise)
4. **Crop Extraction** — Re-reads video to extract crop images for confirmed tracks
5. **Composite Rendering** — Lighten blend on darkened background showing temporal trail

## Configuration

See [`detection_config.yaml`](detection_config.yaml) for all parameters with descriptions.

### Resolution-independent units

Every pixel-scale parameter is expressed as a **fraction of the image dimensions**, not as absolute pixels, so the same config works across resolutions. Two reference dimensions are used:

- **Length** → fraction of image width `W` (keys: `morph_kernel_size`, `min_displacement`, `max_frame_jump`, `revisit_radius`)
- **Area**   → fraction of image area `W * H` (keys: `min_area`, `max_area`)

Fractions are resolved to absolute pixels at runtime via `resolve_detection_params(params, W, H)` once the frame size is known. The `1080 px` column below shows the resolved pixel value for a 1080×1080 frame (width = 1080, area = 1 166 400) for intuition.

| Parameter | Default | 1080 px wide | Description |
|-----------|---------|--------------|-------------|
| **GMM** | | | |
| `gmm_history` | 500 | — | Frames to build background model |
| `gmm_var_threshold` | 16 | — | Foreground variance threshold |
| **Morphological** | | | |
| `morph_kernel_size` | 0.002 | 3 | Kernel size, fraction of image width (rounded to odd int) |
| **Cohesiveness** | | | |
| `min_largest_blob_ratio` | 0.80 | — | Min largest blob / total motion |
| `max_num_blobs` | 5 | — | Max blobs in detection |
| `min_motion_ratio` | 0.15 | — | Min motion pixels / bbox area |
| **Shape** | | | |
| `min_area` | 0.0002 | ~233 px² | Min contour area, fraction of image area |
| `max_area` | 0.035 | ~40 824 px² | Max contour area, fraction of image area |
| `min_density` | 3.0 | — | Min area/perimeter ratio (unitless) |
| `min_solidity` | 0.55 | — | Min convex hull fill ratio |
| **Tracking** | | | |
| `min_displacement` | 0.05 | 54 px | Min net movement, fraction of image width |
| `min_path_points` | 10 | — | Min points for topology |
| `max_frame_jump` | 0.1 | 108 px | Max jump between frames, fraction of image width |
| `max_lost_frames` | 45 | — | Frames before track deleted |
| `max_area_change_ratio` | 3.0 | — | Max area change ratio |
| **Tracker Matching** | | | |
| `tracker_w_dist` | 0.6 | — | Distance weight (0-1) |
| `tracker_w_area` | 0.4 | — | Area weight (0-1) |
| `tracker_cost_threshold` | 0.3 | — | Max cost for match |
| **Topology** | | | |
| `max_revisit_ratio` | 0.30 | — | Max revisited positions |
| `min_progression_ratio` | 0.70 | — | Min forward progression |
| `max_directional_variance` | 0.90 | — | Max heading variance |
| `revisit_radius` | 0.05 | 54 px | Revisit radius, fraction of image width |
