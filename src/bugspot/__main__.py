"""
CLI entry point for BugSpot.

Run with:
    python -m bugspot video.mp4
    python -m bugspot video.mp4 --config detection_config.yaml
    bugspot video.mp4 --config detection_config.yaml
"""

import argparse
import json
import os
import sys

import yaml

from .pipeline import DetectionPipeline
from .detector import get_default_config


def _load_config(path: str) -> dict:
    """Load a YAML or JSON config file and merge with defaults."""
    defaults = get_default_config()
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            overrides = yaml.safe_load(f) or {}
        else:
            overrides = json.load(f)
    defaults.update(overrides)
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bugspot",
        description="BugSpot — lightweight insect detection and tracking.",
    )
    parser.add_argument(
        "video",
        help="Path to input video file.",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to YAML/JSON config file (default: built-in defaults).",
    )
    parser.add_argument(
        "-o", "--output",
        default="bugspot_output",
        help="Output directory for crops and composites (default: bugspot_output).",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Skip crop extraction.",
    )
    parser.add_argument(
        "--no-composites",
        action="store_true",
        help="Skip composite rendering.",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.config and not os.path.isfile(args.config):
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = _load_config(args.config) if args.config else get_default_config()

    # Set up output dirs
    crops_dir = None if args.no_crops else os.path.join(args.output, "crops")
    composites_dir = None if args.no_composites else os.path.join(args.output, "composites")

    # Run pipeline
    print(f"Processing: {args.video}")
    if args.config:
        print(f"Config:     {args.config}")
    print(f"Output:     {args.output}")
    print()

    pipeline = DetectionPipeline(config)
    result = pipeline.process_video(
        args.video,
        extract_crops=not args.no_crops,
        render_composites=not args.no_composites,
        save_crops_dir=crops_dir,
        save_composites_dir=composites_dir,
    )

    # Summary
    info = result.video_info
    print(f"Video:      {info['width']}x{info['height']} @ {info['fps']:.1f} fps, "
          f"{info['total_frames']} frames ({info['duration']:.1f}s)")
    print(f"Detections: {len(result.all_detections)}")
    print(f"Confirmed:  {len(result.confirmed_tracks)} tracks")
    print()

    for track_id, track in result.confirmed_tracks.items():
        print(f"  Track {track_id[:8]}  "
              f"{track.num_detections} detections, "
              f"{track.duration:.1f}s, "
              f"displacement={track.topology_metrics.get('net_displacement', 0):.0f}px")

    if not result.confirmed_tracks:
        print("  (no confirmed insect tracks)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

