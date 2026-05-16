#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dronecam.tracking.yolo_tracker import YoloTracker
from dronecam.tracking.sahi_tracker import SahiTracker


def main():
    parser = argparse.ArgumentParser(description="Run Video Object Tracking.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/tracked_output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["yolo", "sahi"],
        default="yolo",
        help="Tracking method to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g. cpu, cuda:0) - only for SAHI method",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=512,
        help="Slice size for SAHI (width and height)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "yolo":
        tracker = YoloTracker(model_path=args.model)
        tracker.track_video(video_path=args.video, output_path=str(output_path))
    elif args.method == "sahi":
        tracker = SahiTracker(model_path=args.model, device=args.device)
        tracker.track_video(
            video_path=args.video,
            output_path=str(output_path),
            slice_size=args.slice_size,
        )


if __name__ == "__main__":
    main()
