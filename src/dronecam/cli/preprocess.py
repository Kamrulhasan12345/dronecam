#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dronecam.preprocessing import PreprocessConfig, VisDroneYoloPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VisDrone dataset to YOLO format."
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw", help="Path to raw VisDrone dataset"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to output processed data",
    )

    args = parser.parse_args()

    config = PreprocessConfig(
        raw_data_dir=args.raw_dir, processed_data_dir=args.processed_dir
    )

    preprocessor = VisDroneYoloPreprocessor(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
