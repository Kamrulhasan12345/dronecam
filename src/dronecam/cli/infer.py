#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dronecam.inference.predict import SahiPredictor


def main():
    parser = argparse.ArgumentParser(description="Run SAHI Inference on images.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions/result.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g. cpu, cuda:0)",
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

    print(f"Loading model from {args.model}...")
    predictor = SahiPredictor(
        model_path=args.model, model_type="ultralytics", device=args.device
    )

    print(f"Running sliced prediction on {args.image}...")
    result = predictor.predict_image(
        args.image, slice_height=args.slice_size, slice_width=args.slice_size
    )

    predictor.save_prediction(result, str(output_path))
    print(f"Prediction saved to {output_path.parent / output_path.stem}.png")


if __name__ == "__main__":
    main()
