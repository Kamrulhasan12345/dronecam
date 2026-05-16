#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dronecam.training.trainer import YoloTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model on processed dataset."
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="data/processed/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="yolo26n.pt",
        help="YOLO base model (e.g., yolo26n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for training"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (e.g., '0' or '0,1' or 'cpu')",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="best_model.pt",
        help="Name to save the trained model in models/",
    )

    args = parser.parse_args()

    trainer = YoloTrainer(data_yaml=args.data_yaml, model_version=args.model_version)
    best_weights = trainer.train(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device
    )

    trainer.save_artifacts(
        best_model_path=best_weights, output_model_name=args.output_name
    )


if __name__ == "__main__":
    main()
