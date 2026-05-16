import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List

from .config import PreprocessConfig


class VisDroneYoloPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.raw_dir = Path(config.raw_data_dir)
        self.processed_dir = Path(config.processed_data_dir)

    def setup_directories(self) -> None:
        """Create YOLO format directory structure."""
        for split in ["train", "val", "test"]:
            (self.processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(self, split_name: str, split_dir_name: str) -> None:
        """Process a single data split (train, val, test)."""
        images_dir = self.raw_dir / split_dir_name / "images"
        labels_dir = self.raw_dir / split_dir_name / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(
                f"Warning: Raw data directories for {split_name} not found. Skipping."
            )
            return

        label_files = list(labels_dir.glob("*.txt"))

        for ann_path in tqdm(label_files, desc=f"Processing {split_name}"):
            img_path = images_dir / f"{ann_path.stem}.jpg"

            if not img_path.exists():
                continue

            # Read existing YOLO format: class x_center y_center width height
            try:
                df = pd.read_csv(
                    ann_path,
                    sep=" ",
                    header=None,
                    names=["class", "x_center", "y_center", "width", "height"],
                )
            except Exception:
                continue

            yolo_labels: List[str] = []

            for _, row in df.iterrows():
                orig_class = int(row["class"])

                # Check if we care about this class
                if orig_class in self.config.v1_class_mapping:
                    target_class = self.config.v1_class_mapping[orig_class]
                    x_c, y_c = row["x_center"], row["y_center"]
                    w, h = row["width"], row["height"]

                    # Already normalized in YOLO format
                    if w > 0 and h > 0:
                        yolo_labels.append(
                            f"{target_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                        )
                        
            # Remove exact duplicate bounding boxes
            yolo_labels = list(set(yolo_labels))

            # Save if there are valid labels
            if yolo_labels:
                # Copy image
                target_img_path = (
                    self.processed_dir / "images" / split_name / img_path.name
                )
                shutil.copy(img_path, target_img_path)

                # Save label
                target_label_path = (
                    self.processed_dir / "labels" / split_name / f"{img_path.stem}.txt"
                )
                with open(target_label_path, "w") as f:
                    f.write("\n".join(yolo_labels))

    def generate_yaml(self) -> None:
        """Generate dataset.yaml for YOLO training."""
        # Using absolute paths for all splits to ensure robust cross-platform/environment compatibility
        yaml_content = f"""# Auto-generated YOLO dataset configuration
path: {self.processed_dir.absolute()}
train: {(self.processed_dir / 'images' / 'train').absolute()}
val: {(self.processed_dir / 'images' / 'val').absolute()}
test: {(self.processed_dir / 'images' / 'test').absolute()}

nc: {len(self.config.target_class_names)}
names:
"""
        for i, name in enumerate(self.config.target_class_names):
            yaml_content += f"  {i}: {name}\n"

        yaml_path = self.processed_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        print(f"Generated {yaml_path}")

    def run(self) -> None:
        """Run the full preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        self.setup_directories()

        # Mapping standard dataset names to your specific raw directory names if needed
        # We assume basic names, adjust if your raw folder has specific names like VisDrone2019-DET-train
        self.process_split("train", "VisDrone2019-DET-train")
        self.process_split("val", "VisDrone2019-DET-val")
        self.process_split("test", "VisDrone2019-DET-test-dev")

        self.generate_yaml()
        print("Preprocessing completed!")
