from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing pipeline."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Original VisDrone Classes (YOLO format provided):
    # 0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
    # Target: 0: human, 1: car
    v1_class_mapping: Dict[int, int] = field(
        default_factory=lambda: {
            0: 0,  # pedestrian -> human
            1: 0,  # people -> human
            3: 1,  # car -> car
        }
    )

    target_class_names: List[str] = field(default_factory=lambda: ["human", "car"])
