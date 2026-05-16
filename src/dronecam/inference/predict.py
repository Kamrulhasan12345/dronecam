import cv2
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

class SahiPredictor:
    def __init__(self, model_path: str, model_type: str = "ultralytics", device: str = "cuda:0"):
        """Initialize SAHI Predictor for high-res drone images."""
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=0.25,
            device=device
        )
        
    def predict_image(self, image_path: str, slice_height: int = 512, slice_width: int = 512, overlap: float = 0.2):
        """Run sliced prediction on a single image."""
        img = read_image(image_path)
        
        # Sliced inference
        result = get_sliced_prediction(
            img,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap
        )
        return result

    def save_prediction(self, result, output_path: str):
        """Export visualizing result to a file with human counting."""
        import cv2
        import numpy as np
        
        # Get count of humans (class 0)
        human_count = sum(1 for obj in result.object_prediction_list if obj.category.id == 0)
        
        # We can export visuals directly, but to add text overlay cleanly, we annotate manually or overlay after export
        result.export_visuals(export_dir=str(Path(output_path).parent), file_name=Path(output_path).stem)
        
        # Read the exported image, add the text, and save it back
        img_path = str(Path(output_path).parent / f"{Path(output_path).stem}.png")
        img = cv2.imread(img_path)
        if img is not None:
            cv2.putText(
                img, 
                f"Total Human Count: {human_count}", 
                (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 0, 255), 
                3
            )
            cv2.imwrite(img_path, img)
