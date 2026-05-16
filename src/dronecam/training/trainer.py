import shutil
from pathlib import Path

from ultralytics import YOLO


class YoloTrainer:
    def __init__(self, data_yaml: str, model_version: str = "yolo26n.pt"):
        self.data_yaml = Path(data_yaml)
        self.model_version = model_version
        self.model = YOLO(model_version)

    def train(
        self, epochs: int = 50, imgsz: int = 640, batch: int = 16, device: str = ""
    ) -> str:
        """Trains the YOLO model and returns the path to the best weights."""
        print(f"Starting training for {epochs} epochs with image size {imgsz}...")

        self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            optimizer="SGD",
            project="runs/detect",
            name="train",
            exist_ok=False,
        )

        # model.trainer is populated after training but typed as None | Unknown;
        # assert it is set before accessing .best.
        if self.model.trainer is None:
            raise RuntimeError(
                "model.trainer is None after training — training may have failed."
            )

        best_model_path = str(self.model.trainer.best)
        print(f"Training complete. Best model saved at: {best_model_path}")
        return best_model_path

    def save_artifacts(
        self,
        best_model_path: str,
        output_model_name: str,
        models_dir: str = "models",
        metrics_dir: str = "outputs/metrics",
    ) -> None:
        """Copies the best model and metrics to the project's output directories."""
        models_path = Path(models_dir)
        metrics_path = Path(metrics_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)

        # Save model
        target_model_path = models_path / output_model_name
        shutil.copy(best_model_path, target_model_path)
        print(f"Model saved to {target_model_path}")

        # Save metrics (copying from runs folder)
        run_dir = Path(best_model_path).parent.parent
        for metric_file in [
            "results.png",
            "confusion_matrix.png",
            "BoxPR_curve.png",
            "results.csv",
        ]:
            src = run_dir / metric_file
            if src.exists():
                shutil.copy(src, metrics_path / metric_file)

        print(f"Metrics saved to {metrics_path}")

