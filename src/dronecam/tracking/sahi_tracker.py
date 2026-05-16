import sys

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    import supervision as sv
except ImportError:
    print("Please install supervision: pip install supervision")
    sys.exit(1)

from dronecam.inference.predict import SahiPredictor
from sahi.predict import get_sliced_prediction


class SahiTracker:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """Initialize SAHI Predictor and Supervision ByteTrack."""
        self.predictor = SahiPredictor(
            model_path=model_path, model_type="ultralytics", device=device
        )
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def track_video(
        self,
        video_path: str,
        output_path: str,
        slice_size: int = 512,
        overlap: float = 0.2,
    ) -> None:
        """Run SAHI for detection and Supervision ByteTrack for tracking on high-res drone video."""
        print(f"Starting SAHI + ByteTrack tracking on {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (cv2) to RGB for SAHI prediction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Run SAHI Prediction
            result = get_sliced_prediction(
                rgb_frame,
                self.predictor.detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )

            # Extract boxes, confidences, and classes into Supervision format
            boxes = []
            confidences = []
            class_ids = []
            for obj in result.object_prediction_list:
                # [xmin, ymin, xmax, ymax]
                boxes.append(
                    [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
                )
                confidences.append(obj.score.value)
                class_ids.append(obj.category.id)

            human_count = 0
            # Cast to uint8 explicitly: cv2 stubs type cap.read() frames as
            # ndarray[integer | floating], but BoxAnnotator.annotate requires uint8.
            annotated_frame: np.ndarray = np.asarray(frame, dtype=np.uint8).copy()

            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids),
                )

                # 2. Update Tracker
                detections = self.tracker.update_with_detections(detections)

                # Narrow Optional fields — ByteTrack always populates these after update,
                # but supervision's stubs type them as ndarray | None.
                assert detections.class_id is not None
                assert detections.tracker_id is not None
                assert detections.confidence is not None

                # category_mapping is typed as dict | None; assert it is populated.
                category_mapping = self.predictor.detection_model.category_mapping
                assert category_mapping is not None

                # Count humans (class_id == 0)
                human_count = int(np.sum(detections.class_id == 0))

                # 3. Annotate Frame (Adding Human ID / Tracker ID)
                labels = [
                    f"ID:{tracker_id if tracker_id is not None else '?'} "
                    f"{category_mapping[str(class_id)]} "
                    f"{conf:.2f}"
                    for class_id, tracker_id, conf in zip(
                        detections.class_id,
                        detections.tracker_id,
                        detections.confidence,
                    )
                ]

                # BoxAnnotator preserves input type (ndarray in → ndarray out).
                # LabelAnnotator.annotate types scene as PIL Image specifically,
                # so convert explicitly between the two calls.
                _box_scene = self.box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                _pil_scene = PILImage.fromarray(_box_scene)
                _label_scene = self.label_annotator.annotate(
                    scene=_pil_scene, detections=detections, labels=labels
                )
                annotated_frame = np.asarray(_label_scene, dtype=np.uint8).copy()

            # annotate() returns a generic ImageType; cast back to uint8 ndarray so
            # cv2.putText and VideoWriter.write see the concrete type they expect.
            out_frame: np.ndarray = np.asarray(annotated_frame, dtype=np.uint8).copy()

            # Add Human Count Text Overlay
            cv2.putText(
                out_frame,
                f"Total Human Count: {human_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

            out.write(out_frame)

            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}...")
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Video saved to {output_path}")
