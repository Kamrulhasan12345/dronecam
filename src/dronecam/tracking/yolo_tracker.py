import cv2
from pathlib import Path
from ultralytics import YOLO

class YoloTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def track_video(self, video_path: str, output_path: str) -> None:
        """Run standard YOLO native tracking (BoT-SORT) on a video and save output."""
        print(f"Starting standard YOLO tracking on {video_path}")
        
        # model.track returns a generator, saving is built-in if we let ultralytics handle it,
        # but to have explicit control over output paths we do it frame by frame or use their save logic.
        # Ultralytics natively saves to runs/detect/track, but we want it in outputs/videos.
        
        results = self.model.track(
            source=video_path,
            persist=True,
            tracker="botsort.yaml",
            stream=True,
            verbose=False
        )
        
        # Get video properties for writer
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_idx, r in enumerate(results):
            # r.plot() returns the BGR image frame with boxes and track IDs drawn natively by YOLO
            annotated_frame = r.plot()
            
            # Count humans (class 0 is human based on our config)
            # r.boxes.cls contains the detected class ids
            human_count = 0
            if r.boxes is not None and r.boxes.cls is not None:
                human_count = (r.boxes.cls == 0).sum().item()
                
            # Add Human Count Text Overlay
            cv2.putText(
                annotated_frame, 
                f"Total Human Count: {human_count}", 
                (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 0, 255), 
                3
            )
            
            out.write(annotated_frame)
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}...")
                
        out.release()
        print(f"Video saved to {output_path}")
