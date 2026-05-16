# DroneCAM - Drone Human Detection & Counting System

This project is a computer vision pipeline built to detect humans and cars in high-resolution drone imagery, along with counting and tracking capabilities. It was developed as a technical submission for the Antlings Internship Program.

---

## Deliverables & Quick Links

* **Source Code:** Contained in this repository (`dronecam` package).
* **Demonstration Video:** [Drive Folder Link](https://drive.google.com/drive/folders/1pDpY7peyc-l4mw0VPhRvVkvk7t3b7Trz?usp=drive_link)
* **Output Visualizations:** Check the `outputs/` directory for processed images, tracking videos, and evaluation metrics.

---

## Installation & Setup

We use `uv` for Python package management.

```bash
# 1. Clone the repository
git clone https://github.com/kamrulhasan12345/dronecam.git
cd dronecam

# 2. Install dependencies and the package
uv sync

# Note: Ensure the dataset is placed inside data/raw/ before running commands
```

---

## Task Breakdown & Implementation

### Task-01: Dataset Understanding & Preprocessing
**Dataset Structure:** The dataset is based on high-altitude aerial drone imagery (VisDrone), divided into training, validation, and testing splits. The provided annotations were already in YOLO format (`[class_id, x_center, y_center, width, height]`).

**Preprocessing Steps:** 
* Filtered classes to strictly retain `human` (by merging `pedestrian` and `people` classes) and `car`.
* Ignored irrelevant classes (bicycles, tricycles, awning-tricycles, etc.).
* *Command:* `dronecam-preprocess --raw-dir data/raw --processed-dir data/processed`

**Challenges:** The massive scale variance. Humans and cars occupy a very tiny fraction of the 4K/1080p frames. Resizing these images down to standard resolutions (e.g., 640x640) obliterates feature details for these small objects.

*(See `outputs/samples/` for preprocessed dataset visualizations).*

### Task-02: Model Training
**Training Approach:** I chose the Ultralytics YOLO architecture for its balance of speed and accuracy. To combat the small object challenge, I experimented with higher resolutions (`imgsz=1280`) and utilized Mosaic augmentation to improve robustness against scale variations. 

*Command:* `dronecam-train --epochs 50 --imgsz 1280 --batch 16 --output-name yolov8n_imgsz1280_ep50.pt`

*(See `outputs/metrics/` for training curves, PR curves, and confusion matrices).*

### Task-03: Human & Car Detection with Human Counting
**Implementation:** The inference pipeline utilizes SAHI (Slicing Aided Hyper Inference) to dynamically slice high-resolution images into smaller grids (e.g., 512x512) during testing. This preserves pixel density and allows the model to detect tiny humans that a standard pass would miss. The script draws bounding boxes for cars and humans, and displays the total human count directly on the image.

*Command:* `dronecam-infer --image data/raw/test/images/0000.jpg --model models/yolov8n_imgsz1280_ep50.pt`

*(See `outputs/predictions/` for sample output images with bounding boxes and counting logic applied).*

### Task-04: Object Tracking (Bonus)
**Implementation:** Implemented tracking to maintain consistent IDs across video frames. Two methods are provided:
1. **SAHI + ByteTrack:** For maximum accuracy on tiny, distant objects.
2. **YOLO + BoT-SORT:** A native, faster approach for real-time tracking.

*Command:* 
```bash
dronecam-track --video data/raw/video.mp4 --model models/yolov8n_imgsz1280_ep50.pt --method sahi
```

*(See `outputs/tracking/` for the exported `.mp4` tracking results).*

### Task-05: Evaluation & Visualization
**Metrics:** Baseline test runs on the validation split yielded the following results for the target classes:
* **mAP@0.5:** ~0.728
* **mAP@0.5:0.95:** ~0.436
* **Precision:** ~0.771
* **Recall:** ~0.660

**Strengths:**
* Structured as a modular, reproducible Python package rather than disconnected notebooks.
* Utilizing SAHI combined with tracking algorithms (ByteTrack) significantly improves the detection capabilities for drone-captured footage.
* Dynamic overlay logic handles dense crowds and visualization efficiently.

**Limitations & Challenges Faced:**
* **Computational Cost:** Processing high-resolution video using SAHI + ByteTrack is heavy. It typically runs below real-time FPS on standard hardware. I mitigated this by providing standard YOLO tracking (`--method yolo`) as a faster fallback.
* **Extreme Crowds:** Highly overlapping individuals in dense crowds occasionally lead to merged bounding boxes, slightly undercounting the exact number of humans.

---

*For task-related queries and code walkthrough, please refer to the attached Demonstration Video.*
