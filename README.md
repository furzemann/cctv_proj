# Advanced Person Tracking with YOLOv8, Part-Based Re-ID, and Kalman Filtering

An integrated, high-performance system for robust, real-time person tracking in complex surveillance scenarios. This project combines state-of-the-art deep learning models and classical tracking algorithms to deliver superior accuracy and minimize identity switches.


---

## 📋 Table of Contents
- [Overview](#-overview)
- [Core Features](#-core-features)
- [System Pipeline](#-system-pipeline)
- [Hardware Requirements & Performance](#-hardware-requirements--performance)
- [Setup & Installation](#-setup--installation)
- [Execution](#-execution)
- [Configuration & Tuning](#-configuration--tuning)
- [Project Outputs](#-project-outputs)

---

## 📝 Overview

Standard object trackers often fail in real-world scenarios due to frequent occlusions, erratic motion, and changes in appearance, leading to fragmented tracks and identity switches. This project implements a sophisticated tracking pipeline designed to overcome these challenges.

By integrating *Ultralytics YOLOv8* for rapid detection, a *Part-Based ResNet50* model for robust visual identification, a *Kalman Filter* for smooth motion prediction, and the *Hungarian Algorithm* for optimal data association, our system provides a highly effective and scalable solution for real-world person tracking. Advanced logic such as adaptive smoothing, safe re-attachment gating, and fallback IoU matching is incorporated to ensure tracking continuity and reliability.

---

## ✨ Core Features

* *⚡ High-Speed Detection: Leverages the state-of-the-art **Ultralytics YOLOv8* model for an exceptional balance of speed and accuracy, ideal for real-time analysis.
* *🛡 Robust Re-Identification (Re-ID): Creates a powerful visual signature by extracting features from **3 horizontal body parts* (head, torso, legs) using a ResNet50 model. This part-based approach is more resilient to partial occlusions and appearance changes than global feature methods.
* *📈 Smooth Motion Prediction: Employs a **Kalman Filter* for each track to model its motion state, predict future positions, and seamlessly bridge temporary detection gaps.
* *🤝 Optimal Data Association: Utilizes the **Hungarian Algorithm* to globally minimize assignment cost (a weighted sum of visual and spatial distance), preventing sub-optimal local matches and ensuring logical track-to-detection pairing.
* *🧠 Advanced Heuristics*:
    * *Safe Re-attachment Gating*: Implements strict criteria (high visual similarity, spatial proximity, and recentness) to prevent incorrect track re-identifications after a person is lost.
    * *Fallback IoU Matching*: Prioritizes spatial continuity by re-attaching tracks based on spatial overlap for short-term misses, effectively preventing the creation of duplicate IDs.
    * *Adaptive Bounding Box Smoothing*: Dynamically adjusts the smoothing factor based on object velocity to provide stable, jitter-free boxes for slow-moving targets and responsive boxes for fast-moving ones.
* *📊 Actionable Outputs*: Generates both a processed video with annotated tracks and a detailed, frame-by-frame text log for comprehensive post-analysis.

---

## ⚙ System Pipeline

Our system processes video through a multi-stage pipeline designed for maximum accuracy and resilience.


1.  *Video Input*: The system receives a live or pre-recorded video stream.
2.  *YOLO Detection*: An Ultralytics YOLO model scans each frame to identify all persons, generating bounding boxes and confidence scores. Detections with confidence above a set threshold are passed to the next stage.
3.  *Part-Based Feature Extraction*: For each valid detection, a ResNet50 model extracts unique visual features from the head, torso, and legs to create a robust, occlusion-resistant appearance signature.
4.  *Motion Prediction*: A Kalman Filter assigned to each existing track predicts its next location based on its current motion state (position and velocity).
5.  *Association & Matching*: A cost matrix is constructed based on a weighted sum of visual similarity (Cosine Distance) and spatial proximity (IoU) between predicted tracks and new detections. The Hungarian Algorithm solves this matrix to find the optimal assignments.
6.  *State Update & Refinement*: Matched tracks are updated with new detection data, and their bounding boxes are smoothed. Unmatched detections may initiate new tracks.
7.  *Output & Logging*: The final output is rendered, including a processed video with annotated tracks and a detailed text log file for analysis.

---

## 💻 Hardware Requirements & Performance

For optimal performance, specific hardware is recommended. The tracker's computational speed (FPS) is highly dependent on the system's hardware, the video resolution, and the density of objects being tracked.

### Hardware
* *GPU: An NVIDIA CUDA-enabled GPU is **highly recommended* for real-time performance.
    * *Recommended*: NVIDIA GeForce RTX 3060 (8GB VRAM) or higher.
    * *Minimum*: NVIDIA GeForce RTX 2060 (6GB VRAM).
    * The system can run on a *CPU*, but performance will be significantly slower (typically < 5 FPS) and is not suitable for real-time applications.
* *System RAM*: 16 GB or more.
* *Storage*: SSD for faster video I/O.

### Performance Estimation
This system does not require training; it performs inference using pre-trained models. The processing time is therefore the time it takes to run the pipeline on each frame of the video.

To estimate the total time required to process a video, you can use the following formula:

$$
\text{Total Processing Time (seconds)} = \frac{\text{Total Video Frames}}{\text{Inference FPS}}
$$

*Example Calculation:*
* *Video*: 5 minutes long at 30 frames per second.
* *Total Frames*: $5 \text{ minutes} \times 60 \text{ seconds/minute} \times 30 \text{ frames/second} = 9000 \text{ frames}$.
* *Your System's FPS*: Let's assume your GPU achieves an average of 45 FPS during processing.
* *Estimated Time*: $ \frac{9000 \text{ frames}}{45 \text{ FPS}} = 200 \text{ seconds} \approx 3.3 \text{ minutes}$.

You can find your system's inference FPS from the console output while the script is running.

---

## 🚀 Setup & Installation

Follow these steps to set up the project environment.

### Prerequisites
* Python 3.8+
* pip and git
* *For GPU acceleration*: A compatible NVIDIA GPU with the latest drivers and CUDA Toolkit installed.

### Installation Guide

*1. Clone the Repository*
```bash
git clone [https://your-repository-url.git](https://your-repository-url.git)
cd your-repository-directory
2. Create and Activate a Virtual Environment (Recommended)

On macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
On Windows:

Bash

python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies
This project relies on several key libraries. Create a requirements.txt file with the content below and then run pip install -r requirements.txt.

# requirements.txt
torch
torchvision
ultralytics
numpy
opencv-python
scipy
Note on PyTorch: For GPU support, ensure you install the correct version of PyTorch that corresponds to your CUDA version. It is highly recommended to follow the official instructions at the PyTorch website.

4. Prepare Input Video
Place your input video file (e.g., in .mp4 format) in the root of the project directory. The script is configured by default to look for a file named input_video.mp4.

▶ Execution
Once the setup is complete, you can run the tracker with a single command.

Bash

python improved_person_reid_tracker_v2.py
The script will:

Initialize the YOLO and Re-ID models.

Open the input video file.

Process the video frame by frame, displaying the output with tracking annotations in a window titled "PersonReID_v2".

Save the processed video and the tracking log to the files specified in the configuration.

Print progress to the console. You can stop the process at any time by pressing 'q' in the display window.

🔧 Configuration & Tuning
The tracker's behavior can be fine-tuned by adjusting the parameters in the USER TUNABLE SETTINGS section at the top of the Python script.

Parameter	Default	Description
HIGH_CONF_THR	0.45	The minimum YOLO confidence score required for a detection to be processed for tracking.
IOU_THR	0.30	The minimum Intersection over Union (IoU) required for a primary spatial match.
MATCH_WEIGHT_IOU	0.4	The weight given to spatial distance (IoU) in the final matching cost matrix.
MATCH_WEIGHT_REID	0.6	The weight given to visual similarity (Re-ID) in the final matching cost matrix.
REID_STRICT_THRESH	0.45	A strict visual similarity threshold used for confidently re-attaching a 'Lost' track.
MAX_AGE	150	The maximum number of consecutive frames a track can remain 'Lost' before it is pruned.
MIN_HITS_TO_CONFIRM	2	The number of successful updates required before a new track is considered "Confirmed".
REID_REATTACH_MAX_AGE	30	Re-ID based re-attachment is only attempted for tracks lost within the last 30 frames.
FALLBACK_IOU_FOR_NEW	0.18	If an unmatched detection has an IoU > this value with a lost track, it will be attached to prevent duplicates.

Export to Sheets
📤 Project Outputs
The system generates two critical outputs for analysis and review.

1. Processed Video (output_person_reid_tracking_v2.mp4)
An .mp4 video file showing the original footage overlaid with tracking information.

Confirmed tracks are displayed with a unique, color-coded, and smoothed bounding box with a track ID.

2. Detailed Tracking Log (tracking_person_reid_v2.txt)
A human-readable text file that provides a complete, frame-by-frame history of all confirmed tracks, perfect for quantitative analysis.

Log Format: F<frame_num>,ID<track_id>,Status,x1,y1,x2,y2

Example Entry: F000123,ID007,Tracked,105,207,159,355