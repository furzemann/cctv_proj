# Human Tracking with YOLO11 and Re-ID

This project provides a robust solution for tracking humans in video streams using YOLO11 object detection combined with Person Re-Identification (Re-ID) for maintaining consistent IDs across occlusions and re-entries.

## 🚀 Features

- **YOLO11 Detection**: Real-time person detection using the latest YOLO11 model
- **ByteTrack Logic**: Efficient multi-object tracking with IoU-based matching
- **Person Re-Identification**: Feature-based re-identification after track loss
- **Detailed Logging**: Comprehensive tracking logs with timestamps and events
- **Video Processing**: Supports various video formats with visualization
- **Dataset Training**: Tools for training custom YOLO models on CCTV data

## 📦 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cctv-tracking.git
cd cctv-tracking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup FastReID
```bash
git clone https://github.com/JDAI-CV/fast-reid.git
```

### 4. Download Models
- YOLO11 will auto-download on first run
- Download Re-ID model weights: [market_bot_R50.pth](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- Place `.pth` file in `models/` directory

### 5. Run Tracking
```bash
# Place your video as input_video.mp4
python cctv.py
```

## 🎯 Usage

### Basic Tracking
```python
python cctv.py  # Uses input_video.mp4 or webcam
```

### Process Dataset for Training
```python
python process_dataset.py  # Converts videos to YOLO format
```

### Train Custom YOLO Model
```python
python train_yolo.py  # Trains on processed dataset
```

## 📁 Project Structure

```
├── cctv.py                 # Main tracking script
├── process_dataset.py      # Dataset processing for training
├── train_yolo.py          # YOLO model training
├── models/                # Model weights (.pth files)
├── fast-reid/             # FastReID library (clone)
├── examples/              # Example outputs and logs
└── README.md
```

## ⚙️ Configuration

Key parameters in `cctv.py`:
- `high_conf_thresh`: High confidence detection threshold (default: 0.3)
- `low_conf_thresh`: Low confidence threshold for ByteTrack (default: 0.1)
- `iou_thresh`: IoU threshold for track matching (default: 0.4)
- `reid_thresh`: Re-ID similarity threshold (default: 0.7)
- `max_age`: Maximum frames to keep lost tracks (default: 240)

## 🎨 Visualization

- 🟢 **Green**: New person detected
- 🟣 **Magenta**: Re-identified person
- 🟠 **Orange**: Person in occlusion
- 🔴 **Red**: Lost track

## 📊 Output Files

- `output_bytetrack_reid.mp4`: Processed video with tracking
- `tracking_log.txt`: Detailed tracking events log
- `activity_records.json`: Summary statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)