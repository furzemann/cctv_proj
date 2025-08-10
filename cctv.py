import sys
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime, timedelta

fast_reid_path = os.path.abspath('fast-reid')
if fast_reid_path not in sys.path:
    sys.path.append(fast_reid_path)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from PIL import Image
import torchvision.transforms as T

input_video = "input_video.mp4"
output_file = "output_bytetrack_reid.mp4"
record_file = "activity_records.json"
tracking_log_file = "tracking_log.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"

high_conf_thresh = 0.3
low_conf_thresh = 0.2
iou_thresh = 0.1
reid_thresh = 0.87
max_age = 600

reid_config_path = r"fast-reid/configs/Market1501/bagtricks_R50.yml"
reid_model_weights = r"models/market_bot_R50.pth"

def setup_fastreid_predictor(config_file, weights_file, device):
    try:
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        return predictor
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a necessary model file.")
        print(f"Details: {e}")
        print("Please ensure the paths for reid_config_path and reid_model_weights are correct.")
        sys.exit(1)

def calc_iou(boxa, boxb):
    x1, y1, x2, y2 = boxa
    xa, ya, xb, yb = boxb
    xx1, yy1 = max(x1, xa), max(y1, ya)
    xx2, yy2 = min(x2, xb), min(y2, yb)
    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (xb - xa) * (yb - ya)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0


class TrackingLogger:
    def __init__(self, log_file, video_start_time=None):
        self.log_file = log_file
        self.video_start_time = video_start_time or datetime.now()
        self.track_events = {}
        
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PERSON TRACKING LOG\n")
            f.write(f"Video Start Time: {self.video_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write("Format: [TIMESTAMP] FRAME_NUM | ID_XXX | STATUS | DETAILS\n")
            f.write("-" * 80 + "\n\n")
    
    def log_event(self, frame_num, frame_time, track_id, status, details="", box=None):
        timestamp = self.video_start_time + timedelta(seconds=frame_time)
        
        box_str = ""
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            box_str = f" at ({x1},{y1})-({x2},{y2})"
        
        log_entry = f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] F{frame_num:06d} | ID_{track_id:03d} | {status:12s} | {details}{box_str}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
        
        if track_id not in self.track_events:
            self.track_events[track_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'total_frames': 0,
                'events': [],
                'status_counts': {}
            }
        
        track_data = self.track_events[track_id]
        track_data['last_seen'] = timestamp
        track_data['total_frames'] += 1
        track_data['events'].append((frame_num, status, details))
        track_data['status_counts'][status] = track_data['status_counts'].get(status, 0) + 1
    
    def write_summary(self, total_frames, fps):
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRACKING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(f"Video FPS: {fps}\n")
            f.write(f"Total Duration: {total_frames/fps:.1f} seconds\n")
            f.write(f"Unique People Detected: {len(self.track_events)}\n")
            f.write("-" * 80 + "\n")
            
            for track_id, data in sorted(self.track_events.items()):
                duration = (data['last_seen'] - data['first_seen']).total_seconds()
                f.write(f"\nID_{track_id:03d}:\n")
                f.write(f"  First Seen: {data['first_seen'].strftime('%H:%M:%S')}\n")
                f.write(f"  Last Seen:  {data['last_seen'].strftime('%H:%M:%S')}\n")
                f.write(f"  Duration:   {duration:.1f} seconds\n")
                f.write(f"  Frames:     {data['total_frames']}\n")
                
                f.write(f"  Status breakdown:\n")
                for status, count in data['status_counts'].items():
                    f.write(f"    {status}: {count} frames\n")


class MyTracker:
    def __init__(self, reid_predictor, iou_thresh=0.5, reid_thresh=0.5, max_age=30, logger=None):
        self.reid_predictor = reid_predictor
        self.iou_thresh = iou_thresh
        self.reid_thresh = reid_thresh
        self.max_age = max_age
        self.memory = {}
        self.lost_embeddings = {}
        self.next_id = 1
        self.logger = logger
        print("Tracker initialized with ByteTrack + Re-ID logic.")

    def get_feat(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        
        h, w = frame.shape[:2]
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(crop_rgb)
            
            pil_img = pil_img.resize((128, 256))
            
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(pil_img).unsqueeze(0)
            
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                features = self.reid_predictor(input_tensor)
            
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            elif isinstance(features, (list, tuple)):
                features = features[0].cpu().numpy() if hasattr(features[0], 'cpu') else features[0]
            
            if features.size > 0:
                features = features / (np.linalg.norm(features) + 1e-8)
                return features
            return None
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def update(self, high_conf_boxes, low_conf_boxes, frame, frame_num=0, frame_time=0):
        high_conf_features = [self.get_feat(frame, box) for box in high_conf_boxes]
        
        detections = []
        for i, feat in enumerate(high_conf_features):
            if feat is not None:
                detections.append({'box': high_conf_boxes[i], 'features': feat, 'score': 1.0})

        active_tracks = [t for t in self.memory.values() if t['age'] == 0]
        lost_tracks = [t for t in self.memory.values() if t['age'] > 0]
        
        matched_tracks = []
        unmatched_detections = []
        unmatched_tracks = []

        if detections and active_tracks:
            iou_matrix = np.zeros((len(active_tracks), len(detections)))
            for t_idx, track in enumerate(active_tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = calc_iou(track['box'], det['box'])

            used_detections = set()
            for t_idx, track in enumerate(active_tracks):
                best_iou = 0
                best_d_idx = -1
                for d_idx in range(len(detections)):
                    if d_idx not in used_detections and iou_matrix[t_idx, d_idx] > best_iou:
                        best_iou = iou_matrix[t_idx, d_idx]
                        best_d_idx = d_idx
                
                if best_iou >= self.iou_thresh:
                    det = detections[best_d_idx]
                    track['box'] = det['box']
                    track['features'] = det['features']
                    track['age'] = 0
                    track['lev'] = "Tracked"
                    matched_tracks.append(track)
                    used_detections.add(best_d_idx)
                    
                    if self.logger:
                        self.logger.log_event(frame_num, frame_time, track['tid'], "Tracked", 
                                                f"IoU: {best_iou:.3f}", track['box'])
                else:
                    unmatched_tracks.append(track)
            
            unmatched_detections = [detections[i] for i in range(len(detections)) if i not in used_detections]
        else:
            unmatched_tracks.extend(active_tracks)
            unmatched_detections = detections
        
        still_lost_tracks = []
        
        if len(low_conf_boxes) > 0 and lost_tracks:
            low_conf_list = list(low_conf_boxes)
            used_low_conf = set()
            
            for track in lost_tracks:
                best_iou = 0
                best_idx = -1
                for lc_idx, lc_box in enumerate(low_conf_list):
                    if lc_idx not in used_low_conf:
                        iou = calc_iou(track['box'], lc_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = lc_idx
                
                if best_iou >= self.iou_thresh and best_idx != -1:
                    track['box'] = low_conf_list[best_idx]
                    track['age'] = 0
                    track['lev'] = "Occluded"
                    matched_tracks.append(track)
                    used_low_conf.add(best_idx)
                    
                    if self.logger:
                        self.logger.log_event(frame_num, frame_time, track['tid'], "Occluded", 
                                                f"Recovered via low-conf detection, IoU: {best_iou:.3f}", track['box'])
                else:
                    still_lost_tracks.append(track)
        else:
            still_lost_tracks.extend(lost_tracks)
            
        reid_matched_dets = []
        for d_idx, det in enumerate(unmatched_detections):
            best_dist = 1.0
            best_lost_id = -1
            for lost_id, lost_feat in self.lost_embeddings.items():
                dist = 1 - np.dot(lost_feat.flatten(), det['features'].flatten())
                if dist < self.reid_thresh and dist < best_dist:
                    best_dist = dist
                    best_lost_id = lost_id
            
            if best_lost_id != -1:
                det['tid'] = best_lost_id
                det['age'] = 0
                det['lev'] = f"Re-ID (dist: {best_dist:.3f})"
                matched_tracks.append(det)
                del self.lost_embeddings[best_lost_id]
                reid_matched_dets.append(d_idx)
                
                if self.logger:
                    self.logger.log_event(frame_num, frame_time, best_lost_id, "Re-ID", 
                                                f"Feature distance: {best_dist:.3f}", det['box'])

        unmatched_detections = [d for i, d in enumerate(unmatched_detections) if i not in reid_matched_dets]

        for det in unmatched_detections:
            det['tid'] = self.next_id
            det['age'] = 0
            det['lev'] = "New"
            matched_tracks.append(det)
            
            if self.logger:
                self.logger.log_event(frame_num, frame_time, self.next_id, "New", 
                                        "First detection", det['box'])
            
            self.next_id += 1

        self.memory = {t['tid']: t for t in matched_tracks}
        gone_ids = []
        for track in still_lost_tracks:
            track['age'] += 1
            if track['age'] > self.max_age:
                gone_ids.append(track['tid'])
                if track.get('features') is not None:
                    self.lost_embeddings[track['tid']] = track['features']
                
                if self.logger:
                    self.logger.log_event(frame_num, frame_time, track['tid'], "Removed", 
                                                f"Lost for {self.max_age} frames")
            else:
                track['lev'] = 'Lost'
                self.memory[track['tid']] = track
                
                if track['age'] == 1 and self.logger:
                    self.logger.log_event(frame_num, frame_time, track['tid'], "Lost", 
                                                "Track lost")

        return self.memory


def main():
    print(f"Initializing models on device: {device}...")
    
    print("📦 Loading YOLO11 model...")
    yolo_model = YOLO("yolo11n.pt")
    
    if device == "cuda":
        yolo_model.to(device)
    
    print(f"✅ YOLO11 model loaded on {device}")
    
    if not os.path.exists(reid_config_path):
        print(f"ERROR: Config file not found: {reid_config_path}")
        print("Please download the FastReID model files.")
        return
    
    if not os.path.exists(reid_model_weights):
        print(f"ERROR: Model weights not found: {reid_model_weights}")
        print("Please download the FastReID model weights.")
        return
    
    reid_predictor = setup_fastreid_predictor(reid_config_path, reid_model_weights, device)

    video_start_time = datetime.now()
    logger = TrackingLogger(tracking_log_file, video_start_time)

    tracker = MyTracker(
        reid_predictor=reid_predictor,
        iou_thresh=iou_thresh,
        reid_thresh=reid_thresh,
        max_age=max_age,
        logger=logger
    )

    if os.path.exists(input_video):
        v = cv2.VideoCapture(input_video)
        print(f"Using video file: {input_video}")
    else:
        print(f"❌ Input video not found: {input_video}")
        return


    if not v.isOpened():
        print("Error: Could not open video source.")
        return

    w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(v.get(cv2.CAP_PROP_FPS)) if os.path.exists(input_video) else 20
    total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT)) if os.path.exists(input_video) else 0
    
    save = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("✅ Tracking started. Press 'q' to quit.")
    print(f"📝 Logging tracking data to: {tracking_log_file}")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = v.read()
            if not ret:
                print("End of video reached.")
                break

            frame_count += 1
            frame_time = frame_count / fps
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {fps_current:.1f}")
                else:
                    print(f"Processed {frame_count} frames - FPS: {fps_current:.1f}")

            results = yolo_model.predict(
                source=frame,
                classes=[0],
                device=device,
                verbose=False,
                conf=low_conf_thresh,
                iou=0.7,
                half=True if device == "cuda" else False,
                imgsz=640
            )
            
            if len(results[0].boxes.data) == 0:
                cv2.imshow("ByteTrack + Re-ID Tracker", frame)
                save.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            all_boxes_data = results[0].boxes.data.cpu().numpy()

            high_conf_mask = all_boxes_data[:, 4] >= high_conf_thresh
            low_conf_mask = (all_boxes_data[:, 4] >= low_conf_thresh) & (all_boxes_data[:, 4] < high_conf_thresh)

            high_conf_boxes = all_boxes_data[high_conf_mask][:, :4]
            low_conf_boxes = all_boxes_data[low_conf_mask][:, :4]
            
            tracked_objects = tracker.update(high_conf_boxes, low_conf_boxes, frame, frame_count, frame_time)

            show_frame = frame.copy()
            for tid, data in tracked_objects.items():
                x1, y1, x2, y2 = map(int, data['box'])
                
                color = ((tid * 60) % 255, (tid * 100) % 255, (tid * 40) % 255)
                
                lev = data.get('lev', 'N/A')
                if lev == "New":
                    thickness = 3
                    color = (0, 255, 0)
                elif "Re-ID" in lev:
                    thickness = 3
                    color = (255, 0, 255)
                elif lev == "Occluded":
                    thickness = 2
                    color = (0, 165, 255)
                elif lev == "Lost":
                    thickness = 1
                    color = (0, 0, 255)
                else:
                    thickness = 2
                
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), color, thickness)
                label = f"ID:{tid} ({lev})"
                cv2.putText(show_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            info_text = f"YOLO11 | Frame: {frame_count} | Active Tracks: {len([t for t in tracked_objects.values() if t['age'] == 0])}"
            cv2.putText(show_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("YOLO11 ByteTrack + Re-ID Tracker", show_frame)
            save.write(show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    finally:
        v.release()
        save.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.write_summary(frame_count, fps)
        
        print(f"\n✅ Tracking completed!")
        print(f"🤖 Used YOLO11 for detection")
        print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Output saved to: {output_file}")
        print(f"📝 Tracking log saved to: {tracking_log_file}")


if __name__ == "__main__":
    main()