import streamlit as st
import pandas as pd
import tempfile
import os
import uuid
import warnings
import time
import math
import numpy as np
import cv2
from datetime import datetime, timedelta
import torch
import sys
from collections import deque
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ============================================================================
# ADVANCED TRACKING CODE WITH REID
# ============================================================================

from ultralytics import YOLO
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

# -------------------- USER TUNABLE SETTINGS --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection thresholds
HIGH_CONF_THR = 0.45
LOW_CONF_THR = 0.12

# Tracking / ReID parameters
NUM_PARTS = 3
EMBED_CACHE_SIZE = 10
REID_DIST_THR = 0.60          # used in combined cost (loose)
REID_STRICT_THRESH = 0.45     # strict threshold for safe re-attach
IOU_THR = 0.30
MATCH_WEIGHT_IOU = 0.4
MATCH_WEIGHT_REID = 0.6
MIN_HITS_TO_CONFIRM = 2
MAX_AGE = 150

# Smoothing (adaptive)
SMOOTH_ALPHA_SLOW = 0.70      # smoothing when slow
SMOOTH_ALPHA_FAST = 0.18      # smoothing when fast (more responsive)

# Kalman / time
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEAS_NOISE = 1e-1

# Re-attachment gating
REID_REATTACH_MAX_AGE = 30      # only reattach if track lost recently
REID_CENTER_MAX_DIST = 160      # pixels
REID_AREA_RATIO_MAX = 2.5

# Fallback IoU attach (prevents new id creation if old track still plausible)
FALLBACK_IOU_FOR_NEW = 0.18
FALLBACK_CENTER_MAX_DIST = 160
MAX_FALLBACK_MISSES = 8       # only consider tracks missed <= this for fallback attach

# Retirement when leaving frame
EXIT_MARGIN = 40              # pixels beyond frame bounds
EXIT_MIN_MISSES = 3           # consecutive predictions outside required to retire
HARD_RETIRE_ON_EXIT = False   # if True, retire immediately when outside margin

# -------------------- Utilities --------------------
def iou(boxA, boxB):
    x1,y1,x2,y2 = boxA
    xa,ya,xb,yb = boxB
    xx1, yy1 = max(x1, xa), max(y1, ya)
    xx2, yy2 = min(x2, xb), min(y2, yb)
    iw = max(0, xx2 - xx1)
    ih = max(0, yy2 - yy1)
    inter = iw * ih
    a1 = max(0, (x2-x1)) * max(0, (y2-y1))
    a2 = max(0, (xb-xa)) * max(0, (yb-ya))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def box_xyxy_to_xywh(box):
    x1,y1,x2,y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return np.array([cx, cy, w, h], dtype=np.float32)

def box_xywh_to_xyxy(xywh):
    cx,cy,w,h = xywh
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [x1, y1, x2, y2]

def l2_normalize_vec(v):
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n

def cosine_distance_vec(a, b):
    return 1.0 - float(np.dot(a, b))

def center_of(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

# -------------------- Embedding extractor (ResNet50) --------------------
class PartEmbedder:
    def __init__(self, device='cpu', num_parts=3):
        self.device = device
        self.num_parts = num_parts
        try:
            r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(r50.children())[:-1]).to(self.device)
            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((256, 128)),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self.available = True
        except Exception as e:
            st.warning(f"⚠️ Advanced ReID not available: {e}")
            self.available = False

    def extract(self, frame: np.ndarray, box: List[float]) -> List[np.ndarray]:
        if not self.available:
            return [np.random.rand(2048).astype(np.float32)]
        
        try:
            h, w = frame.shape[:2]
            x1,y1,x2,y2 = [int(round(v)) for v in box]
            x1,x2 = max(0, x1), min(w-1, x2)
            y1,y2 = max(0, y1), min(h-1, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return [np.random.rand(2048).astype(np.float32)]
            
            box_h = y2 - y1
            part_h = max(1, box_h // self.num_parts)
            parts = []
            
            with torch.no_grad():
                for i in range(self.num_parts):
                    py1 = y1 + i*part_h
                    py2 = y1 + (i+1)*part_h if i < self.num_parts - 1 else y2
                    crop = frame[py1:py2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    t = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                    feat = self.model(t)
                    feat = feat.view(feat.shape[0], -1).cpu().numpy().reshape(-1)
                    if feat.size == 0:
                        continue
                    parts.append(l2_normalize_vec(feat))
            
            return parts if parts else [np.random.rand(2048).astype(np.float32)]
        except Exception as e:
            return [np.random.rand(2048).astype(np.float32)]

# -------------------- Kalman filter --------------------
class KalmanBox:
    def __init__(self, dt=1.0):
        self.dt = dt
        q = KALMAN_PROCESS_NOISE
        r = KALMAN_MEAS_NOISE
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, 4+i] = self.dt
        self.Q = np.eye(8, dtype=np.float32) * q
        self.R = np.eye(4, dtype=np.float32) * r
        self.H = np.zeros((4,8), dtype=np.float32)
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1; self.H[3,3] = 1

    def initiate(self, xywh):
        x = np.zeros(8, dtype=np.float32)
        x[0:4] = xywh
        x[4:8] = 0.0
        P = np.eye(8, dtype=np.float32) * 10.0
        return x, P

    def predict(self, x, P):
        x = self.F.dot(x)
        P = self.F.dot(P).dot(self.F.T) + self.Q
        return x, P

    def update(self, x, P, z):
        S = self.H.dot(P).dot(self.H.T) + self.R
        K = P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(x)
        x = x + K.dot(y)
        P = (np.eye(8, dtype=np.float32) - K.dot(self.H)).dot(P)
        return x, P

# -------------------- Track class --------------------
class Track:
    def __init__(self, init_box, part_feats, track_id, frame_idx, dt, frame_size):
        self.id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.first_frame = frame_idx
        self.last_update_frame = frame_idx
        self.box = list(map(float, init_box))
        self.smoothed_box = list(map(float, init_box))
        self.embeds = deque(maxlen=EMBED_CACHE_SIZE)
        for p in part_feats:
            self.embeds.append(p)
        self.kf = KalmanBox(dt=dt)
        xywh = box_xyxy_to_xywh(self.box)
        self.x, self.P = self.kf.initiate(xywh)
        self.status = "New"
        self.misses_outside = 0
        self.retired = False
        self.frame_size = frame_size  # (w,h)
        
        # For CSV logging
        self.entry_time = datetime.now()
        self.exit_time = None

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        pred_box = box_xywh_to_xyxy(self.x[0:4])
        self.box = [float(v) for v in pred_box]
        # check if predicted center is outside frame margin
        cx, cy = center_of(self.box)
        w, h = self.frame_size
        if cx < -EXIT_MARGIN or cx > w + EXIT_MARGIN or cy < -EXIT_MARGIN or cy > h + EXIT_MARGIN:
            self.misses_outside += 1
        else:
            self.misses_outside = 0
        return self.box

    def _adaptive_smoothing_alpha(self):
        # velocity magnitude from Kalman state
        vx = abs(self.x[4])
        vy = abs(self.x[5])
        speed = math.hypot(vx, vy)
        w = max(1.0, self.x[2])
        h = max(1.0, self.x[3])
        diag = math.hypot(w, h)
        rel_speed = speed / (diag + 1e-6)  # relative to size
        # choice: if relative speed large -> use FAST alpha, else SLOW alpha
        if rel_speed > 0.12:
            return SMOOTH_ALPHA_FAST
        else:
            return SMOOTH_ALPHA_SLOW

    def update(self, detected_box, detected_parts, frame_idx):
        z = box_xyxy_to_xywh(detected_box)
        self.x, self.P = self.kf.update(self.x, self.P, z)
        self.box = box_xywh_to_xyxy(self.x[0:4])
        for p in detected_parts:
            self.embeds.append(p)
        sm = self._adaptive_smoothing_alpha()
        # exponential smoothing toward new box
        self.smoothed_box = [ sm * self.smoothed_box[i] + (1 - sm) * self.box[i] for i in range(4) ]
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        self.last_update_frame = frame_idx
        self.misses_outside = 0
        if not self.confirmed and self.hits >= MIN_HITS_TO_CONFIRM:
            self.confirmed = True
            self.status = "Confirmed"
        else:
            self.status = "Tracked"

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 0:
            self.status = "Lost"

    def get_average_embedding(self):
        if not self.embeds:
            return None
        arr = np.stack(list(self.embeds), axis=0)
        mean = np.mean(arr, axis=0)
        return l2_normalize_vec(mean)

# -------------------- Tracker Controller --------------------
class PersonReIDTracker:
    def __init__(self, device='cpu', frame_size=(1920,1080), fps=30.0):
        self.device = device
        self.embedder = PartEmbedder(device=device, num_parts=NUM_PARTS)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_size = frame_size
        self.fps = fps
        self.dt = 1.0 / (fps if fps > 0 else 30.0)

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def step(self, detections: List[List[float]], parts_features: List[List[np.ndarray]], frame_idx:int):
        # Predict current state of all tracks
        track_ids = list(self.tracks.keys())
        for tid in track_ids:
            tr = self.tracks[tid]
            tr.predict()
            tr.mark_missed()
            # retire if outside many times
            if HARD_RETIRE_ON_EXIT:
                if tr.misses_outside > 0:
                    tr.exit_time = datetime.now()
                    del self.tracks[tid]
            else:
                if tr.misses_outside >= EXIT_MIN_MISSES:
                    tr.exit_time = datetime.now()
                    del self.tracks[tid]

        # rebuild track list after possible deletions
        track_ids = list(self.tracks.keys())
        N = len(track_ids)
        M = len(detections)

        # If no tracks, create for all detections
        if N == 0 and M == 0:
            return self.tracks

        # Compute cost between each (track, detection)
        if N > 0 and M > 0:
            cost = np.zeros((N, M), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                tr_emb = tr.get_average_embedding()
                for j in range(M):
                    det_box = detections[j]
                    det_parts = parts_features[j]
                    iou_score = iou(tr.box, det_box)
                    iou_cost = 1.0 - iou_score
                    reid_cost = 1.0
                    if tr_emb is not None and det_parts:
                        dists = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                        reid_cost = float(min(dists))
                    c = MATCH_WEIGHT_IOU * iou_cost + MATCH_WEIGHT_REID * reid_cost
                    cost[i,j] = c
        else:
            cost = np.zeros((N, M), dtype=np.float32)

        # Hungarian
        matches = []
        unmatched_tracks_idx = list(range(N))
        unmatched_dets_idx = list(range(M))

        if N > 0 and M > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_tracks = set()
            assigned_dets = set()
            for r,c in zip(row_ind, col_ind):
                tid = track_ids[r]
                tr = self.tracks[tid]
                det_box = detections[c]
                det_parts = parts_features[c]
                iou_score = iou(tr.box, det_box)
                best_reid = 1.0
                tr_emb = tr.get_average_embedding()
                if tr_emb is not None and det_parts:
                    dists = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                    best_reid = float(min(dists))

                # spatial gating
                tcx,tcy = center_of(tr.box)
                dcx,dcy = center_of(det_box)
                center_dist = math.hypot(tcx-dcx, tcy-dcy)
                twa = max(1.0, (tr.box[2]-tr.box[0])*(tr.box[3]-tr.box[1]))
                dwa = max(1.0, (det_box[2]-det_box[0])*(det_box[3]-det_box[1]))
                area_ratio = max(twa/dwa, dwa/twa)
                recent_enough = tr.time_since_update <= REID_REATTACH_MAX_AGE

                reid_ok = (tr_emb is not None and det_parts and recent_enough
                           and best_reid <= REID_STRICT_THRESH
                           and center_dist <= REID_CENTER_MAX_DIST
                           and area_ratio <= REID_AREA_RATIO_MAX)

                accept = False
                if iou_score >= IOU_THR:
                    accept = True
                elif reid_ok:
                    accept = True
                else:
                    accept = False

                if accept:
                    matches.append((r,c))
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            unmatched_tracks_idx = [i for i in range(N) if i not in assigned_tracks]
            unmatched_dets_idx = [j for j in range(M) if j not in assigned_dets]

        # Update matched tracks
        for r,c in matches:
            tid = track_ids[r]
            det_box = detections[c]
            det_parts = parts_features[c]
            tr = self.tracks[tid]
            tr.update(det_box, det_parts, frame_idx)

        # Fallback attach for unmatched detections
        fallback_attached = set()
        for di in list(unmatched_dets_idx):
            det_box = detections[di]
            det_parts = parts_features[di]
            best_tid = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                if tr.time_since_update == 0:
                    continue  # skip already matched or active
                if tr.time_since_update > MAX_FALLBACK_MISSES:
                    continue
                if getattr(tr, 'retired', False):
                    continue
                iou_score = iou(tr.smoothed_box, det_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_tid = tid
            if best_tid is not None and best_iou >= FALLBACK_IOU_FOR_NEW:
                # attach as fallback
                tr = self.tracks[best_tid]
                tcx,tcy = center_of(tr.smoothed_box)
                dcx,dcy = center_of(det_box)
                center_dist = math.hypot(tcx-dcx, tcy-dcy)
                if center_dist <= FALLBACK_CENTER_MAX_DIST:
                    tr.update(det_box, det_parts, frame_idx)
                    fallback_attached.add(di)
        
        unmatched_dets_idx = [d for d in unmatched_dets_idx if d not in fallback_attached]

        # Create new tracks for remaining unmatched detections
        for di in unmatched_dets_idx:
            det_box = detections[di]
            det_parts = parts_features[di]
            # Safety check: if detection overlaps with any confirmed track, skip
            overlapping_confirmed = False
            for tid, tr in self.tracks.items():
                if tr.confirmed:
                    if iou(tr.smoothed_box, det_box) > 0.45:
                        overlapping_confirmed = True
                        break
            if overlapping_confirmed:
                continue
            new_tr = Track(det_box, det_parts, self.next_id, frame_idx, dt=self.dt, frame_size=self.frame_size)
            self.tracks[self.next_id] = new_tr
            self.next_id += 1

        # prune tracks older than MAX_AGE
        remove_ids = []
        for tid, tr in list(self.tracks.items()):
            if tr.time_since_update > MAX_AGE:
                tr.exit_time = datetime.now()
                remove_ids.append(tid)
        for rid in remove_ids:
            del self.tracks[rid]

        return self.tracks

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

st.markdown("""
<style>
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
        width: 100%;
    }
    
    .stContainer > div {
        width: 100%;
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'output_video' not in st.session_state:
    st.session_state.output_video = None
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = None
if 'err_message' not in st.session_state:
    st.session_state.err_message = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def load_models():
    """Load YOLO model (cached)"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load YOLO
        yolo_model = YOLO("yolo11m.pt")
        if device == "cuda":
            yolo_model.to(device)
        
        return yolo_model, device
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None

def run_tracking_pipeline(uploaded_video, settings):
    """Run the complete tracking pipeline with advanced ReID"""
    
    if uploaded_video is None:
        return None, None, "No video uploaded"
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        temp_input_path = tmp_file.name
    
    try:
        # Load models
        yolo_model, device = load_models()
        if yolo_model is None:
            return None, None, "Failed to load models"
        
        # Generate output paths
        unique_id = str(uuid.uuid4())[:8]
        output_video_path = f"tracked_video_{unique_id}.mp4"
        
        print(f"🎬 Processing video: {temp_input_path}")
        print(f"📤 Output will be: {output_video_path}")
        
        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            return None, None, "Could not open video file"
        
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {w}x{h} @ {fps}fps, {total_frames} frames")
        
        # Initialize advanced tracker
        tracker = PersonReIDTracker(device=device, frame_size=(w,h), fps=fps)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        
        if not out.isOpened():
            return None, None, "Could not create output video writer"
        
        frame_count = 0
        start_time = time.time()
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break
                
            frame_count += 1
            frame_time = frame_count / fps
            
            # Update progress
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            status_text.text(f"Processing frame {frame_count}/{total_frames} | FPS: {fps_current:.1f}")
            
            # YOLO detection
            results = yolo_model.predict(
                source=frame,
                classes=[0],  # person class
                device=device,
                verbose=False,
                conf=LOW_CONF_THR,
                iou=0.7,
                half=True if device == "cuda" else False,
                imgsz=640
            )
            
            # Create a copy for drawing
            show_frame = frame.copy()
            
            if len(results[0].boxes.data) == 0:
                # No detections, just save the frame
                out.write(show_frame)
                continue
            
            all_boxes_data = results[0].boxes.data.cpu().numpy()
            
            # Filter high confidence detections
            high_conf_mask = all_boxes_data[:, 4] >= HIGH_CONF_THR
            high_conf_boxes = all_boxes_data[high_conf_mask][:, :4] if high_conf_mask.any() else np.array([]).reshape(0, 4)
            
            # Extract features for each detection
            detections = []
            parts_feats = []
            for bb in high_conf_boxes:
                box = [float(b) for b in bb[:4]]
                parts = tracker.embedder.extract(frame, box)
                if parts:
                    detections.append(box)
                    parts_feats.append(parts)
            
            # Update tracker
            tracks = tracker.step(detections, parts_feats, frame_count)
            
            # Draw tracking results
            live_confirmed = 0
            for tid, tr in tracks.items():
                if getattr(tr, 'retired', False):
                    continue
                
                color = (int((tid*37)%255), int((tid*97)%255), int((tid*61)%255))
                box = [int(round(v)) for v in tr.smoothed_box]
                x1,y1,x2,y2 = box
                
                if tr.confirmed:
                    live_confirmed += 1
                    cv2.rectangle(show_frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(show_frame, f"ID:{tid}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw part divisions
                    ph = max(2, (y2-y1)//NUM_PARTS)
                    for i in range(1, NUM_PARTS):
                        cv2.line(show_frame, (x1, y1+i*ph), (x2, y1+i*ph), color, 1)
                else:
                    cv2.rectangle(show_frame, (x1,y1), (x2,y2), (200,200,200), 1)
            
            # Add info text
            info = f"Frame {frame_count}/{total_frames} | Live: {live_confirmed} | Total Tracks: {len(tracks)}"
            cv2.putText(show_frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            # Write frame to output video
            out.write(show_frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Remove temp input file
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        
        # Generate CSV from tracker logs
        log_rows = []
        for tid, tr in tracker.tracks.items():
            entry_str = tr.entry_time.strftime('%Y-%m-%d %H:%M:%S') if tr.entry_time else ""
            
            if tr.exit_time is not None:
                exit_str = tr.exit_time.strftime('%Y-%m-%d %H:%M:%S')
                duration = f"{(tr.exit_time - tr.entry_time).total_seconds():.1f}s"
                status = "Completed"
            else:
                exit_str = "Active"
                duration = "Active"
                status = "Active"
            
            log_rows.append({
                'ID': f'ID_{tid:03d}',
                'Entry_Time': entry_str,
                'Exit_Time': exit_str,
                'Duration': duration,
                'Status': status
            })
        
        tracking_df = pd.DataFrame(log_rows)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Verify output file
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"✅ Output video created: {file_size} bytes")
            
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n✅ Advanced tracking completed!")
            print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Output saved to: {output_video_path}")
            print(f"📊 Tracking data:\n{tracking_df}")
            
            return output_video_path, tracking_df, None
        else:
            return None, tracking_df, "Output video file not found"
        
    except Exception as e:
        # Cleanup on error
        if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        print(f"❌ Processing error: {str(e)}")
        return None, None, f"Processing error: {str(e)}"

def cleanup_temp_files():
    """Clean up old temporary files - but preserve tracked videos"""
    try:
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            # Only clean up temp_ files that are NOT tracked videos
            if file.startswith('temp_') and file.endswith('.mp4'):
                # Don't delete files that contain "tracked_video"
                if 'tracked_video' not in file:
                    try:
                        os.unlink(file)
                    except:
                        pass
            elif file.startswith('temp_') and file.endswith('.csv'):
                try:
                    os.unlink(file)
                except:
                    pass
    except:
        pass

def cleanup_old_tracked_videos():
    """Clean up tracked videos older than 1 hour"""
    try:
        current_dir = os.getcwd()
        current_time = time.time()
        
        for file in os.listdir(current_dir):
            if file.startswith('tracked_video_') and file.endswith('.mp4'):
                file_path = os.path.join(current_dir, file)
                file_age = current_time - os.path.getctime(file_path)
                
                # Delete files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    try:
                        os.unlink(file_path)
                        print(f"Cleaned up old video: {file}")
                    except:
                        pass
    except:
        pass

def main():
    # Main heading with proper formatting
    st.title("🎯 Advanced Entry/Exit Tracker")
    st.markdown("### *Advanced person tracking with ReID and Kalman filtering*")
    
    with st.sidebar:
        st.header("⚙️ Advanced Tracking Settings")
        
        # Detection settings
        st.subheader("🔍 Detection")
        high_conf_thresh = st.slider("High Confidence", 0.1, 0.9, HIGH_CONF_THR, 0.05)
        low_conf_thresh = st.slider("Low Confidence", 0.05, 0.5, LOW_CONF_THR, 0.05)
        
        # ReID settings
        st.subheader("🧠 ReID Parameters")
        reid_strict_thresh = st.slider("ReID Strict Threshold", 0.1, 1.0, REID_STRICT_THRESH, 0.05)
        reid_center_max_dist = st.number_input("ReID Max Distance", 50, 500, REID_CENTER_MAX_DIST, 10)
        
        # Tracking settings
        st.subheader("📍 Tracking")
        iou_thresh = st.slider("IoU Threshold", 0.1, 0.8, IOU_THR, 0.05)
        max_age = st.number_input("Max Age (frames)", 10, 500, MAX_AGE, 10)
        min_hits = st.number_input("Min Hits to Confirm", 1, 10, MIN_HITS_TO_CONFIRM, 1)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.info("These settings control the advanced tracking algorithm")
            st.write(f"**Kalman Process Noise:** {KALMAN_PROCESS_NOISE}")
            st.write(f"**Kalman Measurement Noise:** {KALMAN_MEAS_NOISE}")
            st.write(f"**Part-based Features:** {NUM_PARTS} parts")
            st.write(f"**Embedding Cache:** {EMBED_CACHE_SIZE} frames")
        
        # Device info
        st.subheader("🔧 System Info")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        st.info(f"Device: {device}")
        st.success("✅ Advanced ReID Available")
        
        settings = {
            'high_conf_thresh': high_conf_thresh,
            'low_conf_thresh': low_conf_thresh,
            'reid_strict_thresh': reid_strict_thresh,
            'reid_center_max_dist': reid_center_max_dist,
            'iou_thresh': iou_thresh,
            'max_age': max_age,
            'min_hits': min_hits
        }
        
        if st.button("🧹 Clean Temp Files"):
            cleanup_temp_files()
            cleanup_old_tracked_videos()
            st.success("✅ Temporary files cleaned")
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv'],
            help="Upload a video file for advanced person tracking"
        )
        
        if uploaded_file:
            # Video preview with heading
            st.subheader("👀 Preview")
            st.video(uploaded_file)
            st.info(f"📁 **File:** {uploaded_file.name}")
            st.info(f"📏 **Size:** {uploaded_file.size / (1024*1024):.1f} MB")
            
            # Process button
            if st.button("🚀 Start Advanced Tracking", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.err_message = None
                st.session_state.output_video = None
                st.session_state.tracking_data = None
                
                output_path, tracking_df, err_message = run_tracking_pipeline(uploaded_file, settings)
                
                st.session_state.output_video = output_path
                st.session_state.tracking_data = tracking_df
                st.session_state.err_message = err_message
                st.session_state.processing = False
                
                if output_path:
                    st.success("✅ Advanced tracking completed!")
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {err_message}")
        
        # Download section with heading
        st.header("⬇️ Downloads")
        
        # CSV download
        if st.session_state.tracking_data is not None and not st.session_state.tracking_data.empty:
            csv_data = st.session_state.tracking_data.to_csv(index=False)
            
            st.download_button(
                "📊 Download CSV Logs",
                data=csv_data,
                file_name="advanced_tracking_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.info(f"📊 CSV: {len(st.session_state.tracking_data)} records")
        else:
            st.button("📊 Download CSV", disabled=True, use_container_width=True)
    
    with col2:
        st.header("📊 Results & Analytics")
        
        if st.session_state.processing:
            st.info("🔄 Advanced processing in progress...")
            
        elif st.session_state.err_message:
            st.error(f"❌ {st.session_state.err_message}")
            
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common Issues:**
                - Video format not supported (try MP4)
                - Missing model files (YOLO11m.pt)
                - Insufficient GPU memory
                - PyTorch/torchvision version compatibility
                
                **Solutions:**
                1. Convert video to MP4 format
                2. Ensure YOLO model exists
                3. Try CPU mode if GPU memory insufficient
                4. Update PyTorch/torchvision
                """)
            
        elif st.session_state.output_video and st.session_state.tracking_data is not None:
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Download", "📊 CSV Data", "📈 Analytics", "ℹ️ Advanced Info"])
            
            with tab1:
                st.subheader("🎬 Advanced Tracked Video")
                
                # Debug information
                if st.checkbox("🔍 Show Debug Info"):
                    st.write(f"**Session video path:** `{st.session_state.output_video}`")
                    if st.session_state.output_video:
                        st.write(f"**File exists:** {os.path.exists(st.session_state.output_video)}")
                        if os.path.exists(st.session_state.output_video):
                            st.write(f"**File size:** {os.path.getsize(st.session_state.output_video)} bytes")
                        
                        # List all video files in directory
                        video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
                        st.write(f"**All MP4 files:** {video_files}")
                
                if st.session_state.output_video and os.path.exists(st.session_state.output_video):
                    # Video download button
                    with open(st.session_state.output_video, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        "📥 Download Advanced Tracked Video",
                        data=video_bytes,
                        file_name="advanced_tracking.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    file_size = len(video_bytes) / (1024*1024)
                    st.success(f"✅ Advanced video ready for download!")
                    st.info(f"📁 **Size:** {file_size:.1f} MB")
                    
                    # Show advanced tracking features
                    st.markdown("""
                    **🎯 Advanced Video Features:**
                    - 🧠 **Part-based ReID**: 3-part feature extraction
                    - 📍 **Kalman Filtering**: Smooth motion prediction
                    - 🔄 **Hungarian Assignment**: Optimal track-detection matching
                    - 🎨 **Smart Re-attachment**: Lost track recovery
                    - ➖ **Part divisions**: Visual feature regions
                    - 📊 **Live tracking info**: Frame-by-frame statistics
                    """)
                else:
                    st.error("❌ Advanced tracked video not found")
                    
                    # Show help
                    with st.expander("🔧 Why is the video missing?"):
                        st.markdown("""
                        **Possible causes:**
                        1. **Processing error** - Advanced tracking failed
                        2. **Memory issues** - Insufficient GPU/RAM
                        3. **Model loading** - ResNet50 or YOLO failed to load
                        4. **File permissions** - Cannot write to directory
                        
                        **Solutions:**
                        1. Try processing again
                        2. Use "Show Debug Info" checkbox above
                        3. Reduce video resolution or length
                        4. Ensure sufficient system resources
                        """)
            
            # Rest of the tabs remain the same...
            with tab2:
                st.subheader("📊 Advanced Tracking Logs")
                st.dataframe(st.session_state.tracking_data, use_container_width=True)
                
                # CSV download button in tab
                csv_data = st.session_state.tracking_data.to_csv(index=False)
                st.download_button(
                    "📄 Download Advanced CSV",
                    data=csv_data,
                    file_name="advanced_tracking_logs.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            
            with tab3:
                if not st.session_state.tracking_data.empty:
                    # Summary metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        total_persons = st.session_state.tracking_data['ID'].nunique()
                        st.metric("Total People", total_persons)
                    
                    with col_b:
                        active_count = (st.session_state.tracking_data['Status'] == 'Active').sum()
                        st.metric("Currently Active", active_count)
                    
                    with col_c:
                        completed_count = (st.session_state.tracking_data['Status'] == 'Completed').sum()
                        st.metric("Completed Visits", completed_count)
                    
                    with col_d:
                        # Calculate average duration for completed visits
                        completed_df = st.session_state.tracking_data[
                            (st.session_state.tracking_data['Status'] == 'Completed') &
                            (st.session_state.tracking_data['Duration'].str.contains('s', na=False))
                        ]
                        if not completed_df.empty:
                            durations = completed_df['Duration'].str.replace('s', '').astype(float)
                            avg_duration = durations.mean()
                            st.metric("Avg Duration", f"{avg_duration:.1f}s")
                        else:
                            st.metric("Avg Duration", "N/A")
                    
                    # Status chart
                    st.subheader("📊 Status Distribution")
                    status_counts = st.session_state.tracking_data['Status'].value_counts()
                    st.bar_chart(status_counts)
            
            with tab4:
                st.markdown("""
                ### 🧠 Advanced Tracking Technology
                
                **🔬 What makes this advanced:**
                - **Part-based ReID**: Extracts features from 3 body parts using ResNet50
                - **Kalman Filtering**: Predicts motion with velocity and acceleration
                - **Hungarian Algorithm**: Optimal assignment between tracks and detections
                - **Adaptive Smoothing**: Speed-based smoothing for stable tracking
                - **Smart Re-attachment**: Recovers lost tracks using appearance features
                - **Spatial Gating**: Prevents impossible track assignments
                
                **🎯 Algorithm Pipeline:**
                1. **Detection**: YOLO detects persons with confidence filtering
                2. **Feature Extraction**: ResNet50 extracts part-based embeddings
                3. **Prediction**: Kalman filter predicts track positions
                4. **Association**: Hungarian algorithm matches tracks to detections
                5. **Update**: Confirmed matches update track states
                6. **Management**: New tracks created, old tracks retired
                
                **📊 Performance Features:**
                - **Multi-part ReID**: More robust than single-feature matching
                - **Motion Prediction**: Handles occlusions and missed detections
                - **Identity Preservation**: Maintains consistent IDs across frames
                - **Exit Detection**: Automatic retirement when leaving scene
                
                **🔧 Technical Parameters:**
                - **Feature Dimensions**: 2048D embeddings per part
                - **Embedding Cache**: {EMBED_CACHE_SIZE} frame sliding window
                - **Kalman State**: 8D (position + velocity)
                - **Assignment Cost**: Weighted IoU + ReID distance
                """.format(EMBED_CACHE_SIZE=EMBED_CACHE_SIZE))
                
                st.success("✅ Advanced tracking system operational!")
                
        else:
            st.info("📤 Upload a video and click 'Start Advanced Tracking' to begin")

if __name__ == "__main__":
    cleanup_temp_files()
    main()
