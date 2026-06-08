#!/usr/bin/env python3
"""
detect_helmet_video.py

Biker POV Helmet Violation Detection and Premium Video Annotation.
Detects vehicles, helmets, and riders without helmets. Automatically matches
head detections with vehicles using Intersection over Area (IoA). If a rider 
without a helmet is confirmed with conf > 90% (or custom), crops out the vehicle
and saves it to a timestamp-named folder. Saves the annotated video with professional overlays.

"""

import os
import sys
import time
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR

# Define colors (BGR format)
COLOR_VIOLATION = (50, 50, 240)        # Crimson Red
COLOR_COMPLIANT = (76, 201, 24)        # Mint Green
COLOR_UNKNOWN = (180, 180, 180)        # Slate Grey
COLOR_HUD_ACCENT_NORMAL = (255, 150, 0) # Cool Cyan/Blue
COLOR_HUD_BG = (18, 18, 18)            # Dark grey for HUD background
COLOR_TEXT = (255, 255, 255)           # White


@dataclass
class DetectionConfig:
    """Stores system configurations, settings, thresholds, and execution flags."""
    video_path: str
    model_path: str = r"runs\detect\train\weights\best.pt"
    output_path: Optional[str] = None
    crop_dir: str = "crops"
    conf_threshold: float = 0.80
    det_conf: float = 0.40
    iou_match_threshold: float = 0.80
    save_crops: bool = True
    show_compliant: bool = True


class HelmetDetector:
    """Manages model loading, object detection/tracking, and matching head-to-vehicle associations."""
    def __init__(self, model_path: str, iou_match_threshold: float = 0.80):
        print(f"Loading RT-DETR model: {model_path}")
        self.model = RTDETR(model_path)
        self.iou_match_threshold = iou_match_threshold

    @staticmethod
    def compute_ioa(box_head: np.ndarray, box_vehicle: np.ndarray) -> float:
        """
        Computes Intersection over Area (IoA) of the head box.
        This measures how much of the head box is inside the vehicle box.
        """
        hx1, hy1, hx2, hy2 = box_head
        vx1, vy1, vx2, vy2 = box_vehicle
        
        ix1 = max(hx1, vx1)
        iy1 = max(hy1, vy1)
        ix2 = min(hx2, vx2)
        iy2 = min(hy2, vy2)
        
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            head_area = (hx2 - hx1) * (hy2 - hy1)
            return inter_area / head_area if head_area > 0 else 0.0
        return 0.0

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Computes standard Intersection over Union (IoU) between two boxes.
        Used for fallback tracking deduplication.
        """
        b1x1, b1y1, b1x2, b1y2 = box1
        b2x1, b2y1, b2x2, b2y2 = box2
        
        ix1 = max(b1x1, b2x1)
        iy1 = max(b1y1, b2y1)
        ix2 = min(b1x2, b2x2)
        iy2 = min(b1y2, b2y2)
        
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
            area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0
        return 0.0

    def track(self, frame: np.ndarray, conf_threshold: float = 0.25):
        """Runs tracking inference on a frame."""
        return self.model.track(frame, persist=True, verbose=False, conf=conf_threshold)

    def group_and_match(self, results) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Groups raw detections into helmets, vehicles, and no-helmets,
        then matches heads/riders to vehicles using Intersection over Area (IoA).
        """
        if len(results) == 0 or results[0].boxes is None:
            return [], [], []

        boxes = results[0].boxes.cpu()
        xyxy = boxes.xyxy.numpy()
        clss = boxes.cls.numpy().astype(int)
        confs = boxes.conf.numpy()
        
        if boxes.id is not None:
            track_ids = boxes.id.numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)
            
        helmets = []
        vehicles = []
        no_helmets = []
        
        # Group predictions
        for i in range(len(boxes)):
            det = {
                'box': xyxy[i],
                'conf': confs[i],
                'track_id': track_ids[i],
                'cls': clss[i]
            }
            if clss[i] == 0:
                helmets.append(det)
            elif clss[i] == 1:
                vehicles.append(det)
            elif clss[i] == 2:
                no_helmets.append(det)
        
        # 1. Match heads to vehicles
        matched_vehicles = []
        for v in vehicles:
            matched_vehicles.append({
                'det': v,
                'helmets': [],
                'no_helmets': []
            })
            
        for h in helmets:
            best_v = None
            best_ioa = 0.0
            for mv in matched_vehicles:
                ioa = self.compute_ioa(h['box'], mv['det']['box'])
                if ioa > best_ioa:
                    best_ioa = ioa
                    best_v = mv
            if best_v is not None and best_ioa >= self.iou_match_threshold:
                best_v['helmets'].append((h, best_ioa))
                
        for nh in no_helmets:
            best_v = None
            best_ioa = 0.0
            for mv in matched_vehicles:
                ioa = self.compute_ioa(nh['box'], mv['det']['box'])
                if ioa > best_ioa:
                    best_ioa = ioa
                    best_v = mv
            if best_v is not None and best_ioa >= self.iou_match_threshold:
                best_v['no_helmets'].append((nh, best_ioa))

        return matched_vehicles, helmets, no_helmets


class ViolationSaver:
    """Handles saving cropped violation frames and performs deduplication to prevent double-saving."""
    def __init__(self, save_dir: str, enabled: bool = True):
        self.save_dir = save_dir
        self.enabled = enabled
        self.saved_vehicle_ids: Set[int] = set()
        self.saved_crops_history: List[Dict] = []
        
        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Cropped violation images will be saved in: {self.save_dir}")
        else:
            print("Saving crops is disabled.")

    def should_save_violation(self, box: np.ndarray, track_id: Optional[int], frame_idx: int) -> bool:
        """Determines if a violation is unique and needs to be cropped and saved."""
        if not self.enabled:
            return False
            
        if track_id is not None:
            if track_id not in self.saved_vehicle_ids:
                self.saved_vehicle_ids.add(track_id)
                return True
        else:
            # Fallback duplicate check: compare with crops in last 30 frames
            duplicate = False
            for logged_crop in self.saved_crops_history:
                if frame_idx - logged_crop['frame_idx'] < 30:
                    if HelmetDetector.compute_iou(box, logged_crop['box']) > 0.40:
                        duplicate = True
                        break
            if not duplicate:
                self.saved_crops_history.append({'box': box, 'frame_idx': frame_idx})
                return True
        return False

    def save_crop_image(self, raw_frame: np.ndarray, box: np.ndarray, filename_prefix: str, frame_idx: int, video_time_str: str, track_id: Optional[int], conf: float) -> Optional[str]:
        """Crops the designated bounding box with 5% padding and saves it to disk."""
        h, w = raw_frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # 5% padding around the bounding box
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)
        
        x1_c = max(0, x1 - pad_w)
        y1_c = max(0, y1 - pad_h)
        x2_c = min(w, x2 + pad_w)
        y2_c = min(h, y2 + pad_h)
        
        crop_img = raw_frame[y1_c:y2_c, x1_c:x2_c]
        if crop_img.size == 0:
            return None
            
        # Generate structured filename
        track_str = f"id_{track_id}" if track_id is not None else "no_track"
        filename = f"{filename_prefix}_frame_{frame_idx:05d}_time_{video_time_str}_{track_str}_conf_{conf:.2f}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, crop_img)
        return filepath


class VideoAnnotator:
    """Encapsulates the premium drawing overlays and HUD design system."""
    @staticmethod
    def draw_semi_transparent_rect(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int], alpha: float = 0.15):
        """Fills a bounding box with a semi-transparent color overlay."""
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return
        sub_img = img[y1:y2, x1:x2]
        rect = np.full_like(sub_img, color, dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)

    @staticmethod
    def draw_premium_corners(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int], thickness: int = 2, corner_len: int = 15):
        """Draws thin border lines with thicker, glowing corner segments (L-shapes)."""
        h, w = img.shape[:2]
        scale = w / 1280.0
        scaled_thickness = int(max(1, round(thickness * scale)))
        scaled_corner_len = int(max(5, round(corner_len * scale)))
        scaled_border_thickness = int(max(1, round(1 * scale)))

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        # Draw thin boundary first
        cv2.rectangle(img, (x1, y1), (x2, y2), color, scaled_border_thickness)
        
        # Adjust corner length to box dimensions if box is small
        box_w, box_h = x2 - x1, y2 - y1
        c_len = min(scaled_corner_len, box_w // 3, box_h // 3)
        if c_len <= 0:
            return
            
        # Top-Left Corner
        cv2.line(img, (x1, y1), (x1 + c_len, y1), color, scaled_thickness)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), color, scaled_thickness)
        
        # Top-Right Corner
        cv2.line(img, (x2, y1), (x2 - c_len, y1), color, scaled_thickness)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), color, scaled_thickness)
        
        # Bottom-Left Corner
        cv2.line(img, (x1, y2), (x1 + c_len, y2), color, scaled_thickness)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), color, scaled_thickness)
        
        # Bottom-Right Corner
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, scaled_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, scaled_thickness)

    @staticmethod
    def draw_dotted_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1, gap: int = 5):
        """Draws a dotted connecting line between two points, scaling gap and thickness."""
        h, w = img.shape[:2]
        scale = w / 1280.0
        scaled_thickness = int(max(1, round(thickness * scale)))
        scaled_gap = int(max(2, round(gap * scale)))

        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        
        for i in range(0, int(dist), scaled_gap * 2):
            start_x = int(pt1[0] + dx * i)
            start_y = int(pt1[1] + dy * i)
            end_x = int(pt1[0] + dx * min(i + scaled_gap, dist))
            end_y = int(pt1[1] + dy * min(i + scaled_gap, dist))
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, scaled_thickness)

    @staticmethod
    def draw_premium_tag(img: np.ndarray, text: str, x1: int, y1: int, bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int] = (255, 255, 255), font_scale: float = 0.4, thickness: int = 1, position: str = 'top'):
        """
        Draws an anti-aliased label tag inside a rounded/colored solid banner.
        All dimensions scale with image size to keep labels readable on high resolution frames.
        Position can be 'top' (above y1) or 'bottom' (below y1).
        """
        h, w = img.shape[:2]
        scale = w / 1280.0
        scaled_font_scale = font_scale * scale
        scaled_thickness = int(max(1, round(thickness * scale)))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        padding_w = int(max(4, round(10 * scale)))
        padding_h = int(max(2, round(8 * scale)))
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scaled_font_scale, scaled_thickness)
        
        if position == 'top':
            # Place tag slightly above y1
            tag_y1 = max(0, y1 - text_h - padding_h)
            tag_y2 = y1 if y1 - text_h - padding_h >= 0 else y1 + text_h + padding_h
            text_y = y1 - int(4 * scale) if y1 - text_h - padding_h >= 0 else y1 + text_h + int(2 * scale)
        else:
            # Place tag slightly below y1
            tag_y1 = y1
            tag_y2 = min(h, y1 + text_h + padding_h)
            text_y = y1 + text_h + int(2 * scale)
            
        # Draw background banner
        cv2.rectangle(img, (x1, tag_y1), (x1 + text_w + padding_w, tag_y2), bg_color, -1)
        
        # Write text inside the banner
        cv2.putText(img, text, (x1 + int(5 * scale), text_y), font, scaled_font_scale, text_color, scaled_thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_hud(img: np.ndarray, violation_detected: bool, active_violations_count: int, total_saved_violations: int, fps: Optional[float] = None):
        """Draws a glassmorphic top HUD banner showing system scanning status and metrics, scaled dynamically."""
        h, w = img.shape[:2]
        scale = w / 1280.0
        bar_h = int(55 * scale)
        
        # Extract sub-region and blend with dark rectangle
        sub_img = img[0:bar_h, 0:w]
        rect = np.full_like(sub_img, COLOR_HUD_BG, dtype=np.uint8)
        img[0:bar_h, 0:w] = cv2.addWeighted(sub_img, 0.3, rect, 0.7, 0)
        
        # Draw accent line at the bottom of the HUD
        accent_color = COLOR_VIOLATION if violation_detected else COLOR_HUD_ACCENT_NORMAL
        cv2.line(img, (0, bar_h), (w, bar_h), accent_color, int(max(1, round(2 * scale))))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scaled_status_font_scale = 0.65 * scale
        scaled_hud_font_scale = 0.55 * scale
        scaled_thickness_status = int(max(1, round(2 * scale)))
        scaled_thickness_hud = int(max(1, round(1 * scale)))
        
        # Status / Alerts
        if violation_detected:
            status_text = "WARNING: NO HELMET VIOLATION"
            status_color = COLOR_VIOLATION
        else:
            status_text = "SYSTEM STATUS: SCANNING POV..."
            status_color = (0, 255, 0)  # Green
            
        cv2.putText(img, status_text, (int(20 * scale), int(38 * scale)), font, scaled_status_font_scale, status_color, scaled_thickness_status, lineType=cv2.LINE_AA)
        
        # Stats Text
        hud_text = f"ACTIVE VIOLATIONS: {active_violations_count} | TOTAL SAVED: {total_saved_violations}"
        if fps is not None:
            hud_text += f" | FPS: {fps:.1f}"
            
        (text_w, text_h), _ = cv2.getTextSize(hud_text, font, scaled_hud_font_scale, scaled_thickness_hud)
        cv2.putText(img, hud_text, (w - text_w - int(20 * scale), int(36 * scale)), font, scaled_hud_font_scale, COLOR_TEXT, scaled_thickness_hud, lineType=cv2.LINE_AA)


class VideoProcessor:
    """Orchestrates video loading, processing loop, detection, annotation, and savings workflow."""
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.detector = HelmetDetector(config.model_path, config.iou_match_threshold)
        
        # Create session path for crops
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(config.crop_dir, f"session_{session_timestamp}")
        self.saver = ViolationSaver(self.save_dir, enabled=config.save_crops)
        
        self.annotator = VideoAnnotator()

    @staticmethod
    def format_video_time(msecs: float) -> str:
        """Formats video milliseconds into a readable file/folder friendly time string."""
        secs = int(msecs // 1000) % 60
        mins = int(msecs // 60000) % 60
        hours = int(msecs // 3600000)
        ms = int(msecs % 1000)
        return f"{hours:02d}h_{mins:02d}m_{secs:02d}s_{ms:03d}ms"

    def process(self):
        """Processes the configured video stream from end to end."""
        if not os.path.exists(self.config.video_path):
            print(f"Error: Video file not found: {self.config.video_path}")
            sys.exit(1)
            
        if not self.config.output_path:
            video_dir, video_name = os.path.split(self.config.video_path)
            name_stem, _ = os.path.splitext(video_name)
            self.config.output_path = os.path.join(video_dir if video_dir else ".", f"{name_stem}_annotated.mp4")
            
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {self.config.video_path}")
            sys.exit(1)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30.0
            
        print(f"Video Info: {width}x{height} | {fps} FPS | {total_frames} frames total")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.config.output_path, fourcc, fps, (width, height))
        
        total_saved_violations = 0
        frame_idx = 0
        start_time = time.time()
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_time_str = self.format_video_time(video_time_ms)
            
            # Make a copy for rendering overlays
            annotated_frame = frame.copy()
            
            # Scale drawing parameters dynamically based on video resolution (e.g. 4K)
            scale = width / 1280.0
            thick_vehicle = int(max(2, round(3.5 * scale)))
            thick_head = int(max(1, round(2.0 * scale)))
            
            # Run inference & tracking
            results = self.detector.track(frame, conf_threshold=self.config.det_conf)
            
            violation_detected = False
            active_violations_in_frame = 0
            
            # Group and match detections
            matched_vehicles, helmets, no_helmets = self.detector.group_and_match(results)
            matched_heads_indices = set()
            
            # 1. Process and draw matched vehicles
            for mv in matched_vehicles:
                v_det = mv['det']
                v_box = v_det['box']
                v_tid = v_det['track_id']
                v_conf = v_det['conf']
                
                # Filter violations: any matched rider without helmet with conf >= config.conf_threshold
                valid_violations = [nh for nh, ioa in mv['no_helmets'] if nh['conf'] >= self.config.conf_threshold]
                
                if len(valid_violations) > 0:
                    violation_detected = True
                    active_violations_in_frame += 1
                    max_violation_conf = max(nh['conf'] for nh in valid_violations)
                    
                    # Save Crop logic
                    if self.saver.should_save_violation(v_box, v_tid, frame_idx):
                        saved_path = self.saver.save_crop_image(
                            frame, v_box, "violation_vehicle", 
                            frame_idx, video_time_str, v_tid, max_violation_conf
                        )
                        if saved_path:
                            total_saved_violations += 1
                    
                    # Visual design for violation vehicle: Crimson
                    # Fill background with translucent overlay
                    self.annotator.draw_semi_transparent_rect(annotated_frame, v_box[0], v_box[1], v_box[2], v_box[3], COLOR_VIOLATION, 0.18)
                    # Premium corners
                    self.annotator.draw_premium_corners(annotated_frame, v_box[0], v_box[1], v_box[2], v_box[3], COLOR_VIOLATION, thickness=thick_vehicle, corner_len=18)
                    # Draw text label
                    v_label = f"VIOLATION: NO HELMET" + (f" #{v_tid}" if v_tid is not None else "")
                    self.annotator.draw_premium_tag(annotated_frame, v_label, int(v_box[0]), int(v_box[1]), COLOR_VIOLATION, COLOR_TEXT, font_scale=0.45)
                    
                    # Draw connected lines and boxes for violation heads
                    v_center = (int((v_box[0] + v_box[2])/2), int((v_box[1] + v_box[3])/2))
                    for nh, ioa in mv['no_helmets']:
                        h_box = nh['box']
                        h_conf = nh['conf']
                        matched_heads_indices.add(id(nh))
                        
                        # Draw head box (thickness scaled)
                        cv2.rectangle(annotated_frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), COLOR_VIOLATION, thick_head)
                        # Dotted link line to vehicle center
                        h_center = (int((h_box[0] + h_box[2])/2), int((h_box[1] + h_box[3])/2))
                        self.annotator.draw_dotted_line(annotated_frame, h_center, v_center, COLOR_VIOLATION, thickness=thick_head)
                        # Tag head below its bounding box
                        self.annotator.draw_premium_tag(annotated_frame, f"No Helmet {h_conf:.2f}", int(h_box[0]), int(h_box[3]), COLOR_VIOLATION, COLOR_TEXT, font_scale=0.38, position='bottom')
                        
                elif len(mv['helmets']) > 0:
                    # Compliant rider (has helmet)
                    if self.config.show_compliant:
                        self.annotator.draw_premium_corners(annotated_frame, v_box[0], v_box[1], v_box[2], v_box[3], COLOR_COMPLIANT, thickness=int(max(1, round(2 * scale))), corner_len=12)
                        v_label = f"SAFE RIDER" + (f" #{v_tid}" if v_tid is not None else "")
                        self.annotator.draw_premium_tag(annotated_frame, v_label, int(v_box[0]), int(v_box[1]), COLOR_COMPLIANT, COLOR_TEXT, font_scale=0.45)
                        
                        for h, ioa in mv['helmets']:
                            h_box = h['box']
                            h_conf = h['conf']
                            matched_heads_indices.add(id(h))
                            cv2.rectangle(annotated_frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), COLOR_COMPLIANT, thick_head)
                            self.annotator.draw_premium_tag(annotated_frame, f"Helmet {h_conf:.2f}", int(h_box[0]), int(h_box[3]), COLOR_COMPLIANT, COLOR_TEXT, font_scale=0.38, position='bottom')
                else:
                    # Unknown vehicle (e.g. parked, no head detected yet)
                    if self.config.show_compliant:
                        cv2.rectangle(annotated_frame, (int(v_box[0]), int(v_box[1])), (int(v_box[2]), int(v_box[3])), COLOR_UNKNOWN, thick_head)
                        v_label = f"Vehicle" + (f" #{v_tid}" if v_tid is not None else "")
                        self.annotator.draw_premium_tag(annotated_frame, v_label, int(v_box[0]), int(v_box[1]), COLOR_UNKNOWN, COLOR_TEXT, font_scale=0.45)
            
            # 2. Handle unmatched heads (fallback verification)
            for nh in no_helmets:
                if id(nh) in matched_heads_indices:
                    continue
                    
                h_box = nh['box']
                h_conf = nh['conf']
                
                # Check confidence threshold for warning
                if h_conf >= self.config.conf_threshold:
                    violation_detected = True
                    active_violations_in_frame += 1
                    
                    # Create a virtual vehicle bounding box around this unmatched head
                    hw = h_box[2] - h_box[0]
                    hh = h_box[3] - h_box[1]
                    vx1 = max(0, int(h_box[0] - hw))
                    vy1 = max(0, int(h_box[1] - hh * 0.5))
                    vx2 = min(width, int(h_box[2] + hw))
                    vy2 = min(height, int(h_box[3] + hh * 5.0))
                    virtual_box = np.array([vx1, vy1, vx2, vy2])
                    
                    if self.saver.should_save_violation(virtual_box, nh['track_id'], frame_idx):
                        saved_path = self.saver.save_crop_image(
                            frame, virtual_box, "violation_rider_fallback", 
                            frame_idx, video_time_str, nh['track_id'], h_conf
                        )
                        if saved_path:
                            total_saved_violations += 1
                    
                    # Draw virtual box and link
                    self.annotator.draw_semi_transparent_rect(annotated_frame, virtual_box[0], virtual_box[1], virtual_box[2], virtual_box[3], COLOR_VIOLATION, 0.15)
                    cv2.rectangle(annotated_frame, (virtual_box[0], virtual_box[1]), (virtual_box[2], virtual_box[3]), COLOR_VIOLATION, thick_head, lineType=cv2.LINE_AA)
                    cv2.rectangle(annotated_frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), COLOR_VIOLATION, thick_head)
                    
                    self.annotator.draw_dotted_line(
                        annotated_frame, 
                        (int((h_box[0]+h_box[2])/2), int((h_box[1]+h_box[3])/2)), 
                        (int((virtual_box[0]+virtual_box[2])/2), int((virtual_box[1]+virtual_box[3])/2)), 
                        COLOR_VIOLATION, thick_head
                    )
                    self.annotator.draw_premium_tag(annotated_frame, f"NO HELMET {h_conf:.2f}", int(h_box[0]), int(h_box[3]), COLOR_VIOLATION, COLOR_TEXT, font_scale=0.45, position='bottom')
                    self.annotator.draw_premium_tag(annotated_frame, "ESTIMATED VIOLATION VEHICLE", virtual_box[0], virtual_box[1], COLOR_VIOLATION, COLOR_TEXT, font_scale=0.45)
                else:
                    # Draw unmatched low-confidence no helmet box in gray/white
                    cv2.rectangle(annotated_frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), COLOR_UNKNOWN, thick_head)
                    self.annotator.draw_premium_tag(annotated_frame, f"No Helmet {h_conf:.2f}", int(h_box[0]), int(h_box[3]), COLOR_UNKNOWN, COLOR_TEXT, font_scale=0.38, position='bottom')

            # 3. Draw unmatched helmets
            for h in helmets:
                if id(h) in matched_heads_indices:
                    continue
                h_box = h['box']
                h_conf = h['conf']
                cv2.rectangle(annotated_frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), COLOR_COMPLIANT, thick_head)
                self.annotator.draw_premium_tag(annotated_frame, f"Helmet {h_conf:.2f}", int(h_box[0]), int(h_box[3]), COLOR_COMPLIANT, COLOR_TEXT, font_scale=0.38, position='bottom')

            # Draw HUD overlay
            fps_val = 1.0 / (time.time() - start_time) if frame_idx > 1 else None
            self.annotator.draw_hud(annotated_frame, violation_detected, active_violations_in_frame, total_saved_violations, fps=fps_val)
            
            # Reset start_time for next frame's FPS estimation
            start_time = time.time()
            
            # Write the annotated frame
            out.write(annotated_frame)
            pbar.update(1)
            
        cap.release()
        out.release()
        pbar.close()
        
        print(f"\nProcessing finished.")
        print(f"Annotated video saved: {self.config.output_path}")
        if self.config.save_crops:
            print(f"Saved {total_saved_violations} crops inside {self.save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biker POV Helmet Violation Detection and Video Annotation")
    parser.add_argument("--video", "-v", required=True, type=str, help="Path to input video file")
    parser.add_argument("--model", "-m", default=r"runs\detect\train\weights\best.pt", type=str, help="Path to RT-DETR weights")
    parser.add_argument("--output", "-o", type=str, help="Path to save annotated video")
    parser.add_argument("--crop-dir", "-c", default="crops", type=str, help="Parent directory to save cropped violation images")
    parser.add_argument("--conf", type=float, default=0.80, help="Confidence threshold for no-helmet detection to trigger violations")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Confidence threshold for object detection model")
    parser.add_argument("--iou-match", type=float, default=0.80, help="Minimum IoA (Intersection over Area) to match head/rider to vehicle")
    
    # Boolean flag parsing
    parser.add_argument('--save-crops', action='store_true', default=False, help="Enable saving cropped violation images")
    parser.add_argument('--no-save-crops', action='store_false', dest='save_crops', help="Disable saving cropped violation images")
    
    parser.add_argument('--show-compliant', action='store_true', default=True, help="Draw boxes for compliant riders")
    parser.add_argument('--no-show-compliant', action='store_false', dest='show_compliant', help="Do not draw boxes for compliant riders")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = DetectionConfig(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        crop_dir=args.crop_dir,
        conf_threshold=args.conf,
        det_conf=args.det_conf,
        iou_match_threshold=args.iou_match,
        save_crops=args.save_crops,
        show_compliant=args.show_compliant
    )
    processor = VideoProcessor(config)
    processor.process()


if __name__ == "__main__":
    main()
