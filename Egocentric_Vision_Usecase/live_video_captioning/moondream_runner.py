#!/usr/bin/env python3
"""
Moondream Runner (Image & Video)
A CLI utility to perform image captioning and sequential video analysis locally
using OpenCV and Ollama's Moondream (1.8b) model.
"""

import argparse
import os
import sys
import json
import cv2
import ollama
from tqdm import tqdm
from PIL import Image
import threading
import time
import datetime

# Terminal colors for professional look
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'

def print_banner():
    # Force stdout encoding to UTF-8 on Windows terminal if needed to prevent codec errors
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    banner = f"""
{Colors.CYAN}{Colors.BOLD}======================================================================
     *** MOONDREAM LOCAL IMAGE & VIDEO RUNNER (via Ollama) ***
======================================================================{Colors.ENDC}
"""
    print(banner)

def verify_ollama_model(model_name="moondream"):
    """Verify if the requested model is pulled in Ollama."""
    try:
        response = ollama.list()
        models = response.models if hasattr(response, 'models') else response.get('models', [])
        for model in models:
            name = getattr(model, 'model', '') if not isinstance(model, dict) else model.get('model', model.get('name', ''))
            if model_name.lower() in name.lower():
                return True
        return False
    except Exception as e:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Failed to connect to Ollama service: {e}")
        print(f"{Colors.WARNING}[Tip]{Colors.ENDC} Make sure the Ollama application is running on your system.")
        sys.exit(1)

def get_model_processor_status(model_name="moondream"):
    """Query Ollama's active state to find out if the model is loaded on GPU or CPU."""
    try:
        response = ollama.ps()
        models = response.models if hasattr(response, 'models') else response.get('models', [])
        for model in models:
            name = getattr(model, 'model', '') if not isinstance(model, dict) else model.get('model', model.get('name', ''))
            if model_name.lower() in name.lower():
                size = getattr(model, 'size', 0) if not isinstance(model, dict) else model.get('size', 0)
                size_vram = getattr(model, 'size_vram', 0) if not isinstance(model, dict) else model.get('size_vram', 0)
                
                if size == 0:
                    return "CPU/GPU (Undetermined)"
                    
                gpu_percent = (size_vram / size) * 100
                if gpu_percent >= 99.9:
                    return f"100% GPU (VRAM: {size_vram / (1024**3):.2f} GB)"
                elif gpu_percent > 0:
                    return f"Hybrid (GPU: {gpu_percent:.1f}%, CPU: {100 - gpu_percent:.1f}%)"
                    
                return "100% CPU"
        return "Unknown (Not loaded in memory)"
    except Exception:
        return "Unknown (Failed to query Ollama process)"

def analyze_image(image_path, prompt, model_name="moondream"):
    """Analyze a single static image using Ollama's Moondream model."""
    if not os.path.exists(image_path):
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Image file '{image_path}' not found.")
        sys.exit(1)

    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Analyzing image: '{image_path}'")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Prompt: '{prompt}'\n")

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ]
        )
        description = response.get('message', {}).get('content', '')
        
        # Display execution processor device status
        processor = get_model_processor_status(model_name)
        print(f"{Colors.BLUE}[Info]{Colors.ENDC} Model hardware device: {Colors.BOLD}{Colors.GREEN}{processor}{Colors.ENDC}\n")
        
        print(f"{Colors.GREEN}{Colors.BOLD}--- Analysis Result ---{Colors.ENDC}")
        print(description)
        print(f"{Colors.GREEN}{Colors.BOLD}-----------------------{Colors.ENDC}\n")
        return {"input_file": image_path, "prompt": prompt, "analysis": description}
    except Exception as e:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Failed during image analysis: {e}")
        sys.exit(1)

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def draw_caption_box(img, text, max_width_percent=0.45, font_scale=None, thickness=None):
    """Draw a semi-transparent background box with word-wrapped caption text in upper-right corner."""
    h, w, _ = img.shape
    
    # Calculate scale factor relative to standard 720p height
    scale_factor = h / 720.0
    
    # Configure boundaries and font
    max_w = int(w * max_width_percent)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Determine dynamic font scale and thickness if not provided
    if font_scale is None:
        font_scale = max(0.5, 0.75 * scale_factor)
    if thickness is None:
        thickness = max(1, int(2.0 * scale_factor))
        
    # Wrap text into lines of max width
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        (line_w, line_h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if line_w > max_w:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
                current_line = ""
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
        
    # If no lines, we display nothing
    if not lines:
        return
        
    # Calculate height and width of the box dynamically scaled
    spacing = max(4, int(8 * scale_factor))
    line_height = 0
    max_line_width = 0
    
    # Get dimensions for each line
    line_dims = []
    for line in lines:
        (line_w, line_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_height = max(line_height, line_h + baseline)
        max_line_width = max(max_line_width, line_w)
        line_dims.append((line_w, line_h))
        
    total_height = len(lines) * (line_height + spacing) - spacing
    
    # Margins and paddings dynamically scaled
    padding = max(6, int(12 * scale_factor))
    margin_top = max(10, int(20 * scale_factor))
    margin_right = max(10, int(20 * scale_factor))
    
    box_w = max_line_width + 2 * padding
    box_h = total_height + 2 * padding
    
    # Coordinates in top right
    x2 = w - margin_right
    x1 = x2 - box_w
    y1 = margin_top
    y2 = y1 + box_h
    
    # Ensure within frame bounds
    if x1 < 0: x1 = 0
    if y2 > h: y2 = h
    
    # Draw semi-transparent background box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1) # Fill black
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img) # Blend overlay
    
    # Draw border for elegant tech look (Cyan border in BGR is (255, 255, 0))
    border_thickness = max(1, int(1.5 * scale_factor))
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), border_thickness)
    
    # Draw text lines
    vertical_offset = max(1, int(3 * scale_factor))
    current_y = y1 + padding + line_height - vertical_offset
    for line in lines:
        cv2.putText(img, line, (x1 + padding, current_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        current_y += line_height + spacing

def analyze_video(video_path, prompt, interval=1.0, model_name="moondream", save_video_path=None):
    """Analyze a video frame-by-frame at specific time intervals."""
    if not os.path.exists(video_path):
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Video file '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Could not open video file '{video_path}' using OpenCV.")
        sys.exit(1)

    # Gather video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Safety checks for video parameters
    if fps <= 0 or total_frames <= 0:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Invalid video metadata (FPS: {fps}, Total Frames: {total_frames}).")
        cap.release()
        sys.exit(1)

    duration = total_frames / fps
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Video duration: {duration:.2f}s | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Sampling interval: every {interval:.2f} second(s)")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Prompt per frame: '{prompt}'\n")

    timeline = []
    
    # Calculate frames to sample based on the interval
    frame_step = int(fps * interval)
    if frame_step < 1:
        frame_step = 1

    frames_to_process = list(range(0, total_frames, frame_step))
    
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Processing {len(frames_to_process)} frames...")
    
    # Progress bar
    pbar = tqdm(frames_to_process, desc="Captioning video frames", unit="frame")
    
    for frame_idx in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # Reached end of video or read error
            break
            
        timestamp_sec = frame_idx / fps
        timestamp_str = format_timestamp(timestamp_sec)
        
        # Update progress bar description
        pbar.set_postfix_str(f"Time: {timestamp_str[:-4]}")
        
        # Compress frame to JPEG bytes in-memory to send to Ollama API
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print(f"\n{Colors.WARNING}[Warning]{Colors.ENDC} Failed to encode frame at {timestamp_str}, skipping.")
            continue
            
        frame_bytes = buffer.tobytes()
        
        try:
            # Query local Ollama moondream model
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [frame_bytes]
                    }
                ]
            )
            caption = response.get('message', {}).get('content', '').strip()
            
            timeline.append({
                "frame": frame_idx,
                "timestamp_sec": timestamp_sec,
                "timestamp": timestamp_str,
                "caption": caption
            })
            
            # Print execution device info upon first frame inference loading
            if len(timeline) == 1:
                processor = get_model_processor_status(model_name)
                print(f"\n{Colors.BLUE}[Info]{Colors.ENDC} Model hardware device: {Colors.BOLD}{Colors.GREEN}{processor}{Colors.ENDC}")
            
        except Exception as e:
            print(f"\n{Colors.FAIL}[Error]{Colors.ENDC} Failed to process frame at {timestamp_str}: {e}")
            # Continue processing subsequent frames

    print(f"\n{Colors.GREEN}{Colors.BOLD}=== Sequential Video Analysis Timeline ==={Colors.ENDC}")
    for entry in timeline:
        print(f"[{Colors.CYAN}{entry['timestamp'][:-4]}{Colors.ENDC}] {entry['caption']}")
    print(f"{Colors.GREEN}{Colors.BOLD}=========================================={Colors.ENDC}\n")

    # Generate captioned video overlay if requested
    if save_video_path and timeline:
        print(f"{Colors.BLUE}[Info]{Colors.ENDC} Generating captioned video output: '{save_video_path}'...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
        
        active_caption = "Analyzing..."
        timeline_idx = 0
        
        # Progress bar for video writing
        write_pbar = tqdm(range(total_frames), desc="Writing output video", unit="frame")
        
        for frame_idx in write_pbar:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update current active caption based on frame timeline index
            # If the current frame has reached or passed the frame_idx in timeline, update caption
            while (timeline_idx < len(timeline) - 1 and 
                   frame_idx >= timeline[timeline_idx + 1]['frame']):
                timeline_idx += 1
            
            if timeline_idx < len(timeline) and frame_idx >= timeline[timeline_idx]['frame']:
                active_caption = timeline[timeline_idx]['caption']
                
            # Draw overlay box in top-right corner
            draw_caption_box(frame, active_caption)
            
            # Write to file
            out_writer.write(frame)
            
        out_writer.release()
        print(f"{Colors.GREEN}[Success]{Colors.ENDC} Captioned video successfully saved to '{save_video_path}'\n")

    cap.release()

    return {
        "input_file": video_path,
        "video_duration_seconds": duration,
        "sampling_interval_seconds": interval,
        "prompt": prompt,
        "timeline": timeline
    }

def analyze_webcam(camera_input, prompt, interval=2.0, model_name="moondream", save_video_path=None):
    """Analyze a live webcam feed in real-time with background thread inference and a sci-fi HUD overlay."""
    import time
    import threading
    import datetime

    # Parse camera input
    # If it is a string representation of an integer, convert it
    if isinstance(camera_input, str) and camera_input.isdigit():
        camera_source = int(camera_input)
    else:
        camera_source = camera_input

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Could not open webcam source '{camera_input}'.")
        sys.exit(1)

    # Try to set typical resolution for speed and quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read a frame to get actual dimensions
    ret, frame = cap.read()
    if not ret:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} Failed to read frame from webcam.")
        cap.release()
        sys.exit(1)

    h, w, _ = frame.shape
    fps_feed = cap.get(cv2.CAP_PROP_FPS)
    if fps_feed <= 0:
        fps_feed = 30.0 # Fallback

    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Webcam resolution: {w}x{h} | Target FPS: {fps_feed:.1f}")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Auto-caption interval: every {interval:.2f} second(s)")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Prompt: '{prompt}'")
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Controls: Press '{Colors.BOLD}q{Colors.ENDC}' to exit, '{Colors.BOLD}c{Colors.ENDC}' to caption now, '{Colors.BOLD}s{Colors.ENDC}' to toggle auto-caption.\n")

    # Shared thread states
    lock = threading.Lock()
    latest_caption = "Initializing system..."
    is_inferring = False
    inference_latency = 0.0
    timeline = []
    running = True
    auto_caption = True

    # Get active hardware processor device
    processor_device = get_model_processor_status(model_name)

    # Initialize VideoWriter if save_video_path is set
    out_writer = None
    if save_video_path:
        print(f"{Colors.BLUE}[Info]{Colors.ENDC} Saving recorded stream with HUD overlays to '{save_video_path}'...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_video_path, fourcc, fps_feed, (w, h))

    # Helper function for background inference
    def run_inference(img_copy):
        nonlocal latest_caption, is_inferring, inference_latency
        start_time = time.time()
        try:
            success, buffer = cv2.imencode('.jpg', img_copy)
            if not success:
                return
            frame_bytes = buffer.tobytes()
            
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [frame_bytes]
                    }
                ]
            )
            caption = response.get('message', {}).get('content', '').strip()
            latency = time.time() - start_time
            
            with lock:
                latest_caption = caption
                inference_latency = latency
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                timeline.append({
                    "timestamp": now_str,
                    "caption": caption,
                    "latency": latency
                })
                # Print real-time caption in terminal
                print(f"[{Colors.CYAN}{now_str}{Colors.ENDC}] {caption} ({Colors.GREEN}{latency:.2f}s{Colors.ENDC})")
        except Exception as e:
            with lock:
                latest_caption = f"Error during analysis: {str(e)}"
        finally:
            with lock:
                is_inferring = False

    # Scanning line state
    scan_y = 20
    scan_direction = 1
    scan_speed = 3
    margin = 20

    # FPS counter variables
    frame_count = 0
    start_fps_time = time.time()
    fps_display = 0.0

    last_inference_trigger_time = 0.0

    # OpenCV window creation
    window_name = "Moondream Live Webcam Runner (Ollama)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n{Colors.WARNING}[Warning]{Colors.ENDC} Webcam feed interrupted.")
                break

            # Clone frame for overlays so we don't pollute the source if writing raw
            display_frame = frame.copy()

            current_time = time.time()

            # FPS calculation
            frame_count += 1
            elapsed_fps_time = current_time - start_fps_time
            if elapsed_fps_time >= 1.0:
                fps_display = frame_count / elapsed_fps_time
                frame_count = 0
                start_fps_time = current_time

            # Check and trigger auto-captioning
            with lock:
                currently_inferring = is_inferring
                should_trigger = auto_caption and (not currently_inferring) and (current_time - last_inference_trigger_time >= interval)

            if should_trigger:
                last_inference_trigger_time = current_time
                with lock:
                    is_inferring = True
                # Copy frame and spawn thread
                threading.Thread(target=run_inference, args=(frame.copy(),), daemon=True).start()

            # --- DRAW PREMIUM SCI-FI HUD OVERLAYS ---
            # 1. Overlay Blend layers for transparency
            hud_overlay = display_frame.copy()

            # 2. Outer corner brackets (Sci-Fi Viewfinder)
            corner_len = 25
            corner_color = (255, 255, 0) # Cyan in BGR
            # Top-Left
            cv2.line(display_frame, (margin, margin), (margin + corner_len, margin), corner_color, 2)
            cv2.line(display_frame, (margin, margin), (margin, margin + corner_len), corner_color, 2)
            # Top-Right
            cv2.line(display_frame, (w - margin, margin), (w - margin - corner_len, margin), corner_color, 2)
            cv2.line(display_frame, (w - margin, margin), (w - margin, margin + corner_len), corner_color, 2)
            # Bottom-Left
            cv2.line(display_frame, (margin, h - margin), (margin + corner_len, h - margin), corner_color, 2)
            cv2.line(display_frame, (margin, h - margin), (margin, h - margin - corner_len), corner_color, 2)
            # Bottom-Right
            cv2.line(display_frame, (w - margin, h - margin), (w - margin - corner_len, h - margin), corner_color, 2)
            cv2.line(display_frame, (w - margin, h - margin), (w - margin, h - margin - corner_len), corner_color, 2)

            # 3. Scanning horizontal bar
            scan_y += scan_speed * scan_direction
            if scan_y >= h - margin:
                scan_y = h - margin
                scan_direction = -1
            elif scan_y <= margin:
                scan_y = margin
                scan_direction = 1

            # Glow scanner bar
            cv2.line(hud_overlay, (margin, int(scan_y)), (w - margin, int(scan_y)), (255, 255, 0), 4) # Thick glow
            cv2.line(display_frame, (margin, int(scan_y)), (w - margin, int(scan_y)), (255, 255, 255), 1) # Sharp inner line

            # 4. Performance Statistics HUD (Top-Left Box)
            tl_x1, tl_y1 = margin + 5, margin + 5
            tl_w, tl_h = 240, 115
            cv2.rectangle(hud_overlay, (tl_x1, tl_y1), (tl_x1 + tl_w, tl_y1 + tl_h), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (tl_x1, tl_y1), (tl_x1 + tl_w, tl_y1 + tl_h), (255, 255, 0), 1) # Cyan border

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, "SYSTEM OVERVIEW", (tl_x1 + 10, tl_y1 + 20), font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(display_frame, (tl_x1 + 10, tl_y1 + 26), (tl_x1 + tl_w - 10, tl_y1 + 26), (255, 255, 0), 1)

            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (tl_x1 + 10, tl_y1 + 45), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"MODEL: {model_name}", (tl_x1 + 10, tl_y1 + 62), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"DEVICE: {processor_device}", (tl_x1 + 10, tl_y1 + 79), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            with lock:
                lat = inference_latency
            lat_str = f"{lat:.2f}s" if lat > 0 else "WAITING"
            cv2.putText(display_frame, f"LATENCY: {lat_str}", (tl_x1 + 10, tl_y1 + 96), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Controls HUD (Bottom-Left Box)
            bl_x1, bl_y1 = margin + 5, h - margin - 50
            bl_w, bl_h = 360, 45
            cv2.rectangle(hud_overlay, (bl_x1, bl_y1), (bl_x1 + bl_w, bl_y1 + bl_h), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (bl_x1, bl_y1), (bl_x1 + bl_w, bl_y1 + bl_h), (255, 255, 0), 1)

            auto_status = "ON" if auto_caption else "OFF"
            auto_color = (0, 255, 0) if auto_caption else (0, 0, 255) # Green / Red
            cv2.putText(display_frame, "[Q] Quit  |  [C] Caption Now  |  [S] Auto-Caption:", (bl_x1 + 10, bl_y1 + 26), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, auto_status, (bl_x1 + 312, bl_y1 + 26), font, 0.38, auto_color, 1, cv2.LINE_AA)

            # 6. Inference Pulse indicator (Bottom-Right Box)
            br_x1, br_y1 = w - margin - 155, h - margin - 40
            br_w, br_h = 150, 35
            cv2.rectangle(hud_overlay, (br_x1, br_y1), (br_x1 + br_w, br_y1 + br_h), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (br_x1, br_y1), (br_x1 + br_w, br_y1 + br_h), (255, 255, 0), 1)

            with lock:
                currently_inferring = is_inferring

            if currently_inferring:
                dot_color = (0, 165, 255) if (int(current_time * 4) % 2 == 0) else (0, 80, 120)
                status_text = "INFERRING..."
                text_color = (0, 165, 255)
            else:
                dot_color = (0, 255, 0)
                status_text = "READY / IDLE"
                text_color = (0, 255, 0)

            cv2.circle(display_frame, (br_x1 + 15, br_y1 + 17), 5, dot_color, -1)
            cv2.putText(display_frame, status_text, (br_x1 + 30, br_y1 + 22), font, 0.4, text_color, 1, cv2.LINE_AA)

            # 7. Word-wrapped Caption Box in Top-Right
            with lock:
                cap_text = latest_caption
            draw_caption_box(display_frame, cap_text, max_width_percent=0.45)

            # Blend the HUD layers
            cv2.addWeighted(hud_overlay, 0.45, display_frame, 0.55, 0, display_frame)

            # Show frame
            cv2.imshow(window_name, display_frame)

            # Write to output video file if enabled
            if out_writer:
                out_writer.write(display_frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n{Colors.BLUE}[Info]{Colors.ENDC} Exiting webcam session...")
                break
            elif key == ord('c'):
                with lock:
                    currently_inferring = is_inferring
                if currently_inferring:
                    print(f"\n{Colors.WARNING}[Warning]{Colors.ENDC} Model is busy processing a frame. Please wait.")
                else:
                    print(f"\n{Colors.BLUE}[Info]{Colors.ENDC} Manual caption trigger activated...")
                    with lock:
                        is_inferring = True
                    last_inference_trigger_time = current_time
                    threading.Thread(target=run_inference, args=(frame.copy(),), daemon=True).start()
            elif key == ord('s'):
                with lock:
                    auto_caption = not auto_caption
                    print(f"\n{Colors.BLUE}[Info]{Colors.ENDC} Auto-Caption toggled: {Colors.BOLD}{'ENABLED' if auto_caption else 'DISABLED'}{Colors.ENDC}")

    except KeyboardInterrupt:
        print(f"\n{Colors.BLUE}[Info]{Colors.ENDC} Webcam session interrupted by keyboard.")
    finally:
        with lock:
            running = False
        cap.release()
        if out_writer:
            out_writer.release()
            print(f"{Colors.GREEN}[Success]{Colors.ENDC} Webcam recording saved to '{save_video_path}'")
        cv2.destroyAllWindows()

    return {
        "input_file": f"webcam_{camera_input}",
        "sampling_interval_seconds": interval,
        "prompt": prompt,
        "timeline": timeline
    }

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Local Moondream runner for captioning and analyzing images, videos, and webcam feeds using Ollama."
    )
    parser.add_argument(
        '--mode',
        choices=['image', 'video', 'webcam'],
        required=True,
        help="Select execution mode: 'image' to analyze a single photo, 'video' to caption sequential frames, or 'webcam' for real-time video."
    )
    parser.add_argument(
        '--input',
        required=False,
        help="Path to the input file (image/video) or camera index/RTSP stream for webcam mode (default: '0')."
    )
    parser.add_argument(
        '--prompt',
        default="Describe this image.",
        help="Prompt / question to send to the Moondream model (default: 'Describe this image.')"
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help="Time interval in seconds between sampled frames (default: 1.0s)."
    )
    parser.add_argument(
        '--output',
        help="Path to save the output results as a JSON or TXT file."
    )
    parser.add_argument(
        '--save-video',
        help="Optional path to save recorded/captioned output video."
    )
    parser.add_argument(
        '--model',
        default="moondream",
        help="Ollama model tag to use (default: 'moondream')."
    )

    args = parser.parse_args()

    # Mode validation for --input
    if args.mode in ['image', 'video'] and not args.input:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} --input is required when mode is 'image' or 'video'.")
        sys.exit(1)

    # Set default input for webcam mode
    input_source = args.input if args.input else '0'

    # Validate interval
    if args.interval <= 0:
        print(f"{Colors.FAIL}[Error]{Colors.ENDC} --interval must be a positive number greater than 0.")
        sys.exit(1)

    # 1. Verify model is available in Ollama
    print(f"{Colors.BLUE}[Info]{Colors.ENDC} Verifying Ollama model '{args.model}'...")
    if not verify_ollama_model(args.model):
        print(f"{Colors.WARNING}[Warning]{Colors.ENDC} Model '{args.model}' not found in Ollama.")
        print(f"{Colors.BLUE}[Info]{Colors.ENDC} Pulling model '{args.model}'...")
        try:
            ollama.pull(args.model)
        except Exception as e:
            print(f"{Colors.FAIL}[Error]{Colors.ENDC} Failed to pull model '{args.model}': {e}")
            sys.exit(1)

    # 2. Run analysis
    results = None
    if args.mode == 'image':
        results = analyze_image(args.input, args.prompt, model_name=args.model)
    elif args.mode == 'video':
        results = analyze_video(
            args.input, 
            args.prompt, 
            interval=args.interval, 
            model_name=args.model, 
            save_video_path=args.save_video
        )
    elif args.mode == 'webcam':
        results = analyze_webcam(
            input_source,
            args.prompt,
            interval=args.interval,
            model_name=args.model,
            save_video_path=args.save_video
        )

    # 3. Output saving
    if args.output and results:
        try:
            _, ext = os.path.splitext(args.output)
            if ext.lower() == '.json':
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            else:
                # Text format output
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Moondream Local Analysis Results\n")
                    f.write(f"================================\n")
                    f.write(f"Mode: {args.mode.upper()}\n")
                    f.write(f"Input Source: {results['input_file']}\n")
                    f.write(f"Prompt: {results['prompt']}\n\n")
                    
                    if args.mode == 'image':
                        f.write(f"Result:\n{results['analysis']}\n")
                    elif args.mode == 'video':
                        f.write(f"Video Duration: {results['video_duration_seconds']:.2f}s\n")
                        f.write(f"Sampling Interval: {results['sampling_interval_seconds']:.2f}s\n\n")
                        f.write(f"Timeline:\n")
                        for entry in results['timeline']:
                            f.write(f"[{entry['timestamp'][:-4]}] {entry['caption']}\n")
                    elif args.mode == 'webcam':
                        f.write(f"Sampling Interval: {results['sampling_interval_seconds']:.2f}s\n\n")
                        f.write(f"Timeline:\n")
                        for entry in results['timeline']:
                            f.write(f"[{entry['timestamp']}] {entry['caption']} (latency: {entry['latency']:.2f}s)\n")
                            
            print(f"{Colors.GREEN}[Success]{Colors.ENDC} Saved analysis results to: '{args.output}'")
        except Exception as e:
            print(f"{Colors.FAIL}[Error]{Colors.ENDC} Failed to save results to '{args.output}': {e}")

if __name__ == '__main__':
    main()
