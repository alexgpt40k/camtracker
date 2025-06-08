# Stable Version 1.1.20
# Generated on: 2025-05-05 UTC+0300
# Changelog:
# - Restored full calibration workflow with sample averaging
# - Added file existence checks and selected monitors loading
# - Integrated font initialization and utility functions
# - Ensured frame resizing and correct color rendering
# - Corrected monitor-specific positioning during calibration

import cv2
import math
import time
import json
import os
import tkinter as tk
import numpy as np
import logging
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks.python import vision
from screeninfo import get_monitors

# Configure logging
logging.basicConfig(filename='calibration.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Version metadata
VERSION = "1.1.20"
GEN_TIME = time.strftime('%Y-%m-%d %H:%M:%S UTC+0300', time.localtime())

# Paths
MODEL_PATH      = "face_landmarker.task"
SELECTED_PATH   = "selected_monitors.json"
CALIB_PREFIX    = "calibration_monitor_"
CALIB_SUFFIX    = ".json"
FONT_PATH       = "C:/Windows/Fonts/arial.ttf"

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load selected monitors
if not os.path.exists(SELECTED_PATH):
    raise FileNotFoundError(f"Please select monitors first: {SELECTED_PATH}")
with open(SELECTED_PATH, 'r', encoding='utf-8') as f:
    selected_monitors = json.load(f)
if not isinstance(selected_monitors, list) or not selected_monitors:
    raise RuntimeError("Selected monitors list is empty or invalid")

# Prepare font for drawing text
font = ImageFont.truetype(FONT_PATH, 20)

# MediaPipe face landmarker setup
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode     = vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Calibration corners with descriptions
corners = [
    ("lt", "левый верхний угол"),
    ("rt", "правый верхний угол"),
    ("rb", "правый нижний угол"),
    ("lb", "левый нижний угол"),
]

# Retrieve monitor geometries
monitors = get_monitors()

# Compute yaw and pitch from landmarks
def get_yaw_pitch(lm):
    nt, nb = lm[1], lm[168]
    dx, dy = nt.x - nb.x, nt.y - nb.y
    yaw = math.degrees(math.atan2(dx, dy))
    pitch = math.degrees(math.atan2(dy, 0.01))
    return yaw, pitch

# Utility: overlay Unicode text onto OpenCV frame
def put_unicode_text(frame, text, pos, color=(255,255,0)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# Main calibration function
def calibrate_monitor(root, monitor_id, canvas=None, img_id=None):
    logging.info(f"Starting calibration for monitor {monitor_id}")

    # Bind Enter key for advancing corners
    enter_flag = {'pressed': False}
    def on_enter(e):
        enter_flag['pressed'] = True
    root.bind('<Return>', on_enter)

    # Open camera and landmarker
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera for calibration")
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Get monitor geometry for target monitor
    mon = monitors[monitor_id]
    mx, my, mw, mh = mon.x, mon.y, mon.width, mon.height

    # Get window size (assumes root has been sized to camera resolution)
    ww, wh = root.winfo_width(), root.winfo_height()

    all_results = {}

    for corner_key, corner_name in corners:
        # Position window at specified corner of the target monitor
        positions = {
            'lt': (mx, my),
            'rt': (mx + mw - ww, my),
            'rb': (mx + mw - ww, my + mh - wh),
            'lb': (mx, my + mh - wh)
        }
        x, y = positions[corner_key]
        root.geometry(f'+{x}+{y}')
        root.update()

        instruction = f"Монитор {monitor_id}: посмотрите в {corner_name} и нажмите Enter"
        enter_flag['pressed'] = False

        # Live preview loop: show frame + instruction in Canvas
        while not enter_flag['pressed']:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            disp = put_unicode_text(frame, instruction, (10, frame.shape[0] - 50))

            if canvas and img_id is not None:
                rgb_disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_disp))
                canvas.itemconfig(img_id, image=imgtk)
                canvas.imgtk = imgtk
            else:
                cv2.imshow('Calibration', disp)

            root.update_idletasks()
            root.update()
            time.sleep(0.03)

        # Collect samples for this corner
        samples = []
        for _ in range(30):
            ret, frame2 = cap.read()
            if not ret:
                continue
            frame2 = cv2.flip(frame2, 1)
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            mp2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
            r2 = landmarker.detect_for_video(mp2, int(time.time() * 1000))
            if r2.face_landmarks:
                y2, p2 = get_yaw_pitch(r2.face_landmarks[0])
                samples.append({'yaw': y2, 'pitch': p2})
            time.sleep(0.03)

        # Compute average yaw/pitch
        if samples:
            avg_yaw   = sum(s['yaw'] for s in samples) / len(samples)
            avg_pitch = sum(s['pitch'] for s in samples) / len(samples)
        else:
            avg_yaw   = 0.0
            avg_pitch = 0.0
        all_results[corner_key] = {'yaw': avg_yaw, 'pitch': avg_pitch}
        logging.info(f"Corner {corner_key}: yaw={avg_yaw:.2f}, pitch={avg_pitch:.2f}")

        if not canvas or img_id is None:
            cv2.destroyAllWindows()

    # Recenter window to center of target monitor
    center_x = mx + (mw - ww) // 2
    center_y = my + (mh - wh) // 2
    root.geometry(f'+{center_x}+{center_y}')
    root.update()

    # Force one frame update to clear 'frozen' calibration image
    if canvas and img_id is not None:
        ret, frame_fresh = cap.read()
        if ret:
            frame_fresh = cv2.flip(frame_fresh, 1)
            rgb_fresh = cv2.cvtColor(frame_fresh, cv2.COLOR_BGR2RGB)
            imgtk_fresh = ImageTk.PhotoImage(Image.fromarray(rgb_fresh))
            canvas.itemconfig(img_id, image=imgtk_fresh)
            canvas.imgtk = imgtk_fresh
            root.update_idletasks()
            root.update()

    # Cleanup resources
    cap.release()
    landmarker.close()
    root.unbind('<Return>')

    # Save calibration data
    out_path = f"{CALIB_PREFIX}{monitor_id}{CALIB_SUFFIX}"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logging.info(f"Calibration saved to {out_path}")

    return all_results
