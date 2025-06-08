# Stable Version 1.1.21
# Generated on: 2025-06-01 UTC+0300
# Changelog:
# - Калибровка “на месте”: рисуем полупрозрачный оверлей с инструкцией в углу основного Canvas
# - Убрано перемещение всего окна, отображение инструкции поверх видео
# - Сбор и усреднение 30 сэмплов yaw/pitch для каждого угла
# - После каждого угла оверлей автоматически скрывается, окно не «зависает»

import cv2
import math
import time
import json
import os
import logging
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks.python import vision

# Configure logging
logging.basicConfig(
    filename='calibration2.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Version metadata
VERSION = "1.1.21"
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

# Compute yaw and pitch from landmarks
def get_yaw_pitch(lm):
    nt, nb = lm[1], lm[168]
    dx, dy = nt.x - nb.x, nt.y - nb.y
    yaw   = math.degrees(math.atan2(dx, dy))
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
def calibrate_monitor(root, monitor_id, canvas, img_id):
    logging.info(f"Starting calibration2 for monitor {monitor_id}")

    # Bind Enter key for advancing corners
    enter_flag = {'pressed': False}
    def on_enter(e):
        enter_flag['pressed'] = True
    root.bind('<Return>', on_enter)

    # Open camera separately for calibration
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera for calibration")
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Determine canvas size (должно совпадать с 800×600)
    cw = int(canvas['width'])
    ch = int(canvas['height'])

    # Retrieve calibration results
    all_results = {}

    for corner_key, corner_name in corners:
        # Build semi-transparent overlay rectangle + instruction text
        sq_w, sq_h = 200, 60
        pad = 10

        if corner_key == 'lt':
            ox, oy = pad, pad
        elif corner_key == 'rt':
            ox, oy = cw - sq_w - pad, pad
        elif corner_key == 'rb':
            ox, oy = cw - sq_w - pad, ch - sq_h - pad
        else:  # 'lb'
            ox, oy = pad, ch - sq_h - pad

        text_pos = (ox + 5, oy + 20)
        instr = f"Монитор {monitor_id}: посмотрите в {corner_name} и нажмите Enter"

        # Create semi-transparent overlay image
        overlay = Image.new('RGBA', (sq_w, sq_h), (0, 0, 0, 128))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([(0, 0), (sq_w, sq_h)], fill=(0, 0, 0, 128))
        pil_img = ImageTk.PhotoImage(overlay)

        # Draw overlay and text on Canvas
        rect_tag = f"overlay_rect_{corner_key}"
        text_tag = f"overlay_text_{corner_key}"
        overlay_id = canvas.create_image(
            ox, oy, image=pil_img, anchor='nw', tags=(rect_tag,)
        )
        canvas.image = pil_img
        txt_id = canvas.create_text(
            text_pos[0], text_pos[1],
            text=instr,
            fill='white',
            font=("Helvetica", 12, "bold"),
            anchor='nw',
            tags=(text_tag,)
        )
        canvas.tag_raise(rect_tag)
        canvas.tag_raise(text_tag)

        root.update_idletasks()
        root.update()

        # Wait for Enter
        enter_flag['pressed'] = False
        while not enter_flag['pressed']:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.itemconfig(img_id, image=imgtk)
            canvas.imgtk = imgtk

            root.update_idletasks()
            root.update()
            time.sleep(0.03)

        # Remove overlay
        canvas.delete(rect_tag)
        canvas.delete(text_tag)

        # Collect 30 samples for this corner
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
            root.update_idletasks()
            root.update()
            time.sleep(0.03)

        if samples:
            avg_yaw   = sum(s['yaw'] for s in samples) / len(samples)
            avg_pitch = sum(s['pitch'] for s in samples) / len(samples)
        else:
            avg_yaw   = 0.0
            avg_pitch = 0.0
        all_results[corner_key] = {'yaw': avg_yaw, 'pitch': avg_pitch}
        logging.info(f"Calibrate2 corner {corner_key}: yaw={avg_yaw:.2f}, pitch={avg_pitch:.2f}")

    # Show one last fresh frame
    ret, frame_last = cap.read()
    if ret:
        frame_last = cv2.flip(frame_last, 1)
        rgb_last = cv2.cvtColor(frame_last, cv2.COLOR_BGR2RGB)
        imgtk_last = ImageTk.PhotoImage(Image.fromarray(rgb_last))
        canvas.itemconfig(img_id, image=imgtk_last)
        canvas.imgtk = imgtk_last
        root.update_idletasks()
        root.update()

    cap.release()
    landmarker.close()
    root.unbind('<Return>')

    # Save calibration data
    out_path = f"{CALIB_PREFIX}{monitor_id}{CALIB_SUFFIX}"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logging.info(f"Calibrate2 saved to {out_path}")

    return all_results
