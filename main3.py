import os
import cv2
import json
import math
import time
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks.python import vision
from screeninfo import get_monitors
import numpy as np

# Пути и константы
MODEL_PATH        = "face_landmarker.task"
SELECTED_PATH     = "selected_monitors.json"  # Убрано: список берётся из calibration.json
CALIBRATION_PATH  = "calibration.json"
FONT_PATH         = "C:/Windows/Fonts/arial.ttf"
CALIB_TILE_PREFIX = "calibration_monitor_"
CALIB_TILE_SUFFIX = ".json"

# Проверка наличия файлов
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл не найден: {MODEL_PATH}")

# Создаем selected_monitors.json, если нет
if not os.path.exists(SELECTED_PATH):
    default = list(range(len(get_monitors())))
    with open(SELECTED_PATH, 'w', encoding='utf-8') as f:
        json.dump(default, f)

# Загружаем список выбранных мониторов
with open(SELECTED_PATH, encoding="utf-8") as f:
    sel = json.load(f)
selected = [int(x) for x in sel]
if not selected:
    raise RuntimeError("Нет выбранных мониторов в selected_monitors.json")

# Мониторы: используем выбранные
monitors = selected

# Калибровка: отдельный файл на каждый монитор
calibration_all = {}
for m in monitors:
    path = f"{CALIB_TILE_PREFIX}{m}{CALIB_TILE_SUFFIX}"
    if os.path.exists(path):
        calibration_all[str(m)] = json.load(open(path, encoding='utf-8'))
    else:
        calibration_all[str(m)] = {}
monitors = selected

# Калибровка: отдельный файл на каждый монитор
calibration_all = {}
for m in monitors:
    path = f"{CALIB_TILE_PREFIX}{m}{CALIB_TILE_SUFFIX}"
    if os.path.exists(path):
        calibration_all[str(m)] = json.load(open(path))
    else:
        calibration_all[str(m)] = {}

monitors = [int(m) for m in calibration_all.keys()]

# Шрифт

with open(SELECTED_PATH, encoding="utf-8") as f:
    sel = json.load(f)
selected = [int(x) for x in sel]
if not selected:
    raise RuntimeError("Нет выбранных мониторов в selected_monitors.json")

# Удалено: чтение единого calibration.json, используем отдельные файлы
# Шрифтont = ImageFont.truetype(FONT_PATH, 20)

# Удаляем загрузку selected_monitors.json, используем все мониторы из calibration
monitors = list(calibration_all.keys())


# Настройка MediaPipe FaceLandmarker
BaseOptions = mp.tasks.BaseOptions
FLOptions   = vision.FaceLandmarkerOptions
FL          = vision.FaceLandmarker
RM          = vision.RunningMode
options = FLOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RM.VIDEO,
    num_faces=1
)
landmarker = FL.create_from_options(options)

# Функция для расчета yaw/pitch
def get_yaw_pitch(lm):
    nt, nb = lm[1], lm[168]
    dx, dy = nt.x - nb.x, nt.y - nb.y
    return math.degrees(math.atan2(dx, dy)), math.degrees(math.atan2(dy, 0.01))

class CameraApp:
    def __init__(self, root, vid):
        self.root = root
        self.vid = vid
        self.running = True
        self.delay = 30  # ms между кадрами

        # Подготовка окна и canvas
        fw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tb.Canvas(root, width=fw, height=fh, borderwidth=0, highlightthickness=0)
        self.canvas.pack()

        self.monitors     = monitors
        self.calibration  = calibration_all
        self.focus_times  = {m: 0.0 for m in self.monitors}
        self.away_time    = 0.0
        self.timer_ids    = {}
        self.monitor_id   = None
        # ID для подсветки области взгляда
        self.highlight_id = None

        # Видео на canvas
        self.img_id = self.canvas.create_image(0, 0, anchor='nw')

        # UI
        self._draw_close()
        self._draw_selectors()
        self._draw_timers()
        self._draw_status_labels()

        # Dragging
        self.canvas.bind('<ButtonPress-1>', self.start_move)
        self.canvas.bind('<B1-Motion>', self.on_move)

        # Запуск обновления
        self.schedule_update()
        self.root = root
        self.vid = vid
        self.running = True
        self.delay = 30  # ms

        # Подготовка окна и canvas
        fw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tb.Canvas(root, width=fw, height=fh, borderwidth=0, highlightthickness=0)
        self.canvas.pack()

        # Состояние
        self.monitors   = selected
        self.calibration = calibration_all
        self.focus_times = {m: 0.0 for m in self.monitors}
        self.away_time   = 0.0
        self.timer_ids   = {}

        # Видео на canvas
        self.img_id = self.canvas.create_image(0, 0, anchor='nw')

        # UI: close, selectors, timers, labels
        self._draw_close()
        self._draw_selectors()
        self._draw_timers()
        self._draw_status_labels()

        # Dragging
        self.canvas.bind('<ButtonPress-1>', self.start_move)
        self.canvas.bind('<B1-Motion>', self.on_move)

        # Запуск обновления
        self.schedule_update()

    def _draw_close(self):
        w = self.canvas.winfo_reqwidth()
        self.canvas.create_text(w-20, 20, text='✕', font=("Helvetica",18,'bold'), fill='white', tags=('close',))
        self.canvas.tag_bind('close', '<Button-1>', lambda e: self.on_closing())

    def _draw_selectors(self):
        sq, pad = 30, 10
        self.rects = {}
        self.gears = {}
        for i, m in enumerate(self.monitors):
            x = pad
            y = pad + i*(sq+pad)
            tag = f'mon{i}'
            sel_fill = 'green' if m in self.monitors else ''
            rect = self.canvas.create_rectangle(x, y, x+sq, y+sq, outline='white', fill=sel_fill, width=2, tags=(tag,))
            self.canvas.create_text(x+sq/2, y+sq/2, text=str(m), fill='white', font=("Helvetica",10,'bold'), tags=(tag,))
            self.canvas.tag_bind(tag, '<Button-1>', lambda e,i=i,r=rect: self.toggle_monitor(i, r))
            self.rects[i] = rect
            # Шестеренка
            self._draw_gear(i)

    def _draw_gear(self, idx):
        sq, pad = 30, 10
        m = self.monitors[idx]
        x = pad + sq + pad*2
        y = pad + idx*(sq+pad) + sq/2
        color = 'green' if str(m) in self.calibration else 'yellow'
        gid = self.canvas.create_text(x, y, text='⚙', font=("Helvetica",10,'bold'), fill=color, tags=(f'gear{idx}',))
        self.canvas.tag_bind(f'gear{idx}', '<Button-1>', lambda e, i=idx: self.calibrate_monitor(i))
        self.gears[idx] = gid

    def _draw_timers(self):
        sq, pad = 30, 10
        for idx, m in enumerate(self.monitors):
            x = pad + sq + pad*3 + 10
            y = pad + idx*(sq+pad) + sq/2
            tid = self.canvas.create_text(x, y, text='00:00:00', fill='white', font=("Helvetica",10), anchor='w', tags=(f'time{idx}',))
            self.timer_ids[idx] = tid

    def _draw_status_labels(self):
        h = self.canvas.winfo_reqheight()
        self.monitor_lbl = self.canvas.create_text(10, h-30, text='Смотрю: -', fill='cyan', anchor='sw', font=("Helvetica",12,'bold'), tags=('mon_lbl',))
        self.away_lbl    = self.canvas.create_text(200, h-30, text='Вне мониторов: 00:00:00', fill='orange', anchor='sw', font=("Helvetica",12,'bold'), tags=('away_lbl',))

    def toggle_monitor(self, idx, rect):
        # переключаем выделение: TODO реализовать
        pass

    def calibrate_monitor(self, idx):
        # вызов вашего скрипта calibrate монитора idx
        pass

    def start_move(self, e):
        self.root.x_off = e.x_root - self.root.winfo_x()
        self.root.y_off = e.y_root - self.root.winfo_y()

    def on_move(self, e):
        x = e.x_root - self.root.x_off
        y = e.y_root - self.root.y_off
        self.root.geometry(f"+{x}+{y}")

    def schedule_update(self):
        self.after_id = self.root.after(self.delay, self.update)

    def update(self):
        if not self.running: return
        ret, frame = self.vid.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(pil)
            self.canvas.itemconfig(self.img_id, image=imgtk)
            self._imgtk = imgtk

            mon = self.determine_monitor(frame)
            if mon is not None:
                idx = self.monitors.index(mon)
                self.focus_times[mon] += self.delay/1000.0
                t = int(self.focus_times[mon]); h=t//3600; m=(t%3600)//60; s=t%60
                self.canvas.itemconfig(f'time{idx}', text=f"{h:02}:{m:02}:{s:02}")
                self.canvas.itemconfig('mon_lbl', text=f"Смотрю: {mon}")
            else:
                self.away_time += self.delay/1000.0
                t = int(self.away_time); h=t//3600; m=(t%3600)//60; s=t%60
                self.canvas.itemconfig('away_lbl', text=f"Вне мониторов: {h:02}:{m:02}:{s:02}")
                self.canvas.itemconfig('mon_lbl', text='Смотрю: -')
        self.schedule_update()

    def determine_monitor(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        if not res.face_landmarks:
            return None
        yaw, pitch = get_yaw_pitch(res.face_landmarks[0])
        best, dist_min = None, float('inf')
        for m in self.monitors:
            if str(m) not in self.calibration: continue
            for c in self.calibration[str(m)].values():
                d = math.hypot(yaw - c['yaw'], pitch - c['pitch'])
                if d < dist_min:
                    dist_min, best = d, m
        return best

    def on_closing(self):
        self.running = False
        if hasattr(self, 'after_id'):
            self.root.after_cancel(self.after_id)
        landmarker.close()
        self.vid.release()
        self.root.quit()

if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    if not vid.isOpened(): raise RuntimeError('Не удалось открыть камеру')
    root = tb.Window(themename='darkly')
    root.overrideredirect(True)
    root.attributes('-topmost', True)
    CameraApp(root, vid)
    root.mainloop()
