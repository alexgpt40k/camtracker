# Stable Version 1.1.21
# Generated on: 2025-06-01 UTC+0300
# Changelog:
# - Перенёс логику калибровки в новый модуль calibrate2.py
# - При нажатии на ⚙ запускаем calibrate2.calibrate_monitor (оверлей в Canvas)
# - Окно не перемещается, «зависание» устранено
# - Повышена стабильность отрисовки и обновления после калибровки

import os
import cv2
import json
import math
import time
import logging
import subprocess
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks.python import vision
from screeninfo import get_monitors

# Обязательно: импорт новой функции калибровки
from calibrate2 import calibrate_monitor

# ------------------------------------------------------------
# Конфигурация логирования
logging.basicConfig(
    filename='tracker5.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

VERSION = "1.1.21"
GEN_TIME = time.strftime('%Y-%m-%d %H:%M:%S UTC+0300', time.localtime())

# Пути до файлов
SELECTED_PATH = "selected_monitors.json"
CALIB_PREFIX  = "calibration_monitor_"
CALIB_SUFFIX  = ".json"
MODEL_PATH    = "face_landmarker.task"

# Список доступных мониторов (по screeninfo)
available = list(range(len(get_monitors())))
if not available:
    raise RuntimeError("Не удалось получить список мониторов")

# Проверка и загрузка выбранных мониторов
if not os.path.exists(SELECTED_PATH):
    # Если файла нет, создаём его с выбором всех мониторов по умолчанию
    with open(SELECTED_PATH, 'w', encoding='utf-8') as f:
        json.dump(available, f)

with open(SELECTED_PATH, 'r', encoding='utf-8') as f:
    selected = json.load(f)
# Фильтруем, чтобы выбранные действительно входили в available
selected = [m for m in selected if m in available]

# Загрузка калибровочных данных из файлов calibration_monitor_{m}.json
calibration = {m: {} for m in available}
for m in available:
    path = f"{CALIB_PREFIX}{m}{CALIB_SUFFIX}"
    if os.path.exists(path):
        try:
            calibration[m] = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            calibration[m] = {}

# Настройка MediaPipe FaceLandmarker
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode     = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Инициализируем один общий landmarker (всего один экземпляр для всей программы)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Утилиты
def get_yaw(lm):
    """
    Вычисляет только yaw (горизонтальный угол наклона головы)
    на основе двух ключевых точек: lm[1] (верхняя точка) и lm[168] (нижняя точка).
    """
    dx = lm[1].x - lm[168].x
    dy = lm[1].y - lm[168].y
    return math.degrees(math.atan2(dx, dy))


class CameraApp:
    def __init__(self, root, vid):
        self.x_off = 0
        self.y_off = 0

        self.root = root
        self.vid  = vid

        # Размер кадра (берём из настроек камеры)
        fw = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Состояние и параметры
        self.running = True
        self.delay   = 30  # миллисекунд между кадрами

        # Авто-блокировка при отсутствии лица
        self.no_face_time      = 0.0
        self.locked            = False
        self.absence_threshold = 30.0  # секунд до блокировки

        # Подготовка Canvas для показа видео
        self.canvas = tb.Canvas(
            root,
            width=fw,
            height=fh,
            borderwidth=0,
            highlightthickness=0
        )
        self.canvas.pack()

        # Состояние мониторов
        self.monitors   = available
        self.selected   = set(selected)
        self.calib      = calibration
        self.focus_time = {m: 0.0 for m in self.monitors}
        self.away_time  = 0.0

        # Хранилища элементов Canvas
        self.rects     = {}  # прямоугольники выбора монитора
        self.gears     = {}  # иконки шестерёнок для калибровки
        self.timers    = {}  # текстовые элементы для таймеров
        self.highlight = None

        # Создаём элемент для вывода кадра (который будет обновляться)
        self.img_id = self.canvas.create_image(
            0, 0,
            anchor='nw',
            image=None
        )

        # Рисуем интерфейс
        self._draw_close()
        self._draw_selectors()
        self._draw_timers()
        self._draw_labels()

        # Привязываем перетаскивание окна
        self.canvas.bind('<ButtonPress-1>', self._start_move)
        self.canvas.bind('<B1-Motion>',  self._on_move)

        # Запускаем цикл обновления видео
        self.after_id = self.root.after(self.delay, self._update)

    def _draw_close(self):
        """Рисует крестик в правом верхнем углу Canvas."""
        w = self.canvas.winfo_reqwidth()
        self.canvas.create_text(
            w - 15, 15,
            text='✕',
            font=("Helvetica", 16, 'bold'),
            fill='white',
            tags=('close',)
        )
        self.canvas.tag_bind('close', '<Button-1>',
                             lambda e: self._on_close())

    def _draw_selectors(self):
        """
        Рисует квадратики выбора монитора (левая часть Canvas).
        При клике меняем его цвет, показываем/скрываем шестерёнку и таймер.
        """
        sq, pad = 30, 10
        for i, m in enumerate(self.monitors):
            x = pad
            y = pad + i * (sq + pad)
            tag_sel  = f'select{i}'
            tag_gear = f'gear{i}'

            col = 'green' if m in self.selected else 'white'
            rect = self.canvas.create_rectangle(
                x, y, x + sq, y + sq,
                outline=col, width=2,
                tags=(tag_sel,)
            )
            self.rects[i] = rect
            self.canvas.tag_bind(tag_sel, '<Button-1>',
                                 lambda e, i=i: self._toggle(i))

            self.canvas.create_text(
                x + sq / 2, y + sq / 2,
                text=str(m),
                fill='white',
                tags=(tag_sel,)
            )

            state = 'normal' if m in self.selected else 'hidden'
            color = 'green' if (m in self.selected and self.calib.get(m)) else 'yellow'
            gear = self.canvas.create_text(
                x + sq + pad * 2, y + sq / 2,
                text='⚙',
                font=("Helvetica", 12),
                fill=color,
                state=state,
                tags=(tag_gear,)
            )
            self.gears[i] = gear
            self.canvas.tag_bind(tag_gear, '<Button-1>',
                                 lambda e, i=i: self._calibrate(i))

    def _draw_timers(self):
        """
        Рисует справа от квадратиков текстовые таймеры,
        показывающие, сколько времени смотрим на каждый монитор.
        """
        sq, pad = 30, 10
        for i, m in enumerate(self.monitors):
            x = pad + sq + pad * 3 + 5
            y = pad + i * (sq + pad) + sq / 2
            state = 'normal' if m in self.selected else 'hidden'
            tid = self.canvas.create_text(
                x, y,
                text='00:00:00',
                font=("Helvetica", 10),
                fill='white',
                anchor='w',
                state=state
            )
            self.timers[i] = tid

    def _draw_labels(self):
        """
        Рисует внизу две надписи:
          1) self.mon_lbl — «Смотрю: {m}» или «Смотрю: -»
          2) self.aw_lbl  — «Вне мониторов: 00:00:00»
        """
        h = int(self.canvas['height'])
        self.mon_lbl = self.canvas.create_text(
            10, h - 20,
            text='Смотрю: -',
            fill='cyan',
            anchor='w'
        )
        self.aw_lbl = self.canvas.create_text(
            200, h - 20,
            text='Вне мониторов: 00:00:00',
            fill='orange',
            anchor='w'
        )
        self.canvas.tag_raise('close')
        for t in self.timers.values():
            self.canvas.tag_raise(t)

    def _toggle(self, i):
        """
        Переключает состояние квадрата выбора i-го монитора:
          — если было выбрано, снимаем выбор (outline становится белым),
            прячем шестерёнку и таймер, сбрасываем фокусное время.
          — если не было выбрано, активируем (outline = 'green'),
            показываем шестерёнку (жёлтую/зелёную) и таймер.
        Сохраняет текущий список selected в SELECTED_PATH.
        """
        m = self.monitors[i]
        curr = self.canvas.itemcget(self.rects[i], 'outline')
        new  = 'green' if curr == 'white' else 'white'
        self.canvas.itemconfig(self.rects[i], outline=new)

        state = 'normal' if new == 'green' else 'hidden'
        self.canvas.itemconfigure(self.gears[i], state=state)
        self.canvas.itemconfigure(self.timers[i], state=state)

        if state == 'hidden':
            self.focus_time[m] = 0.0
            self.canvas.itemconfig(self.timers[i], text='00:00:00')

        if new == 'green':
            self.selected.add(m)
        else:
            self.selected.discard(m)

        with open(SELECTED_PATH, 'w') as f:
            json.dump(sorted(self.selected), f)

    def _calibrate(self, i):
        """
        По клику на шестерёнку запускает calibrate_monitor из calibrate2.py
        с передачей текущего Canvas и img_id, чтобы калибровка шла «на месте».
        После завершения обновляем цвет шестерёнки и возобновляем _update().
        """
        m = self.monitors[i]
        logging.info(f"Start calibration2 for monitor {m}")

        # Прерываем текущий цикл обновления, чтобы калибровка не конфликтовала
        try:
            self.root.after_cancel(self.after_id)
        except:
            pass

        try:
            results = calibrate_monitor(self.root, m, self.canvas, self.img_id)
            self.calib[m] = results
            self.canvas.itemconfig(self.gears[i], fill='green')
            logging.info(f"Calibration2 completed for monitor {m}")
        except Exception as e:
            logging.error(f"Calibration2 failed for monitor {m}: {e}")
        finally:
            # Сразу после калибровки обновляем кадр
            self.after_id = self.root.after(self.delay, self._update)

    def _start_move(self, e):
        """
        Обработчик начала перетаскивания: сохраняем смещение курсора.
        """
        self.x_off = e.x_root - self.root.winfo_x()
        self.y_off = e.y_root - self.root.winfo_y()

    def _on_move(self, e):
        """
        Обработчик движения мыши при зажатой левой кнопке:
        перемещаем окно согласно сохранённому смещению.
        """
        new_x = e.x_root - self.x_off
        new_y = e.y_root - self.y_off
        self.root.geometry(f'+{new_x}+{new_y}')

    def _update(self):
        """
        Основной цикл (каждые self.delay мс):
         1) Берём кадр из камеры, зеркалим, конвертим в RGB, рисуем в Canvas.
         2) Запускаем детекцию лица (MediaPipe). Если лицо не видно > 30 с, блокируем workstation.
         3) Если лицо есть, вычисляем yaw, сравниваем с calib[m][corner]['yaw'] для выбранных m.
         4) Подсвечиваем наилучший монитор, обновляем таймер и текст «Смотрю: {m}».
         5) Если лицо вне диапазона – удаляем подсветку и считаем «вне мониторов».
         6) Запланирован следующий вызов через root.after.
        """
        if not self.running:
            return

        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfig(self.img_id, image=imgtk)
            self._img = imgtk

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            dt = self.delay / 1000.0
            if not res.face_landmarks:
                self.no_face_time += dt
                if self.no_face_time >= self.absence_threshold and not self.locked:
                    try:
                        subprocess.run(['rundll32.exe', 'user32.dll,LockWorkStation'], check=False)
                    except:
                        pass
                    self.locked = True
            else:
                self.no_face_time = 0.0

            best, bd = None, float('inf')
            if res.face_landmarks:
                yaw = get_yaw(res.face_landmarks[0])
                for idx, m in enumerate(self.monitors):
                    if m not in self.selected:
                        continue
                    for corner_data in self.calib[m].values():
                        if not isinstance(corner_data, dict):
                            continue
                        d = abs(yaw - corner_data['yaw'])
                        if d < bd:
                            bd, best = d, idx

            if best is not None:
                if self.highlight:
                    self.canvas.delete(self.highlight)
                coords = self.canvas.coords(self.rects[best])
                self.highlight = self.canvas.create_rectangle(
                    *coords, fill='cyan', stipple='gray25', outline=''
                )
                mv = self.monitors[best]
                self.focus_time[mv] += dt
                t = int(self.focus_time[mv])
                h, mm, s = t // 3600, (t % 3600) // 60, t % 60
                self.canvas.itemconfig(self.timers[best], text=f'{h:02}:{mm:02}:{s:02}')
                self.canvas.itemconfig(self.mon_lbl, text=f'Смотрю: {mv}')
            else:
                if self.highlight:
                    self.canvas.delete(self.highlight)
                self.away_time += dt
                t = int(self.away_time)
                h, mm, s = t // 3600, (t % 3600) // 60, t % 60
                self.canvas.itemconfig(self.aw_lbl,
                                       text=f'Вне мониторов: {h:02}:{mm:02}:{s:02}')
                self.canvas.itemconfig(self.mon_lbl, text='Смотрю: -')

        self.after_id = self.root.after(self.delay, self._update)

    def _on_close(self):
        """
        Обработчик закрытия (крестик):
         — отключает cycle (_update),
         — пытается отменить запланированный after,
         — закрывает landmarker и камеру,
         — завершает главное окно.
        """
        self.running = False
        try:
            self.root.after_cancel(self.after_id)
        except:
            pass
        try:
            landmarker.close()
        except:
            pass
        try:
            self.vid.release()
        except:
            pass
        self.root.quit()


if __name__ == '__main__':
    # Открываем камеру
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not vid.isOpened():
        raise RuntimeError('Unable to open camera')

    root = tb.Window(themename='darkly')
    root.overrideredirect(True)
    root.attributes('-topmost', True)

    # Центрируем окно размером 800×600
    ww, wh = 800, 600
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x = (sw - ww) // 2
    y = (sh - wh) // 2
    root.geometry(f'{ww}x{wh}+{x}+{y}')

    app = CameraApp(root, vid)
    root.mainloop()
