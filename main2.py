import os
import cv2
import time
import json
import logging
import math
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks.python import vision

# Логирование в файл и консоль
logging.basicConfig(filename='tracker.log', filemode='w', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Список мониторов
try:
    from screeninfo import get_monitors
    monitors = get_monitors()
    logging.info(f'Найдено мониторов: {len(monitors)}')
except ImportError:
    monitors = []
    logging.warning('screeninfo не установлен; список мониторов пуст')

# Инициализация MediaPipe FaceLandmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Утилита форматирования времени

def format_time(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# Пути к файлам калибровки
CAL_FILES = {i: f'calibration_monitor_{i}.json' for i in range(len(monitors))}

class CameraApp:
    def __init__(self, window, canvas, vid, calibration):
        self.window = window
        self.canvas = canvas
        self.vid = vid
        self.calibration = calibration  # {idx: {yaw_min,...}}

        # Текстовый элемент для "Смотрю" (нижний левый угол)
        self.label_id = canvas.create_text(
            10, 10,
            text='Смотрю: -',
            fill='cyan', font=("Helvetica",12,'bold'),
            anchor='sw',
            tags=('monitor_label',)
        )

        # Элемент для вывода кадров камеры
        self.img_id = canvas.create_image(0, 0, anchor='nw')

        # Словарь для учёта времени фокуса по каждому монитору
        self.focus = {i: 0.0 for i in calibration}
        self.timer_ids = {}

        # Таймер обновления (мс)
        self.delay = 30
        self.running = True
        self._imgtk = None
        self.after_id = None
        self.update()

    def _get_yaw_pitch(self, lms):
        dx = lms[1].x - lms[168].x
        dy = lms[1].y - lms[168].y
        yaw = math.degrees(math.atan2(dx, dy))
        pitch = math.degrees(math.atan2(dy, 0.01))
        return yaw, pitch

    def determine_monitor(self, frame):
        # Детектируем лицо и вычисляем yaw/pitch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        if not res.face_landmarks:
            return None
        yaw, pitch = self._get_yaw_pitch(res.face_landmarks[0])

        # Сверяем диапазоны калибровки
        for idx, calib in self.calibration.items():
            if all(k in calib for k in ('yaw_min','yaw_max','pitch_min','pitch_max')):
                if calib['yaw_min'] <= yaw <= calib['yaw_max'] and calib['pitch_min'] <= pitch <= calib['pitch_max']:
                    return idx + 1
        return None

    def update(self):
        if not self.running:
            return
        ok, frame = self.vid.read()
        if ok:
            # Отображение кадра
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self.img_id, image=imgtk)
            self._imgtk = imgtk

            # Определяем монитор и обновляем текст
            mon = self.determine_monitor(frame)
            height = self.canvas.winfo_height()
            y_label = height - 10  # отступ 10px с низа
            label = f"Смотрю: {mon}" if mon else "Смотрю: -"
            self.canvas.coords(self.label_id, 10, y_label)
            self.canvas.itemconfig(self.label_id, text=label)
            self.canvas.tag_raise('monitor_label')

            # Обновляем таймеры
            for i, tid in self.timer_ids.items():
                if mon == i + 1:
                    self.focus[i] += self.delay / 1000.0
                total = int(self.focus[i])
                self.canvas.itemconfig(tid, text=format_time(total))
                self.canvas.tag_raise(f'time{i}')

            # Поднимаем остальные UI-элементы
            self.canvas.tag_raise('close')
            for i in self.calibration:
                self.canvas.tag_raise(f'mon{i}')
                self.canvas.tag_raise(f'cal{i}')

        self.after_id = self.window.after(self.delay, self.update)

    def on_closing(self):
        # Завершаем цикл и освобождаем ресурсы
        self.running = False
        if self.after_id:
            self.window.after_cancel(self.after_id)
        self.vid.release()
        self.window.destroy()

# Глобальные функции: выбор и калибровка монитора
selected = set()

def select_mon(idx, rect, canvas):
    # Переключаем выбор и сохраняем в JSON
    if idx in selected:
        selected.remove(idx)
        canvas.itemconfig(rect, fill='')
    else:
        selected.add(idx)
        canvas.itemconfig(rect, fill='green')
    with open('monitor_selection.json', 'w') as f:
        json.dump({'selected_monitors': list(selected)}, f, indent=2)
    logging.info(f'Выбор сохранён: {selected}')


def calibrate_mon(idx, tag, root, canvas, vid, fw, fh):
    # Калибровка по четырём углам
    logging.info(f'Калибровка монитора {idx+1}...')
    corners = ['верхний левый', 'верхний правый', 'нижний правый', 'нижний левый']
    data = {'yaw_min': 999, 'yaw_max': -999, 'pitch_min': 999, 'pitch_max': -999}
    var = tb.BooleanVar(False)
    def on_key(e): var.set(True)
    root.bind('<Key>', on_key)

    for corner in corners:
        # Инструкция пользователю
        canvas.delete('cal_text')
        canvas.create_text(
            fw // 2, fh // 2,
            text=f"Монитор {idx+1}: смотри в {corner}",
            fill='yellow', font=("Arial", 20), tags=('cal_text',)
        )
        root.update(); var.set(False); root.wait_variable(var)
        canvas.delete('cal_text')

        ok, frame = vid.read()
        if not ok:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        if not res.face_landmarks:
            continue
        yaw, pitch = CameraApp._get_yaw_pitch(None, res.face_landmarks[0])
        # Обновляем диапазоны
        data['yaw_min'] = min(data['yaw_min'], yaw)
        data['yaw_max'] = max(data['yaw_max'], yaw)
        data['pitch_min'] = min(data['pitch_min'], pitch)
        data['pitch_max'] = max(data['pitch_max'], pitch)

    root.unbind('<Key>')
    # Сохраняем JSON калибровки
    path = f'calibration_monitor_{idx}.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    canvas.itemconfig(tag, fill='green')
    logging.info(f'Калибровка сохранена: {path}')

# Функция инициализации GUI и логики приложения

def start_camera(root, canvas, vid, fw, fh):
    # Загружаем файлы калибровки
    calibration = {}
    for i, path in CAL_FILES.items():
        if os.path.exists(path):
            try:
                calibration[i] = json.load(open(path))
            except:
                logging.warning(f'Не удалось прочитать {path}')

    app = CameraApp(root, canvas, vid, calibration)

    # Загружаем ранее выбранные мониторы
    try:
        sel_data = json.load(open('monitor_selection.json'))
        prev_sel = set(sel_data.get('selected_monitors', []))
        logging.info(f'Предыдущий выбор: {prev_sel}')
    except:
        prev_sel = set()
        logging.info('Начальный выбор пуст')

    # Кнопка закрытия
    cx, cy = fw - 20, 20
    canvas.create_text(cx, cy, text='✕', font=("Helvetica",18,'bold'), fill='white', tags=('close',))
    canvas.tag_bind('close', '<Button-1>', lambda e: app.on_closing())

    # Перетаскивание окна
    def start_move(e):
        root.x_off = e.x_root - root.winfo_x()
        root.y_off = e.y_root - root.winfo_y()
    def on_move(e):
        root.geometry(f"+{e.x_root - root.x_off}+{e.y_root - root.y_off}")
    canvas.bind('<ButtonPress-1>', start_move)
    canvas.bind('<B1-Motion>', on_move)

    # Отрисовка интерфейса: квадраты выбора, калибровки и таймеры
    sq, pad = 30, 10
    for i in calibration.keys():
        x = pad
        y = pad + i * (sq + pad)

        # Квадрат выбора монитора
        tag_mon = f'mon{i}'
        rect = canvas.create_rectangle(x, y, x+sq, y+sq, outline='white', width=2, tags=(tag_mon,))
        canvas.create_text(x+sq/2, y+sq/2, text=str(i+1), fill='white', font=("Helvetica",10,'bold'), tags=(tag_mon,))
        canvas.tag_bind(tag_mon, '<Button-1>', lambda e, idx=i, r=rect: select_mon(idx, r, canvas))

        # Квадрат калибровки
        tag_cal = f'cal{i}'
        cx2 = x + sq + pad
        cr = canvas.create_rectangle(cx2, y, cx2+sq, y+sq, outline='yellow', fill=('green' if i in calibration else ''), width=2, tags=(tag_cal,))
        canvas.create_text(cx2+sq/2, y+sq/2, text='C', fill='yellow', font=("Helvetica",10,'bold'), tags=(tag_cal,))
        canvas.tag_bind(tag_cal, '<Button-1>', lambda e, idx=i, t=tag_cal: calibrate_mon(idx, t, root, canvas, vid, fw, fh))

        # Таймер справа от квадрата калибровки
        tx = cx2 + sq + pad
        ty = y + sq/2
        tid = canvas.create_text(tx, ty, text=format_time(0), fill='white', font=("Helvetica",10), anchor='w', tags=(f'time{i}',))
        app.timer_ids[i] = tid

    # Запуск цикла обновления
    root.after(0, app.update)
    return app

# Точка входа
if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError('Не удалось открыть камеру')
    fw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Основное окно без рамки, поверх всех окон и в панели задач
    root = tb.Window(themename="darkly")
    root.overrideredirect(True)
    root.attributes('-topmost', True)
    root.geometry(f"{fw}x{fh}")
    try:
        root.iconbitmap(default='')
    except:
        pass
    root.deiconify()

    canvas = tb.Canvas(root, width=fw, height=fh, borderwidth=0, highlightthickness=0)
    canvas.pack()

    app = start_camera(root, canvas, vid, fw, fh)
    root.mainloop()
