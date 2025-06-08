import cv2
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk


def main():
    # Инициализация видеопотока для получения размеров
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError("Не удалось открыть камеру")
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание главного окна без рамки
    root = tb.Window(themename="darkly")
    root.overrideredirect(True)
    root.geometry(f"{frame_width}x{frame_height}")

    # Canvas для видео и крестика
    canvas = tb.Canvas(root, width=frame_width, height=frame_height, borderwidth=0, highlightthickness=0)
    canvas.pack()

    # Прогресс-бар над видео
    progress = tb.Progressbar(root, mode='indeterminate')
    progress.place(relx=0.5, y=frame_height-10, anchor='s', relwidth=0.9)
    progress.start(10)

    def start_camera():
        # Остановить и убрать прогресс-бар
        progress.stop()
        progress.destroy()

        # Создать приложение камеры с заранее заданным canvas
        camera_app = CameraApp(root, canvas, vid)

        # Определяем позицию крестика (15px от правого и верхнего края)
        close_x = frame_width - 15
        close_y = 15
        # Создаём крестик поверх видео
        canvas.create_text(
            close_x,
            close_y,
            text='✕',
            font=("Helvetica", 18, 'bold'),
            fill='white',
            tags=('close',)
        )
        # Обработчик клика по кресту
        canvas.tag_bind('close', '<Button-1>', lambda e: camera_app.on_closing())
        # Биндинги для перетаскивания
        def start_move(event):
            root.x_offset = event.x_root - root.winfo_x()
            root.y_offset = event.y_root - root.winfo_y()
        def on_move(event):
            x = event.x_root - root.x_offset
            y = event.y_root - root.y_offset
            root.geometry(f"+{x}+{y}")
        canvas.bind('<ButtonPress-1>', start_move)
        canvas.bind('<B1-Motion>', on_move)

    root.after(2000, start_camera)
    root.mainloop()

class CameraApp:
    def __init__(self, window, canvas, vid):
        self.window = window
        self.canvas = canvas
        self.vid = vid
        # Создаем image_id один раз
        self.image_id = canvas.create_image(0, 0, anchor='nw')
        self.delay = 15
        self._running = True
        self.after_id = None
        # Переменная для хранения последнего изображения, чтобы избежать сбора мусора
        self._imgtk = None
        self.update()

    def update(self):
        if not self._running:
            return
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            # Обновляем существующий image_id
            self.canvas.itemconfig(self.image_id, image=imgtk)
            # Сохраняем ссылку на изображение
            self._imgtk = imgtk
            # Поднимаем крестик поверх
            self.canvas.tag_raise('close')
        self.after_id = self.window.after(self.delay, self.update)

    def on_closing(self):
        self._running = False
        if self.after_id:
            self.window.after_cancel(self.after_id)
        self.vid.release()
        self.window.destroy()

if __name__ == '__main__':
    main()
