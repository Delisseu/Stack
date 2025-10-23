import ctypes
import os
import time
import tkinter as tk

import cv2
import mouse
import mss
import numpy as np
import pygetwindow as gw
import win32gui

TARGET_FPS = 60
FRAME_DURATION = 1 / TARGET_FPS  # например, 1/30 ≈ 0.033 сек

# Подключаем функции из user32.dll
user32 = ctypes.windll.user32
count_frames = 90


# Устанавливаем позицию курсора
def move_mouse(x, y):
    user32.SetCursorPos(x, y)


# Кликаем мышью
def click():
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("mi", MOUSEINPUT)]

    extra = ctypes.c_ulong(0)
    mi_down = MOUSEINPUT(0, 0, 0, 2, 0, ctypes.pointer(extra))  # 2 = MOUSEEVENTF_LEFTDOWN
    mi_up = MOUSEINPUT(0, 0, 0, 4, 0, ctypes.pointer(extra))  # 4 = MOUSEEVENTF_LEFTUP
    input_down = INPUT(0, mi_down)
    input_up = INPUT(0, mi_up)

    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_down), ctypes.sizeof(input_down))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_up), ctypes.sizeof(input_up))


def start():
    hwnd = win32gui.FindWindow(None, "BlueStacks App Player")

    root = tk.Tk()
    current_height = root.winfo_screenheight()

    new_width = 414
    new_height = 698

    top = int(current_height / 2 - new_height / 2)
    left = 0

    # устанавливаем окно с учетом масштабирования
    win32gui.SetWindowPos(hwnd, None, left, top, new_width, new_height, 0)
    win = gw.getWindowsWithTitle("BlueStacks App Player")[0]
    win32gui.SetWindowPos(hwnd, None, left, top, new_width, new_height, 0x0001)

    move_x = int((win.width / 2 - 20))
    move_y = int((current_height / 2 + win.height / 2.5 - 35))
    move_mouse(move_x, move_y)

    return top, left, new_width, new_height


def get_unique_filename(base_name, extension=".mp4"):
    """
    Возвращает уникальное имя файла: base_name_1.mp4, base_name_2.mp4 и т.д.
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter}{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def record(monitor, FRAME_DURATION, sct, count_frames):
    frames = []  # список для скриншотов в виде numpy array
    for _ in range(count_frames):
        start_time = time.perf_counter()

        frames.append(sct.grab(monitor))

        sleep_time = FRAME_DURATION - (time.perf_counter() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    return frames


def save(frames, folder_name, target_size=(100, 100)):
    path = rf"Data\Images\{folder_name}"
    os.makedirs(path, exist_ok=True)

    existing_files = set(os.listdir(path))
    frames_resized = []

    # Определяем стартовый индекс, чтобы не перезаписать существующие screenshot_X.png
    i = 0
    while f"screenshot_{i}.png" in existing_files:
        i += 1
    start_index = i

    for j, frame in enumerate(frames):
        img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        filename = f"screenshot_{start_index + j}.png"
        cv2.imwrite(os.path.join(path, filename), resized)
        frames_resized.append(resized)

    # Сохраняем как numpy-массив, избегая перезаписи
    npy_index = 0
    npy_filename = "pixels_0.npy"
    while npy_filename in existing_files:
        npy_index += 1
        npy_filename = f"pixels_{npy_index}.npy"

    np.save(os.path.join(path, npy_filename), np.array(frames_resized))


def img_prepare(screenshot):
    img = np.asarray(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    return img.reshape(1, 100, 100, 1)


def timing(sct, monitor):  # Чтобы понять идеальный тайминг для клика ~ 0.8
    screens = []
    tims = []
    click()
    time.sleep(0.05)
    start = time.perf_counter()
    for x in range(80):
        screens.append(sct.grab(monitor))
        tims.append(time.perf_counter() - start)

    for screen, tim in zip(screens, tims):
        print(tim)
        cv2.imshow("", np.array(screen))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def full_cycle():
    click()
    time.sleep(3 * FRAME_DURATION)


def full_cycle_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(45 * FRAME_DURATION)  # Кадр идеальной плитки + input_lag
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_width_cycle_right_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_width_cycle_right():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(45 * FRAME_DURATION)  # Кадр идеальной плитки + input_lag
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_width_cycle_left_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_width_cycle_left():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(45 * FRAME_DURATION)  # Кадр идеальной плитки + input_lag
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_height_cycle_right_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(45 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(45 * FRAME_DURATION)  # Кадр идеальной плитки + input_lag
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_height_cycle_right():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(45 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_height_cycle_left_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(45 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(45 * FRAME_DURATION)  # Кадр идеальной плитки + input_lag
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def half_height_cycle_left():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(45 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def right_up_cycle():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def right_up_cycle_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep((45 + (30 - 45) // 2) * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def right_down_cycle():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def right_down_cycle_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(30 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep((45 + (30 - 45) // 2) * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def left_up_cycle():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def left_up_cycle_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(30 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep((45 + (60 - 45) // 2) * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def left_down_cycle():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def left_down_cycle_T():
    click()
    time.sleep(3 * FRAME_DURATION)  # Задержка в самом начале партии
    time.sleep(60 * FRAME_DURATION)  # Половина плитки
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep(60 * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки

    time.sleep((45 + (60 - 45) // 2) * FRAME_DURATION)
    click()
    time.sleep(2 * FRAME_DURATION)  # Задержка после нажатия кнопки


def data_markup_reg(folders_names, best_frames, count_frames):
    right = True
    for folder_name, best_frame in zip(folders_names, best_frames):
        array = []
        for i in range(count_frames):
            if right:
                array.append([0, get_value_frames(best_frame, i)])
            else:
                array.append([get_value_frames(best_frame, i), 0])
        right = not right
        np.save(rf"Data\Images\{folder_name}\target_reg.npy", np.array(array))


def get_value(ideal_frame, current_frame):
    if current_frame >= ideal_frame:
        return 1

    return 0


def data_markup(folders_names, best_frames, count_frames):
    for folder_name, best_frame in zip(folders_names, best_frames):
        array = []
        for i in range(count_frames):
            array.append(get_value(best_frame, i))
        np.save(rf"Data\Images\{folder_name}\target.npy", np.array(array).reshape(-1, 1))


def data_markup_SuccessNet(folders_names, list_of_need_frames, count_frames):
    for folder_name, list_of_need_frame in zip(folders_names, list_of_need_frames):
        arr = np.zeros(count_frames)
        arr[list_of_need_frame] = 1
        np.save(rf"Data\Images\{folder_name}\target_SuccessNet.npy", np.array(arr).reshape(-1, 1))


def get_value_frames(ideal_frame: int, current_frame: int, fps: int = 60) -> float:
    """
    ideal_frame — кадр, где объект в центре
    current_frame — текущий кадр
    fps — частота кадров (для нормализации)

    Возвращает смещение в секундах
    """
    return (current_frame - ideal_frame) / fps


def data_union(folders_names, count_frames):
    pixel_list = []
    pixel_list_2 = []
    target_list = []
    reg_list = []
    for folder_name in folders_names:
        pixel_array = np.load(rf"Data\Images\{folder_name}\pixels_0.npy").reshape(count_frames, 100,
                                                                                  100, 1)  # shape: (b, h, w, C)
        target_array = np.load(rf"Data\Images\{folder_name}\target.npy").reshape(count_frames, 1)
        reg_array = np.load(rf"Data\Images\{folder_name}\target_reg.npy").reshape(count_frames, 2)
        pixel_list_2.append(255 - pixel_array)  # Инверсия
        pixel_list.append(pixel_array)
        reg_list.append(reg_array)
        target_list.append(target_array)

    pixel = np.concatenate(pixel_list + pixel_list_2, axis=0)
    pixel = np.concatenate([pixel] + [pixel[:, :, ::-1]], axis=0)  # + Отражение по горизонтали
    target = np.concatenate(target_list * 4, axis=0)
    reg = np.concatenate(reg_list * 4, axis=0)

    # Объединение всех массивов по первой оси (кол-во кадров)
    np.save(rf"Data\pixels_all.npy", pixel)
    np.save(rf"Data\target_all.npy", target)
    np.save(rf"Data\reg_all.npy", reg)


def test_data():
    cv2.namedWindow('MyWindow', cv2.WINDOW_NORMAL)
    cv2.moveWindow('MyWindow', 3840 // 2, 2160 // 2)

    pixels = np.load(rf"Data\game_over_data.npy")
    print(len(pixels))
    targets = np.load(rf"Data\game_over_target.npy")
    print(len(targets))
    targets_2 = np.load(rf"Data\target_success_all.npy")
    print(len(targets_2))

    for ind, img_value_targets_2 in enumerate(zip(pixels, targets, targets_2)):
        if ind % 1 == 0:
            # pyperclip.copy(f", {ind}")  # записать в буфер
            img, target, target_2 = img_value_targets_2
            print(target, ind, target_2)
            img = cv2.resize(img.reshape(100, 100, 1), (300, 300))
            cv2.imshow("MyWindow", img)
            cv2.waitKey(0)
            # cv2.waitKey(50)


def data_union_gameover():
    data_list = []
    for folder in ["game_over", "game_over_not"]:
        i = 0
        while True:
            filename = rf"Data\Images\{folder}\pixels_{i}.npy"
            if os.path.isfile(filename):
                data_list.append(np.load(filename).reshape(-1, 100, 100))
            else:
                break
            i += 1

    data = np.concatenate(data_list, axis=0)
    np.save(rf"Data\game_over_data.npy", data)
    target = np.concatenate([np.zeros((len(data) - i * 5, 1)), np.ones((i * 5, 1))], axis=0)
    np.save(rf"Data\game_over_target.npy", target)


def precise_sleep(delay):
    target = time.perf_counter() + delay
    while time.perf_counter() < target:
        pass


def check(sct, monitor):  # Чтобы понять разницу между идеальныйм таймингом и задержкой при нажатии.
    click()
    precise_sleep(0.034)
    i = 0
    screens = []
    y = []
    start = time.perf_counter()
    # screen = sct.grab(monitor)
    time.sleep(0.005)
    while i < 800:
        screen = sct.grab(monitor)
        y.append(time.perf_counter() - start)

        if y[-1] >= 0.79:
            click()
            start = time.perf_counter()
        screens.append(screen)
        i += 1
    for x, i in zip(screens[600:], y[600:]):
        print(i)
        cv2.imshow("", np.array(x))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def success_collect(monitor):
    def on_click():
        screens = []
        with mss.mss() as sct:
            time.sleep(0.034)
            for x in range(5):
                screen = sct.grab(monitor)
                img = img_prepare(screen)[:, 20:-5, 5:-5].reshape(75, 90, 1)
                screens.append(img)
            if os.path.isfile(r"Data\success_collect.npy"):
                prev = np.load(r"Data\success_collect.npy")
                screens = np.concatenate([prev, screens], axis=0)
            print(len(screens))
            np.save(r"Data\success_collect.npy", screens)

    click()
    mouse.on_click(on_click)
    while True:
        mouse.wait()  # блокируем и ждём событий


# def check(sct, monitor):  # Чтобы понять разницу между идеальныйм таймингом и задержкой при нажатии.
#     click()
#     precise_sleep(0.034)
#     i = 0
#     screens = []
#     y = []
#     start = time.perf_counter()
#     while i < 100:
#         x = time.perf_counter()
#         screen = sct.grab(monitor)
#         z = time.perf_counter() - x
#         y.append(time.perf_counter() - start)
#         z = 0.016 - z
#         screens.append(screen)
#         i += 1
#         if z > 0:
#             precise_sleep(z)
#     for x, i in zip(screens, y):
#         print(i)
#         cv2.imshow("", np.array(x))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


folders_names = ["full_cycle", "full_cycle_T",
                 "half_height_cycle_left", "half_height_cycle_left_T",
                 "half_height_cycle_right", "half_height_cycle_right_T",
                 "half_width_cycle_left", "half_width_cycle_left_T",
                 "half_width_cycle_right", "half_width_cycle_right_T",
                 "left_down_cycle", "left_down_cycle_T",
                 "left_up_cycle", "left_up_cycle_T",
                 "right_down_cycle", "right_down_cycle_T",
                 "right_up_cycle", "right_up_cycle_T"]

best_frames = [42, 42,
               42, 33,
               42, 49,
               48, 42,
               35, 42,
               49, 49,
               49, 35,
               35, 49,
               35, 35]


def success_markup_and_save(false_indxs):
    x = np.load(r"Data\success_collect.npy")
    length = len(x)
    target = np.ones((length, 1))
    for false_ind in false_indxs:
        target[false_ind: false_ind + 5] = 0
    np.save(r"Data\success_collect_target.npy", target)


def check_success():
    data = np.load(r"Data\success_collect.npy")
    ind = 0
    false_indxs = []
    for img in data[::5]:
        print(ind)
        img = cv2.resize(img[20:-5, 5:-5], (300, 300))
        cv2.imshow("MyWindow", img)
        key = cv2.waitKey(0)
        if key == 32:  # Space
            false_indxs.append(ind)
        ind += 5
    return false_indxs


def collect_base_data(monitor, sct):
    for func, folders_name in zip(
            [full_cycle, full_cycle_T, half_height_cycle_left, half_height_cycle_left_T, half_height_cycle_right,
             half_height_cycle_right_T, half_width_cycle_left, half_width_cycle_left_T, half_width_cycle_right,
             half_width_cycle_right_T, left_down_cycle, left_down_cycle_T, left_up_cycle, left_up_cycle_T,
             right_down_cycle, right_down_cycle_T, right_up_cycle, right_up_cycle_T], folders_names):
        func()
        data = record(monitor, FRAME_DURATION, sct, count_frames=count_frames)
        click()
        save(data, folders_name)
        time.sleep(3)
        click()
        time.sleep(3)


if __name__ == "__main__":
    top, left, width, height = start()
    monitor = {
        "top": int(top + 174),
        "left": 16,
        "width": 330,
        "height": 370
    }
    sct = mss.mss()

    collect_base_data(monitor, sct)
    data_markup(folders_names, best_frames, count_frames)
    data_markup_reg(folders_names, best_frames, count_frames)
    data_union(folders_names, count_frames)
    # test_data()
    success_collect(monitor)
    success_markup_and_save(check_success())

    save(record(monitor, FRAME_DURATION, sct, count_frames=1), folder_name="game_over")
    save(record(monitor, FRAME_DURATION, sct, count_frames=1), folder_name="game_over_not")
    data_union_gameover()
