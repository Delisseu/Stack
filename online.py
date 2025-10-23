import json
import os
import pickle
import time

import mss
from neuralnet.Optimizers import Adam
from neuralnet.core_gpu import *

from data_prepare import click, start, img_prepare, get_value


def img_prepare_online(screenshot):
    return cp.asarray(img_prepare(screenshot) / np.float32(255))


def img_prepare_online_(screenshot):
    after_prepare = img_prepare(screenshot)
    return cp.asarray(after_prepare / np.float32(255)), after_prepare


def get_value_time(ideal_time: float, current_time: float) -> float:
    """
    ideal_time — секунда, где объект в центре
    current_time — текущий момент (сек)

    Возвращает смещение в секундах
    """
    return current_time - ideal_time


def precise_sleep(delay):
    target = time.perf_counter() + delay
    if delay > 0.005:
        time.sleep(delay - 0.005)  # почти всё время ждём «экономно»
    while time.perf_counter() < target:
        pass


def is_ideal(sct, monitor, success_net, thr):
    results = []
    for x in range(5):
        img = sct.grab(monitor)
        img = img_prepare_online(img)
        results.append((success_net.forward(img[:, 20:-5, 5:-5]).get() > thr))
    if any(results):
        return True
    else:
        return False


def cycle(sct, monitor, nn, game_over_net, success_net, thr1, thr2, thr3, max_iter=20):
    # 0.759, 0.775, 0.791
    ideal_timing_right = 0.775
    ideal_timing_left = 0.775
    right = True
    X = []
    y_true_cls = []
    y_true_reg = []
    click()
    precise_sleep(0.034)  # Задержка в самом начале
    start_ = time.perf_counter()
    precise_sleep(0.006)
    passive_time = 0.3
    current_iter = 0
    sleep = 0
    while True:
        screenshot = sct.grab(monitor)  # BGRA формат (4 канала)
        current_time = time.perf_counter() - start_

        img, after_resize = img_prepare_online_(screenshot)
        over_img = img[:, 94:, 25:-25]

        over = game_over_net.forward(over_img).get()

        if over > thr2 or current_time >= 1.45 or current_iter > max_iter:
            if current_iter > max_iter:
                click()
                time.sleep(0.005)
                click()
                print("time over")

            elif current_time >= 1.45:  # Чтобы цикл не стал бесконечным.
                click()
                time.sleep(0.005)
                click()
                print("too late")

            elif current_iter < 11:
                X, y_true_cls, y_true_reg = X[:-75], y_true_cls[:-75], y_true_reg[:-75]
                pass

            print("GAME OVER")
            time.sleep(2)
            click()
            time.sleep(3)
            if game_over_net.forward(img_prepare_online(sct.grab(monitor))[:, 94:, 25:-25]).get() > thr2:
                click()
            return X, y_true_cls, y_true_reg

        ideal_time = ideal_timing_right if right else ideal_timing_left

        y_pred = nn.forward(img).get()

        X.append(after_resize)
        y_true_cls.append(get_value(ideal_time, current_time))
        if right:
            y_true_reg.append([0, get_value_time(ideal_time, current_time)])
        else:
            y_true_reg.append([get_value_time(ideal_time, current_time), 0])

        if y_pred > thr1 and current_time > passive_time + sleep:
            click()
            start_ = time.perf_counter()
            current_iter += 1
            time.sleep(0.034)

            if is_ideal(sct, monitor, success_net, thr3):
                pass
            else:
                if right:
                    ideal_timing_right = ideal_time + (current_time - ideal_time - 0.013) / 2
                else:
                    ideal_timing_left = ideal_time + (current_time - ideal_time - 0.013) / 2
            right = not right
            sleep = np.abs((ideal_timing_right - 0.775) / 1.35) + np.abs((ideal_timing_left - 0.775) / 1.35)
            if sleep > 0:
                precise_sleep(sleep)


def mix_new_old(x_old, y_old, X_new, y_new, p_new=0.5):
    n_new = len(X_new)
    # сколько старых нужно, чтобы доля новых была p_new:
    n_old = int(np.ceil(n_new * (1 - p_new) / p_new))
    n_old = min(n_old, len(x_old))
    idx = np.random.choice(len(x_old), size=n_old, replace=False)
    X_old_sel, y_old_sel = x_old[idx], y_old[idx]
    X_mix = np.concatenate([X_old_sel, X_new], axis=0)
    y_mix = np.concatenate([y_old_sel, y_new], axis=0)
    return X_mix, y_mix


def training(sct, monitor, online_agent, game_over_net, success_net, thr1, thr2, thr3, max_data_len, max_iter):
    stream = cp.cuda.Stream(non_blocking=True)
    thr1_ = 0.98
    if os.path.isfile(r"Data\Online\X.npy"):
        x_ = np.load(r"Data\Online\X.npy")
        y_ = np.load(r"Data\Online\y.npy")
        y_reg_ = np.load(r"Data\Online\y_reg.npy")
    else:
        x_ = np.load(r"Data\pixels_all.npy")[:1620]
        y_ = np.load(r"Data\target_all.npy")[:1620]
        y_reg_ = np.load(r"Data\reg_all.npy")[:1620]
    while True:
        X, y_cls, y_reg = cycle(sct, monitor, online_agent, game_over_net, success_net, thr1_, thr2, thr3,
                                max_iter)
        if len(X) < 1:
            time.sleep(1)
            continue
        X, y_cls, y_reg = (np.asarray(X).reshape(-1, 100, 100, 1), np.asarray(y_cls, dtype=np.float32).reshape(-1, 1),
                           np.asarray(y_reg, dtype=np.float32).reshape(-1, 2))
        x_ = np.concatenate([x_, X], axis=0)
        y_ = np.concatenate([y_, y_cls], axis=0)
        y_reg_ = np.concatenate([y_reg_, y_reg], axis=0)
        np.save(r"Data\Online\X.npy", x_)
        np.save(r"Data\Online\y.npy", y_)
        np.save(r"Data\Online\y_reg.npy", y_reg_)
        X_mix, y_mix = mix_new_old(x_, y_, X, y_cls)

        if len(x_) < max_data_len * 0.95:
            y_mix = np.where(y_mix > 0.98, 0.99, 0.01).astype(np.float32)
        else:
            thr1_ = thr1

        loader = AsyncCupyDataLoader((X_mix / 255).astype(np.float32), y_mix, batch_size=32, stream=stream)
        online_agent.train(loader, epochs=2)
        var = online_agent.export()

        with open(f"Models/online_agent.pkl", "wb") as f:
            pickle.dump(var, f)

        if len(x_) > max_data_len:
            print(f"Collect {max_data_len} observations. STOP.")
            break
        else:
            print(len(x_))

        del var, loader, X_mix, y_mix, X, y_cls


if __name__ == "__main__":
    top, left, width, height = start()
    monitor = {
        "top": int(top + 174),
        "left": 16,
        "width": 330,
        "height": 370
    }
    sct = mss.mss()
    with open(r"Models\online_agent.pkl", "rb") as f:
        base_var = pickle.load(f)
    with open(r"Models\game_over_net.pkl", "rb") as f:
        over_var = pickle.load(f)
    with open(r"Models\success_net.pkl", "rb") as f:
        success_var = pickle.load(f)
    with open(rf"Models\nets_thresholds.json", "rb") as f:
        thresholds = json.load(f)

    base_nn = NeuralNetwork(base_var, BCE(), Adam())
    game_over_net = NeuralNetwork(over_var)
    success_net = NeuralNetwork(success_var)
    max_data_len = 100_000
    max_iter = 20
    training(sct, monitor, base_nn, game_over_net, success_net, thresholds["online_agent"], thresholds["game_over_net"],
             thresholds["success_net"], max_data_len, max_iter)
