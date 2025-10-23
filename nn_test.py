import json
import pickle
import time

import cv2
import mss
from neuralnet.core_gpu import *

from data_prepare import click, start
from online import img_prepare_online


def cycle(sct, monitor, basic, over_net, thr_1=0.9, thr_2=0.9, max_cycles=1):
    click()
    time.sleep(0.3)
    i = 0
    screens = []
    while i < max_cycles:
        screenshot = sct.grab(monitor)  # BGRA формат (4 канала)
        img = img_prepare_online(screenshot)

        over = over_net.forward(img[:, 94:, 25:-25]).get()
        if over > thr_2:
            i += 1
            print("Game_Over", i)
            screens.append(sct.grab({"top": monitor["top"] - 100, "left": 110, "width": 130, "height": 170}))
            time.sleep(2)
            click()
            time.sleep(8)
            if over_net.forward(img_prepare_online(sct.grab(monitor))[:, 94:, 25:-25]).get() > thr_2:
                click()
            time.sleep(2)
            click()
            time.sleep(0.3)
            continue
        result = basic.forward(img).get()
        if result > thr_1:
            click()
            time.sleep(0.3)
    for x in screens:
        cv2.imshow("", np.asarray(x))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    top, left, width, height = start()
    max_cycles = 10
    monitor = {
        "top": int(top + 174),
        "left": 16,
        "width": 330,
        "height": 370
    }

    sct = mss.mss()

    with open(rf"Models\final_agent.pkl", "rb") as f:
        var_1 = pickle.load(f)
    with open(rf"Models\game_over_net.pkl", "rb") as f:
        var_2 = pickle.load(f)
    with open(rf"Models\nets_thresholds.json", "rb") as f:
        thresholds = json.load(f)

    game_over_net = NeuralNetwork(var_2)
    base_agent_net = NeuralNetwork(var_1)

    cycle(sct, monitor, base_agent_net, game_over_net, thresholds["final_agent"], thresholds["game_over_net"],
          max_cycles=max_cycles)
