import json
import pickle
import time

import mss
from neuralnet.core_gpu import *

from data_prepare import click, start
from online import img_prepare_online


def cycle(sct, monitor, game_over, success_net, thr_1=0.9, thr_2=0.9):
    click()
    time.sleep(0.2)  # Задержка в самом начале партии
    while True:
        screenshot = sct.grab(monitor)  # BGRA формат (4 канала)

        img = img_prepare_online(screenshot)
        img_1 = img[:, 94:, 25:-25].reshape(1, 6, 50, 1)
        img_2 = img[:, 20:-5, 5:-5].reshape(1, 75, 90, 1)

        success = success_net.forward(img_2).get()
        if success > thr_2:
            print("Ideal")
        z = game_over.forward(img_1).get()
        if z > thr_1:
            print("Game_Over")


if __name__ == "__main__":
    with open(rf"Models\game_over_net.pkl", "rb") as f:
        var_1 = pickle.load(f)
    with open(rf"Models\success_net.pkl", "rb") as f:
        var_2 = pickle.load(f)
    with open(rf"Models\nets_thresholds.json", "rb") as f:
        thresholds = json.load(f)

    model_1 = NeuralNetwork(var_1)
    model_2 = NeuralNetwork(var_2)

    top, left, width, height = start()
    monitor = {
        "top": int(top + 174),
        "left": 16,
        "width": 330,
        "height": 370
    }
    sct = mss.mss()

    cycle(sct, monitor, model_1, model_2, thresholds["game_over_net"], thresholds["success_net"])
