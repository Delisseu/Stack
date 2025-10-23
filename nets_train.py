import gc
import json
import pickle
import sys

from neuralnet.Layers_Features import BatchNorm, Pooling, xavier_uniform
from neuralnet.Optimizers import Adam, InverseSqrtSchedulerMod1
from neuralnet.core_gpu import *

var_1 = [
    {"input_dim": 300, 'neurons': 128, "lr": 0.0001}, {"layer": Relu},
    {'neurons': 64, "lr": 0.0001}, {"layer": Relu},
    {'neurons': 1, "lr": 0.0001, "init_func": xavier_uniform}]

var_2 = [
    {"input_dim": (75, 90, 1), "out_channels": 4, "layer": Conv2D, "lr": 0.0001, "bias": False},
    {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 8, "layer": Conv2D, "lr": 0.0001, "bias": False},
    {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu}, {"layer": Pooling},

    {'neurons': 512, "lr": 0.0001, "bias": False}, {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu},
    {'neurons': 64, "lr": 0.0001, "bias": False}, {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu},
    {'neurons': 8, "lr": 0.0001}, {"layer": Relu},
    {'neurons': 1, "lr": 0.0001, "init_func": xavier_uniform}]

var_3 = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "lr": 0.0001, "bias": False},
    {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 16, "layer": Conv2D, "lr": 0.0001, "bias": False},
    {"layer": BatchNorm, "lr": 0.0001}, {"layer": Relu},

    {"out_channels": 1, "layer": Conv2D, "lr": 0.0005, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "lr": 0.0005}, {"layer": Relu},

    {'neurons': 512, "lr": 0.0005, "bias": False}, {"layer": BatchNorm, "lr": 0.0005}, {"layer": Relu},
    {'neurons': 64, "bias": False}, {"layer": BatchNorm, "lr": 0.0005}, {"layer": Relu},
    {'neurons': 8, "lr": 0.0005}, {"layer": Relu},
    {'neurons': 1, "lr": 0.0005, "init_func": xavier_uniform}]  # 0.000643


def get_metrics(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def train_and_save(model, loader, name, X_test=None, y_test=None, epochs=50, early_stop=10, folder="Models",
                   min_delta=0.001):
    var = model.train(loader, epochs=epochs, early_stop=early_stop, x_test=X_test, y_test=y_test, min_delta=min_delta)
    with open(f"{folder}/{name}.pkl", "wb") as f:
        pickle.dump(var, f)

    return var


def best_threshold_Logits(model, X, y, metric="f1"):
    best_val = {metric: float("-inf")}
    best_thr = None
    result = Sigmoid().forward(model.predict(AsyncCupyDataLoader(X, shuffle=False, batch_size=512), numpy=False)).get()
    for threshold in range(1, 100):
        threshold /= 100
        val = get_metrics(result > threshold, y)

        if val[metric] >= best_val[metric]:
            best_val = val
            best_thr = threshold

    return best_thr, best_val


def append_sigmoid(name):
    with open(f"Models/{name}.pkl", "rb") as f:
        var = pickle.load(f)

    var.append({"layer": Sigmoid})
    with open(f"Models/{name}.pkl", "wb") as f:
        pickle.dump(var, f)


if __name__ == "__main__":
    if "final" in sys.argv:

        X_base = np.load(rf"Data\X_nodup.npy").astype(np.float32) / np.asarray(255, dtype=np.float32)
        y_base = np.load(rf"Data\y_nodup.npy")
        final_agent = NeuralNetwork(var_3, BCELogits(), Adam(scheduler=InverseSqrtSchedulerMod1(warmup_steps=3500)))
        train_and_save(final_agent, AsyncCupyDataLoader(X_base, y_base, batch_size=128), "final_agent",
                       epochs=30, X_test=X_base, y_test=y_base)
        append_sigmoid("final_agent")
        thr_final_agent, final_agent_metrics = best_threshold_Logits(final_agent, X_base, y_base)

        print(f"{thr_final_agent=}, {final_agent_metrics=}")

        with open(rf"Models\nets_thresholds.json", "r") as f:
            thresholds = json.load(f)

        with open(rf"Models\nets_thresholds.json", "w") as f:
            thresholds |= {"final_agent": thr_final_agent}
            json.dump(thresholds, f)

        del final_agent
        gc.collect()  # очищает память CPU, а CuPy освобождает привязанную GPU-память
        cp._default_memory_pool.free_all_blocks()
    else:
        X_base = np.load(rf"Data\pixels_all_aug.npy").astype(np.float32) / np.asarray(255, dtype=np.float32)
        y_base = np.load(rf"Data\target_all_aug_cls.npy")

        X_game_over = np.load(r"Data\game_over_data_aug.npy").astype(np.float32) / np.asarray(255, dtype=np.float32)
        y_game_over = np.load(r"Data\game_over_target_aug.npy")

        X_success = np.load(r"Data\success_collect.npy").astype(np.float32) / np.asarray(255, dtype=np.float32)
        y_success = np.load(r"Data\success_collect_target.npy")

        online_agent = NeuralNetwork(var_3, BCELogits(), Adam())
        game_over_net = NeuralNetwork(var_1, BCELogits(), Adam())
        success_net = NeuralNetwork(var_2, BCELogits(), Adam())

        train_and_save(online_agent, AsyncCupyDataLoader(X_base, y_base, batch_size=128), "online_agent",
                       epochs=1, X_test=X_base, y_test=y_base)

        train_and_save(game_over_net, AsyncCupyDataLoader(X_game_over, y_game_over, batch_size=64), "game_over_net",
                       epochs=15, X_test=X_game_over, y_test=y_game_over)

        train_and_save(success_net, AsyncCupyDataLoader(X_success, y_success, batch_size=128), "success_net",
                       epochs=5, X_test=X_success, y_test=y_success)

        append_sigmoid("online_agent")
        append_sigmoid("game_over_net")
        append_sigmoid("success_net")

        thr_online_agent, online_agent_metrics = best_threshold_Logits(online_agent, X_base, y_base)
        thr_game_over_net, game_over_net_metrics = best_threshold_Logits(game_over_net, X_game_over, y_game_over)
        thr_success_net, success_net_metrics = best_threshold_Logits(success_net, X_success, y_success)

        print(f"{thr_online_agent=}, {online_agent_metrics=}\n"
              f"{thr_game_over_net=}, {game_over_net_metrics=}\n"
              f"{thr_success_net=}, {success_net_metrics=}")

        with open(rf"Models\nets_thresholds.json", "w") as f:
            json.dump(
                {"online_agent": thr_online_agent,
                 "game_over_net": thr_game_over_net,
                 "success_net": thr_success_net}, f)

        del game_over_net, success_net, online_agent
        gc.collect()  # очищает память CPU, а CuPy освобождает привязанную GPU-память
        cp._default_memory_pool.free_all_blocks()
