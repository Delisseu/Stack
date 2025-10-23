import albumentations as A
import numpy as np

t_1 = A.Compose([
    A.RandomBrightnessContrast(p=0.9)
])


def add_gaussian_noise(image, mean=0, std=10):
    """
    Добавляет гауссов шум к изображению.

    Args:
        image: numpy array, dtype=np.uint8 или float32, shape (H, W, C) или (H, W)
        mean: среднее значение шума
        std: стандартное отклонение (интенсивность) шума

    Returns:
        noisy_image: изображение с шумом, такого же типа и размера, как input
    """
    # Если изображение в uint8, переводим в float для вычислений
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    noise = np.random.normal(loc=mean, scale=std, size=image.shape)
    noisy_image = image + noise

    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


batch = np.load(r"Data\pixels_all.npy")
target_array_cls = np.load(r"Data\target_all.npy")
target_array_reg = np.load(r"Data\reg_all.npy")

augmented_batch_1 = np.stack([t_1(image=img)["image"] for img in batch])
augmented_batch_2 = np.stack(
    [add_gaussian_noise(img, 0, np.random.randint(1, 8)) for img in batch])
augmented_batch_3 = np.stack(
    [add_gaussian_noise(t_1(image=img)["image"], 0, np.random.randint(1, 8)) for img in batch])

new_batch = np.concatenate([batch, augmented_batch_1, augmented_batch_2, augmented_batch_3], axis=0)
new_target_cls = np.concatenate([target_array_cls] * 4, axis=0)
new_target_reg = np.concatenate([target_array_reg] * 4, axis=0)

np.save(r"Data\pixels_all_aug.npy", new_batch.astype(np.uint8).reshape(-1, 100, 100, 1))
np.save(r"Data\target_all_aug_cls.npy", new_target_cls)
np.save(r"Data\target_all_aug_reg.npy", new_target_reg)

# GameOverNet
game_over_data = np.load(r"Data\game_over_data.npy")
game_over_target = np.load(r"Data\game_over_target.npy")

augmented_game_over_1 = np.stack([t_1(image=img)["image"] for img in game_over_data])
augmented_game_over_2 = np.stack(
    [add_gaussian_noise(img, 0, np.random.randint(1, 4)) for img in game_over_data])
augmented_game_over_3 = np.stack(
    [add_gaussian_noise(t_1(image=img)["image"], 0, np.random.randint(1, 2)) for img in game_over_data])

aug_over_target = np.concatenate([game_over_target] * 4, axis=0)
new_game_over = np.concatenate([game_over_data, augmented_game_over_1, augmented_game_over_2, augmented_game_over_3],
                               axis=0)

indices = np.random.permutation(len(batch))
false_data = new_batch[indices[:len(new_game_over)]]
game_over_data = np.concatenate([new_game_over, false_data], axis=0)
np.save(r"Data\game_over_data_aug.npy", game_over_data[:, 94:, 25:-25].astype(np.uint8))
game_over_target = np.concatenate([aug_over_target, np.zeros((len(false_data), 1))], axis=0)
np.save(r"Data\game_over_target_aug.npy", game_over_target)
