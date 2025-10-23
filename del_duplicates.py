import numpy as np


def find_exact_duplicates(images):
    """
    Находит индексы дублирующихся изображений.

    images: np.array, shape=(N, H, W) или (N, H, W, 1)

    return: список списков индексов дубликатов
    """
    N = images.shape[0]

    # Уберем лишнюю размерность, если есть
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.reshape(N, images.shape[1], images.shape[2])

    seen = {}
    duplicates = []

    for idx, img in enumerate(images):
        # Превращаем изображение в bytes для быстрого сравнения
        img_bytes = img.tobytes()
        if img_bytes in seen:
            seen[img_bytes].append(idx)
        else:
            seen[img_bytes] = [idx]

    # Оставляем только группы с повторениями
    for group in seen.values():
        if len(group) > 1:
            duplicates.append(group)

    return duplicates


def remove_duplicates(X, y, y_reg, duplicates):
    """
    Удаляет дубликаты из X и усредняет y по группам.

    X: np.ndarray (N, H, W, C)
    y: np.ndarray (N, ...) — метки
    duplicates: list[list[int]] — группы индексов-дубликатов

    return: (X_new, y_new)
    """
    keep_indices = []
    y_new = []
    y_new_reg = []

    used = set()
    for group in duplicates:
        group = list(set(group))  # на всякий случай убираем повтор индексов
        # выбираем первый индекс для сохранения картинки
        keep_idx = group[0]
        keep_indices.append(keep_idx)
        used.update(group)

        # усредняем метки по группе
        y_avg = np.mean(y[group], axis=0)
        y_avg_reg = np.mean(y_reg[group], axis=0)

        y_new.append(y_avg)
        y_new_reg.append(y_avg_reg)

    # добавляем те картинки, которые не попали ни в одну группу
    all_indices = set(range(len(X)))
    leftovers = list(all_indices - used)
    keep_indices.extend(leftovers)
    y_new.extend(list(y[leftovers]))
    y_new_reg.extend(list(y_reg[leftovers]))

    # формируем новые массивы
    keep_indices = np.array(keep_indices)
    y_new = np.array(y_new)
    y_new_reg = np.array(y_new_reg)

    # переставляем, чтобы сохранить порядок
    order = np.argsort(keep_indices)
    keep_indices = keep_indices[order]
    y_new = y_new[order]
    y_new_reg = y_new_reg[order]
    return X[keep_indices], y_new, y_new_reg


images = np.load(r"Data\Online\X.npy")
y = np.load(r"Data\Online\y.npy")
y_reg = np.load(r"Data\Online\y_reg.npy")

duplicates = find_exact_duplicates(
    np.where(images > 160, 160, images))  # Чтобы убрать случайнче шумы по типу звезд на заднем плане
print(f"Найдено {len(duplicates)} групп дубликатов")

s = 0
for group in duplicates:
    s += len(group)
print(f"Найдено {s} дубликатов")

X, y, y_reg = remove_duplicates(images, y, y_reg, duplicates)

np.save(r"Data\X_nodup.npy", X)
np.save(r"Data\y_nodup.npy", y)
np.save(r"Data\y_reg_nodup.npy", y_reg)

