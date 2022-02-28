import numpy as np


def scale(data: np.ndarray, factor: int) -> np.ndarray:
    mean = np.mean(data)
    return (data - mean) * factor + mean


def normalize(data: np.ndarray) -> np.ndarray:
    minimum_item, maximum_item = np.min(data), np.max(data)
    return (data - minimum_item) / (maximum_item - minimum_item)
