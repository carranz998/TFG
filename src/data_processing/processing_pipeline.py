import numpy as np
from skimage.transform import resize

from src.utils.constants import IMG_DIM
from src.utils.vector_transformations import normalize, scale


class ProcessingPipeline:
    @classmethod
    def process(cls, data: np.ndarray) -> np.ndarray:
        normalized = cls.__normalize(data)
        scaled = cls.__scale(normalized)
        transformed = cls.__transform(scaled)
        resized = cls.__resize(transformed)

        return resized

    @staticmethod
    def __normalize(data: np.ndarray) -> np.ndarray:
        return np.clip(normalize(data) - 0.1, 0, 1)

    @staticmethod
    def __scale(data: np.ndarray) -> np.ndarray:
        return scale(data ** 0.4, 2)

    @staticmethod
    def __transform(data: np.ndarray) -> np.ndarray:
        return np.clip(data - 0.1, 0, 1)

    @staticmethod
    def __resize(data: np.ndarray) -> np.ndarray:
        return resize(data, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')

