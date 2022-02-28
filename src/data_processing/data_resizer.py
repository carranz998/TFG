import numpy as np
from skimage.transform import resize

from src.utils.constants import IMG_DIM
from src.utils.vector_transformations import normalize, scale


class DataResizer:
    @classmethod
    def custom_resize(cls, data: np.ndarray) -> np.ndarray:
        normalized: np.ndarray = np.clip(normalize(data) - 0.1, 0, 1)
        scaled: np.ndarray = scale(normalized ** 0.4, 2)
        transformed: np.ndarray = np.clip(scaled - 0.1, 0, 1)
        resized: np.ndarray = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')

        return resized
