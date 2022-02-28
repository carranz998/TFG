import os

import nibabel as nib
import numpy as np

from src.utils.find_file import get_abspath


class ImageLoader:
    @classmethod
    def get_data(cls, name_image: str):
        abspath_image = get_abspath(name_image)
        image_data = cls.__get_image_data(abspath_image)

        return image_data

    @staticmethod
    def __get_image_data(abspath_image: os.path.abspath) -> np.ndarray:
        return nib.load(abspath_image).get_fdata()
