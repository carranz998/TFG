import nibabel as nib
import numpy as np

from src.utils.find_file import get_abspath


class ImageLoader:
    @classmethod
    def get_image_data(cls, image_file_name: str) -> np.ndarray:
        image_abspath = get_abspath(image_file_name)
        image_data = nib.load(image_abspath).get_fdata()

        return image_data
