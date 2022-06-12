import os
import pathlib

import gan.config.hyperparameters as hyperparameters
from data_preprocessing.image_preprocessor import ImagePreprocessor
from tqdm import tqdm
from utils.file_management.directory_iterator import DirectoryIterator


class DatasetPreprocessor:
    @classmethod
    def normalize_dataset_images(cls):
        normalized_path_dataset = pathlib.Path(hyperparameters.PATH_DATASET)
        path_subfiles = DirectoryIterator.expanded_subtree(normalized_path_dataset)

        image_preprocessor = ImagePreprocessor()

        for path_image in tqdm(list(path_subfiles)):
            cls.__normalize_image(path_image, image_preprocessor)

    @classmethod
    def __normalize_image(cls, path_image: pathlib.Path, image_preprocessor: ImagePreprocessor) -> None:
        try:
            image_preprocessor.delete_axis(path_image)
        except IndexError:
            os.remove(path_image)
