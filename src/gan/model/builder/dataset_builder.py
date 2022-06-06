from math import log2
from typing import Any

import gan.config.hyperparameters as hyperparameters
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pipetools import pipe
from torch.utils.data import DataLoader


class DatasetBuilder:
    @classmethod
    def __init_processed_images_dataset(cls, created_components: dict[str, Any]) -> dict[str, Any]:
        image_size = created_components['image_size']

        image_transformations = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * hyperparameters.CHANNELS_IMG, 
                [0.5] * hyperparameters.CHANNELS_IMG
            )
        ]

        transformation_pipeline = transforms.Compose(image_transformations)
        dataset = datasets.ImageFolder(hyperparameters.PATH_DATASET, transformation_pipeline)

        created_components['dataset'] = dataset

        return created_components

    @classmethod
    def __init_data_loader(cls, created_components: dict[str, Any]) -> dict[str, Any]:
        dataset = created_components['dataset']
        image_size = created_components['image_size']

        dataloader = DataLoader(
            dataset,
            batch_size=image_size,
            shuffle=True,
            num_workers=hyperparameters.NUM_WORKERS,
            pin_memory=True,
        )

        created_components['dataloader'] = dataloader

        return created_components

    @classmethod
    def __init_batch_size(cls, created_components: dict[str, Any]):
        image_size = created_components['image_size']
        batch_size = hyperparameters.BATCH_SIZES[int(log2(image_size / 4))]

        created_components['batch_size'] = batch_size

        return created_components

    @classmethod
    def build_components(cls, image_size: int):
        default = {'image_size': image_size}

        created_components = cls.__init_batch_size(default)
        created_components = cls.__init_processed_images_dataset(created_components)
        created_components = cls.__init_data_loader(created_components)

        return created_components
