from math import log2

import torch
from tqdm import tqdm
from data_preprocessing.input_image_preprocessor import InputImagePreprocessor
from gan.model.components.discriminator import Discriminator
from gan.model.components.generator import Generator
from gan.training.train import train_all
import gan.config.hyperparameters as hyperparameters

from utils.file_management.directory_iterator import DirectoryIterator

if __name__ == "__main__":
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(Z_DIM, IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")

    path_subfiles = DirectoryIterator.expanded_subtree(hyperparameters.PATH_DATASET)

    input_image_preprocessor = InputImagePreprocessor()
    map(input_image_preprocessor.delete_axis, tqdm(list(path_subfiles)))

    train_all()
