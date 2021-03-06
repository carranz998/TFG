from math import log2

import torch

START_TRAIN_AT_IMG_SIZE = 16
PATH_DATASET = 'C:\\Users\\carra\\OneDrive\\Escritorio\\cositas'
PATH_GENERATOR_CHECKPOINT = 'generator.pth'
PATH_DISCRIMINATOR_CHECKPOINT = 'discriminator.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 512
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1

PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4

FACTORS = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
