import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import torchvision


def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step: int, step: int
):
    writer.add_scalar('Loss Generator', loss_gen, global_step=tensorboard_step)
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image('Real', img_grid_real, global_step=tensorboard_step)
        writer.add_image('Fake', img_grid_fake, global_step=tensorboard_step)

        save_unpacked_image_grid(tensorboard_step, step, img_grid_real, base_dirname='generated_images')
        save_unpacked_image_grid(tensorboard_step, step, img_grid_fake, base_dirname='real_images')


def save_unpacked_image_grid(tensorboard_step, step, img_grid_real, base_dirname):
    filename_output_image = f'{tensorboard_step}.jpeg'
    image_size = 4 * 2 ** step

    dirname = os.path.join(base_dirname, str(image_size))
    basepath_output_images = os.path.join(dirname, filename_output_image)

    create_file(basepath_output_images)

    torchvision.utils.save_image(img_grid_real, basepath_output_images)
    split_image(basepath_output_images, image_size)

    os.remove(basepath_output_images)


def split_image(filepath: str, image_size: int) -> None:
    im = skimage.io.imread(filepath) 

    if image_size == 4:
        n_subimages = 4
    else:
        n_subimages = 8

    for i in range(n_subimages):
        corner_pixels = subimage_corner_pixels(2, image_size, i)
        subimage = crop(im, corner_pixels)

        new_basename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)

        filename, extension = os.path.splitext(new_basename)
        new_basename = f'{filename}-{i}{extension}'
        new_filepath = os.path.join(os.getcwd(), dirname, new_basename)

        plt.imsave(new_filepath, subimage)


def create_file(filepath: str) -> None:
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)

    with open(filepath, 'w'): 
        pass


def subimage_corner_pixels(n_padding_pixels: int, image_size: int, image_counter: int) -> Tuple[Tuple[int, int], ...]:
    padding_per_image = n_padding_pixels + ((image_size + n_padding_pixels) * image_counter)
    
    upper_left = (n_padding_pixels, padding_per_image)
    upper_right = (n_padding_pixels, padding_per_image + image_size) 
    lower_left = (n_padding_pixels + image_size, padding_per_image)
    lower_right = (n_padding_pixels + image_size, padding_per_image + image_size)
    
    return upper_left, upper_right, lower_left, lower_right
    

def crop(image: np.ndarray, corner_pixels: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    upper_left, upper_right, lower_left, _ = corner_pixels

    subimage = image[
        upper_left[0]: lower_left[0], 
        upper_left[1]: upper_right[1]
    ]

    return subimage
