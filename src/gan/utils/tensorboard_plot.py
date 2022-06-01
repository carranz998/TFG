import os

import torch
import torchvision


def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step: int
):
    writer.add_scalar('Loss Generator', loss_gen, global_step=tensorboard_step)
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)

