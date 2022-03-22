from typing import Callable

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from src.gan.discriminator import Discriminator
from src.gan.fully_connected_generator import FullyConnectedGenerator
from src.gan.generator import Generator


class VanillaGAN:
    def __init__(self, generator_network: Generator, discriminator: Discriminator, learning_rate: float):
        torch.manual_seed(0)

        self.generator_network = generator_network
        self.discrimination_network = discriminator

        self.generator_optimization = torch.optim.Adam(self.generator_network.parameters(), lr=learning_rate)
        self.discriminator_optimization = torch.optim.Adam(self.discrimination_network.parameters(), lr=learning_rate)

    @staticmethod
    def __plot_images_from_tensor(image_tensor: torch.Tensor, num_images=25, image_dimensions=(1, 28, 28)) -> None:
        images = image_tensor.detach().cpu().view(-1, *image_dimensions)
        images_grid = make_grid(images[:num_images], nrow=5)

        plt.imshow(images_grid.permute(1, 2, 0).squeeze())
        plt.show()

    @staticmethod
    def __discriminator_loss(generator_network: FullyConnectedGenerator, discrimination_network: Discriminator,
                             real_images: torch.Tensor, batch_size: int, z_dimension: int,
                             device: str, loss_function: Callable) -> torch.Tensor:
        noise = torch.randn(size=[batch_size, z_dimension], device=device)
        fake_images = generator_network(noise).detach()

        discriminator_prediction_fake = discrimination_network(fake_images)
        loss_fake = loss_function(discriminator_prediction_fake, torch.zeros_like(discriminator_prediction_fake))

        discriminator_prediction_real = discrimination_network(real_images)
        loss_real = loss_function(discriminator_prediction_real, torch.ones_like(discriminator_prediction_real))

        discriminator_loss = (loss_fake + loss_real) * .5

        return discriminator_loss

    @staticmethod
    def __generator_loss(generator_network: FullyConnectedGenerator, discrimination_network: Discriminator, batch_size: int,
                         z_dimension: int, device: str, loss_function: Callable) -> torch.Tensor:
        noise = torch.randn(size=[batch_size, z_dimension], device=device)
        fakes = generator_network(noise)

        discriminator_prediction_fake = discrimination_network(fakes)
        generator_loss = loss_function(discriminator_prediction_fake, torch.ones_like(discriminator_prediction_fake))

        return generator_loss

    def train(self, data_loader: DataLoader, batch_size: int, device: str,
              display_step: int, loss_function: Callable, n_epochs: int, z_dimension: int):
        current_step = 0
        mean_discriminator_loss = 0
        mean_generator_loss = 0

        for epoch in range(n_epochs):
            for real_images, _ in tqdm(data_loader):
                current_batch_size = len(real_images)
                real_images = real_images.view(current_batch_size, -1).to(device)

                # Discriminator update
                self.discriminator_optimization.zero_grad()
                discriminator_loss = self.__discriminator_loss(
                    self.generator_network, self.discrimination_network,
                    real_images, batch_size, z_dimension, device, loss_function
                )
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimization.step()

                # Generator update
                self.generator_optimization.zero_grad()
                generator_loss = self.__generator_loss(
                    self.generator_network, self.discrimination_network,
                    batch_size, z_dimension, device, loss_function
                )
                generator_loss.backward(retain_graph=True)
                self.generator_optimization.step()

                # Book keeping
                mean_discriminator_loss += discriminator_loss.item() / display_step
                mean_generator_loss += generator_loss.item() / display_step

                # Visualize current model output
                if current_step % display_step == 0 and current_step > 0:
                    print(f"Step {current_step}: "
                          f"Generator loss: {mean_generator_loss}, "
                          f"discriminator loss: {mean_discriminator_loss}")

                    fake_noise = torch.randn(size=[current_batch_size, z_dimension], device=device)
                    fake_images = self.generator_network(fake_noise)

                    self.__plot_images_from_tensor(fake_images)
                    self.__plot_images_from_tensor(real_images)

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                current_step += 1
