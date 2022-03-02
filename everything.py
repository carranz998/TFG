import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
from src.gan.hyperparameters import *

torch.manual_seed(0)


def plot_images_from_tensor(image_tensor: torch.Tensor, num_images=25, image_dim=(1, 28, 28)):
    """
    Plots images contained in a tensor in a grid.

    :param image_tensor: Flattened tensor containing the images
    :param num_images: The number of images contained in the tensor
    :param image_dim: The dimensions of a single image
    """
    images = image_tensor.detach().cpu().view(-1, *image_dim)
    images_grid = make_grid(images[:num_images], nrow=5)

    plt.imshow(images_grid.permute(1, 2, 0).squeeze())
    plt.show()


def create_noise(n_samples, z_dim, device='cpu'):
    """
    Creates a pytorch tensor with noise sampled from a normal distribution.

    :param n_samples: number of samples to generate, scalar
    :param z_dim: dimension of each noise sample, scalar
    :param device: device type ('cpu' or 'cuda')
    :return: a pytorch tensor with noise
    """
    return torch.randn(size=[n_samples, z_dim], device=device)


def get_disc_loss(gen, disc, loss_function, real_images_tensor, batch_size, z_dim, device):
    """
    Calculate the discriminator's loss with given input values.

    :param gen: generator model
    :param disc: discriminator model
    :param loss_function: the employed loss function
    :param real_images_tensor: batch of real images, tensor
    :param batch_size: the number of images in the batch
    :param z_dim: dimension of the noise vector
    :param device: pytorch device ('CPU' or 'CUDA')
    :return: The loss of the discriminator, torch scalar
    """
    noise = create_noise(batch_size, z_dim, device)
    fakes = gen(noise).detach()
    y_prediction_fake = disc(fakes)
    loss_fakes = loss_function(y_prediction_fake, torch.zeros_like(y_prediction_fake))
    y_prediction_real = disc(real_images_tensor)
    loss_real = loss_function(y_prediction_real, torch.ones_like(y_prediction_real))
    discriminator_loss = (loss_fakes + loss_real) * .5
    return discriminator_loss


def get_gen_loss(gen, disc, loss_function, batch_size, z_dim, device):
    """
    Calculate the generator's loss with given input values.

    :param gen: generator model
    :param disc: discriminator model
    :param loss_function: the employed loss function
    :param batch_size: the number of images in the batch
    :param z_dim: dimension of the noise vector
    :param device: pytorch device ('CPU' or 'CUDA')
    :return: The loss of the generator, torch scalar
    """
    noise = create_noise(batch_size, z_dim, device)
    fakes = gen(noise)
    y_prediction_fake = disc(fakes)
    generator_loss = loss_function(y_prediction_fake, torch.ones_like(y_prediction_fake))
    return generator_loss


# dataset
data_loader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True
)

generator_network = Generator(Z_DIMENSION).to(DEVICE)
generator_optimization = torch.optim.Adam(generator_network.parameters(), lr=LEARNING_RATE)
discrimination_network = Discriminator().to(DEVICE)
discriminator_optimization = torch.optim.Adam(discrimination_network.parameters(), lr=LEARNING_RATE)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False
error = False

for epoch in range(N_EPOCHS):
    for real, _ in tqdm(data_loader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(DEVICE)

        # Discriminator update
        discriminator_optimization.zero_grad()
        disc_loss = get_disc_loss(
            generator_network, discrimination_network, LOSS_FUNCTION, real, cur_batch_size, Z_DIMENSION, DEVICE
        )
        disc_loss.backward(retain_graph=True)
        discriminator_optimization.step()

        # Generator update
        generator_optimization.zero_grad()
        gen_loss = get_gen_loss(
            generator_network, discrimination_network, LOSS_FUNCTION, cur_batch_size, Z_DIMENSION, DEVICE
        )
        gen_loss.backward(retain_graph=True)
        generator_optimization.step()

        # Book keeping
        mean_discriminator_loss += disc_loss.item() / DISPLAY_STEP
        mean_generator_loss += gen_loss.item() / DISPLAY_STEP

        # Visualize current model output
        if cur_step % DISPLAY_STEP == 0 and cur_step > 0:
            print(f"Step {cur_step}: "
                  f"Generator loss: {mean_generator_loss}, "
                  f"discriminator loss: {mean_discriminator_loss}")
            fake_noise = create_noise(cur_batch_size, Z_DIMENSION, device=DEVICE)
            fake = generator_network(fake_noise)
            plot_images_from_tensor(fake)
            plot_images_from_tensor(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
