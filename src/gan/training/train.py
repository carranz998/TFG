import os
from math import log2
from typing import Any, Dict
from dataset_operations.dataset_resolution_adapter import DatasetResolutionAdapter

import gan.config.hyperparameters as hyperparameters
import torch
from gan.model.pro_gan import ProGAN
from gan.utils.fid import fid_main
from gan.utils.gradient_penalty import gradient_penalty
from gan.utils.tensorboard_plot import plot_to_tensorboard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.backends.cudnn.benchmarks = True


def calculate_step() -> int:
    return int(log2(hyperparameters.START_TRAIN_AT_IMG_SIZE / 4))

def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(hyperparameters.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, hyperparameters.Z_DIM, 1, 1).to(hyperparameters.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + hyperparameters.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / ((hyperparameters.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(hyperparameters.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
                step
            )
            tensorboard_step += 1

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

    return tensorboard_step, alpha


def train_all():
    pro_gan = ProGAN()

    generator_network = pro_gan.generator_network
    generator_optimizer = pro_gan.generator_optimizer
    generator_scaler = pro_gan.generator_scaler

    discriminator_network = pro_gan.discriminator_network
    discriminator_optimizer = pro_gan.discriminator_optimizer
    discriminator_scaler = pro_gan.discriminator_scaler

    writer = SummaryWriter(f'logs/gan1')

    if hyperparameters.LOAD_MODEL:
        pro_gan.load_configurations()

    generator_network.train()
    discriminator_network.train()

    tensorboard_step = 0
    step = calculate_step()

    for num_epochs in hyperparameters.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        image_size = 4 * 2 ** step

        dataset_components = DatasetResolutionAdapter.adapt_to(image_size)
        loader = dataset_components['dataloader']
        dataset = dataset_components['dataset']
        
        print(f"Current image size: {image_size}")

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            
            tensorboard_step, alpha = train_fn(
                discriminator_network,
                generator_network,
                loader,
                dataset,
                step,
                alpha,
                discriminator_optimizer,
                generator_optimizer,
                tensorboard_step,
                writer,
                generator_scaler,
                discriminator_scaler,
            )

            if hyperparameters.SAVE_MODEL:
                pro_gan.save_configurations()

        calculate_fid(step, tensorboard_step, dataset_components, writer)

        step += 1


def calculate_fid(step: int, tensorboard_step: int, dataset_components: Dict[str, Any], writer: SummaryWriter):
    image_size = 4 * 2 ** step

    dirname_fake_images = os.path.join('generated_images', str(image_size))
    dirname_real_images = os.path.join('real_images', str(image_size))

    fid_value = fid_main(dirname_fake_images, dirname_real_images, batch_size=dataset_components['batch_size'])
    print(f'EL FID VALUE ES: {fid_value}')

    writer.add_scalar('FID', fid_value, global_step=tensorboard_step)