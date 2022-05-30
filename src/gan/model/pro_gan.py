from math import log2

import gan.config.hyperparameters as hyperparameters
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from gan.model.auxiliar.tensorboard_writer import TensorBoardWriter
from gan.model.builder.pro_gan_builder import ProGANBuilder
from gan.utils.checkpoints import Checkpoint
from gan.utils.gradient_penalty import gradient_penalty
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProGAN:
    def __init__(self) -> None:
        components = ProGANBuilder.build_by_pipeline()
        self.tensorboard_writer = TensorBoardWriter()

        self.generator_network = components['generator_network']
        self.generator_optimizer = components['generator_optimizer']
        self.generator_scaler = components['generator_scaler']

        self.discriminator_network = components['discriminator_network']
        self.discriminator_optimizer = components['discriminator_optimizer']
        self.discriminator_scaler = components['discriminator_scaler']

        if hyperparameters.LOAD_MODEL:
            self.load_configurations()

    def load_configurations(self):
        Checkpoint.load(
            model=self.generator_network, 
            optimizer=self.generator_optimizer, 
            input_filename=hyperparameters.PATH_GENERATOR_CHECKPOINT
        )

        Checkpoint.load(
            model=self.discriminator_network, 
            optimizer=self.discriminator_optimizer, 
            input_filename=hyperparameters.PATH_DISCRIMINATOR_CHECKPOINT
        )

    def save_configurations(self):
        Checkpoint.save(
            model=self.generator_network, 
            optimizer=self.generator_optimizer, 
            output_filename=hyperparameters.PATH_GENERATOR_CHECKPOINT
        )

        Checkpoint.save(
            model=self.discriminator_network, 
            optimizer=self.discriminator_optimizer, 
            output_filename=hyperparameters.PATH_DISCRIMINATOR_CHECKPOINT
        )

    def train(self):
        self.generator_network.train()
        self.discriminator_network.train()

        tensorboard_step = 0
        current_image_size_idx = int(log2(hyperparameters.START_TRAIN_AT_IMG_SIZE / 4))

        for num_epochs in hyperparameters.PROGRESSIVE_EPOCHS[current_image_size_idx:]:
            alpha = 1e-5
            current_image_size = 4 * 2 ** current_image_size_idx
            loader, dataset = self.get_loader(current_image_size)
            
            print(f"Current image size: {current_image_size}")

            for epoch in range(num_epochs):
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                
                tensorboard_step, alpha = self.train_fn(
                    loader,
                    dataset,
                    current_image_size_idx,
                    alpha,
                    tensorboard_step,   
                )

                if hyperparameters.SAVE_MODEL:
                    self.save_configurations()

            current_image_size_idx += 1

    def train_fn(
        self,
        loader,
        dataset,
        step,
        alpha,
        tensorboard_step,
    ):
        loop = tqdm(loader, leave=True)
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(hyperparameters.DEVICE)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
            # which is equivalent to minimizing the negative of the expression
            noise = torch.randn(cur_batch_size, hyperparameters.Z_DIM, 1, 1).to(hyperparameters.DEVICE)

            with torch.cuda.amp.autocast():
                fake = self.generator_network(noise, alpha, step)
                critic_real = self.discriminator_network(real, alpha, step)
                critic_fake = self.discriminator_network(fake.detach(), alpha, step)
                gp = gradient_penalty(self.discriminator_network, real, fake, alpha, step, device=hyperparameters.DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + hyperparameters.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            self.generator_optimizer.zero_grad()
            self.discriminator_scaler.scale(loss_critic).backward()
            self.discriminator_scaler.step(self.discriminator_optimizer)
            self.discriminator_scaler.update()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                gen_fake = self.discriminator_network(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            self.generator_optimizer.zero_grad()
            self.generator_scaler.scale(loss_gen).backward()
            self.generator_scaler.step(self.generator_optimizer)
            self.generator_scaler.update()

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / ((hyperparameters.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
            alpha = min(alpha, 1)

            if batch_idx % 500 == 0:
                with torch.no_grad():
                    fixed_fakes = self.generator_network(hyperparameters.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                plot_to_tensorboard(
                    loss_critic.item(),
                    loss_gen.item(),
                    real.detach(),
                    fixed_fakes.detach(),
                    tensorboard_step,
                )
                tensorboard_step += 1

            loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

        return tensorboard_step, alpha
