from math import log2

import gan.config.hyperparameters as hyperparameters
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from gan.model.discriminator import Discriminator
from gan.model.generator import Generator
from gan.utils.checkpoints import Checkpoint
from gan.utils.gradient_penalty import gradient_penalty
from gan.utils.tensorboard_plot import plot_to_tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.backends.cudnn.benchmarks = True


def calculate_step() -> int:
    return int(log2(hyperparameters.START_TRAIN_AT_IMG_SIZE / 4))


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(hyperparameters.CHANNELS_IMG)],
                [0.5 for _ in range(hyperparameters.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = hyperparameters.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=hyperparameters.DATASET_PATH, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=hyperparameters.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


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
            gp = gradient_penalty(critic, real, fake, alpha, step, device=hyperparameters.DEVICE)
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
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step, alpha


def train_all():
    generator_network = Generator(hyperparameters.Z_DIM, hyperparameters.IN_CHANNELS, img_channels=hyperparameters.CHANNELS_IMG) \
        .to(hyperparameters.DEVICE)

    discriminator_network = Discriminator(hyperparameters.Z_DIM, hyperparameters.IN_CHANNELS, img_channels=hyperparameters.CHANNELS_IMG) \
        .to(hyperparameters.DEVICE)

    # initialize optimizers and scalers for FP16 training
    generator_optimizer = optim.Adam(generator_network.parameters(), lr=hyperparameters.LEARNING_RATE, betas=(0.0, 0.99))
    discriminator_optimizer = optim.Adam(
        discriminator_network.parameters(), lr=hyperparameters.LEARNING_RATE, betas=(0.0, 0.99)
    )
    discriminator_scaler = torch.cuda.amp.GradScaler()
    generator_scaler = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan1")

    if hyperparameters.LOAD_MODEL:
        Checkpoint.load(
            generator_network, generator_optimizer, hyperparameters.GENERATOR_CHECKPOINT, hyperparameters.LEARNING_RATE,
        )
        Checkpoint.load(
            discriminator_network, discriminator_optimizer, hyperparameters.DISCRIMINATOR_CHECKPOINT, hyperparameters.LEARNING_RATE,
        )

    generator_network.train()
    discriminator_network.train()

    tensorboard_step = 0
    step = calculate_step()

    for num_epochs in hyperparameters.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
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
                Checkpoint.save(
                    generator_network, generator_optimizer, output_filename=hyperparameters.GENERATOR_CHECKPOINT)
                Checkpoint.save(
                    discriminator_network, discriminator_optimizer, output_filename=hyperparameters.DISCRIMINATOR_CHECKPOINT)

        step += 1  # progress to the next img size
