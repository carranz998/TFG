import torch
import torch.nn as nn
import gan.config.hyperparameters as hyperparameters


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, 
        alpha: float, train_step: int, device: str='cpu'):
    batch_size, channels, height, width = real.shape
    beta = torch \
        .rand((batch_size, 1, 1, 1)) \
        .repeat(1, channels, height, width) \
        .to(hyperparameters.DEVICE)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
