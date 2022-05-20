import torch
import torch.nn as nn


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, 
        alpha: float, train_step: int, device: str='cpu'):
    batch_size, channels, height, width = real.shape
    beta = torch \
        .rand((batch_size, 1, 1, 1)) \
        .repeat(1, channels, height, width) \
        .to(device)
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

    print(gradient_penalty == GradientCalculations.get_gradient_penalty(critic, alpha, train_step, real, fake, device))

    return gradient_penalty


class GradientCalculations:
    @classmethod
    def get_gradient_penalty(cls, discriminator: nn.Module, alpha: float, train_step: int, 
            real: torch.Tensor, fake: torch.Tensor, device: str='cpu'):
        interpolated_images = cls.__interpolate_images(real, fake, device)
        mixed_scores = cls.__calculate_discriminator_scores(discriminator, interpolated_images, alpha, train_step)
        gradient = cls.__calculate_gradient(interpolated_images, mixed_scores)
        gradient_penalty = cls.__calculate_gradient_penalty(gradient)
        
        return gradient_penalty

    @classmethod
    def __calculate_discriminator_scores(cls, discriminator: nn.Module, interpolated_images: torch.Tensor, 
            alpha: float, train_step: int):
        mixed_scores = discriminator(interpolated_images, alpha, train_step)

        return mixed_scores

    @classmethod
    def __calculate_gradient(cls, interpolated_images: torch.Tensor, mixed_scores: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)

        return gradient

    @classmethod
    def __calculate_gradient_penalty(cls, gradient: torch.Tensor) -> torch.Tensor:
        gradient_norm = torch.linalg.norm(gradient, 2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    @classmethod
    def __interpolate_images(cls, real: torch.Tensor, fake: torch.Tensor, device: str='cpu') -> torch.Tensor:
        batch_size, channels, height, width = real.shape

        beta = torch \
            .rand((batch_size, 1, 1, 1)) \
            .repeat(1, channels, height, width) \
            .to(device)

        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        return interpolated_images