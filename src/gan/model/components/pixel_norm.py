import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelNorm, self).__init__()

    def forward(self, x: torch.Tensor, epsilon: float=1e-8) -> torch.Tensor:
        mean_of_squares = torch.mean(x**2, dim=1, keepdim=True)
        normalization_factor = 1 / torch.sqrt(mean_of_squares + epsilon)
        
        return x * normalization_factor
