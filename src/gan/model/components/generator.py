import gan.config.hyperparameters as hyperparameters
import torch
import torch.nn as nn
import torch.nn.functional as F
from gan.model.components.cnn_block import CNNBlock
from gan.model.components.pixel_norm import PixelNorm
from gan.model.components.ws_conv2d import WSConv2d


class Generator(nn.Module):
    def __init__(self, z_dim: int, in_channels: int, img_channels: int=3):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        for i in range(len(hyperparameters.FACTORS) - 1):
            conv_in_c = int(in_channels * hyperparameters.FACTORS[i])
            conv_out_c = int(in_channels * hyperparameters.FACTORS[i + 1])
            self.prog_blocks.append(CNNBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha: float, upscaled: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        upscaled = torch.Tensor()

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)

