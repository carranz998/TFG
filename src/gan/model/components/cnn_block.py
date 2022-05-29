import torch
import torch.nn as nn
from gan.model.components.pixel_norm import PixelNorm
from gan.model.components.ws_conv2d import WSConv2d


class CNNBlock(nn.Module):
    def __init__(self, input_channel: int, out_channel: int, pixel_norm: bool=True) -> None:
        super(CNNBlock, self).__init__()
        
        self.convolutional_blocks = [
            WSConv2d(input_channel, out_channel), 
            WSConv2d(out_channel, out_channel)
        ]

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()
        self.use_pixel_norm = pixel_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.convolutional_blocks:
            convolution = block(x)
            x = self.leaky_relu(convolution)
            
            if self.use_pixel_norm:
                x = self.pixel_norm(x)

        return x
