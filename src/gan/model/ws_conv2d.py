import torch.nn as nn


class WSConv2d(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3 , stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(input_channel, out_channel, kernel_size, stride, padding)
        self.scale = (gain / (input_channel * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)