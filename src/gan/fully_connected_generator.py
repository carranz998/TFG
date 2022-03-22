from torch import nn

from src.gan.generator import Generator


def create_generator_block(input_dimension, output_dimension) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.BatchNorm1d(output_dimension),
        nn.ReLU(inplace=True),
    )


class FullyConnectedGenerator(Generator):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(FullyConnectedGenerator, self).__init__()
        self.model = nn.Sequential(
            create_generator_block(z_dim, hidden_dim),
            create_generator_block(hidden_dim, hidden_dim * 2),
            create_generator_block(hidden_dim * 2, hidden_dim * 4),
            create_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.model(noise)

    def get_model(self):
        return self.model
