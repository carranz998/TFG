from torch import nn


def create_generator_block(input_dimension, output_dimension) -> nn.Sequential:
    """
    Assembles a block consisting of linear layer, batchnorm and a final ReLU activation function.
    :param input_dimension: dimension of the input feature vector (scalar)
    :param output_dimension: dimension of the output feature vector (scalar)
    :return: a nn.Sequential object containing the block elements.
    """
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.BatchNorm1d(output_dimension),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        """
        Constructs an object of the Generator class.

        :param z_dim: dimension of the noise vector
        :param im_dim: dimension of the flattened image tensor
        :param hidden_dim: dimension of the first hidden layer tensor
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            create_generator_block(z_dim, hidden_dim),
            create_generator_block(hidden_dim, hidden_dim * 2),
            create_generator_block(hidden_dim * 2, hidden_dim * 4),
            create_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_model(self):
        return self.gen
