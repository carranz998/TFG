from torch import nn


def create_discriminator_block(input_dim, output_dim):
    """
    Assembles a block consisting of linear layer, a final LeakyReLU activation function.
    :param input_dim: dimension of the input feature vector (scalar)
    :param output_dim: dimension of the output feature vector (scalar)
    :return: a nn.Sequential object containing the block elements.
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        """
        Construct a discriminator PyTorch model
        :param im_dim: dimension of the flattened image tensor, scalar
        :param hidden_dim: dimension of the first hidden layer, scalar
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            create_discriminator_block(im_dim, hidden_dim * 4),
            create_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            create_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)

    def get_model(self):
        return self.disc
