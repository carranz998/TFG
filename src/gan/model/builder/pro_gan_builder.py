from typing import Any, Dict

import gan.config.hyperparameters as hyperparameters
import torch.optim as optim
from gan.model.components.discriminator import Discriminator
from gan.model.components.generator import Generator
from pipetools import pipe
from torch.cuda.amp.grad_scaler import GradScaler


class ProGANBuilder:
    @classmethod
    def build_by_pipeline(cls) -> Dict[str, Any]:
        create_all_components = pipe \
            | cls.__init_empty_dictionary \
            | cls.__init_networks \
            | cls.__init_optimizers \
            | cls.__init_scalers

        created_components = create_all_components()

        return created_components

    @classmethod
    def __init_empty_dictionary(cls) -> Dict[None, None]:
        return {}

    @classmethod
    def __init_networks(cls, created_components: Dict[str, Any]) -> Dict[str, Any]:
        generator_network = Generator(
            hyperparameters.Z_DIM, 
            hyperparameters.IN_CHANNELS, 
            img_channels=hyperparameters.CHANNELS_IMG
        )

        discriminator_network = Discriminator(
            hyperparameters.Z_DIM, 
            hyperparameters.IN_CHANNELS, 
            img_channels=hyperparameters.CHANNELS_IMG
        )

        generator_network = generator_network.to(hyperparameters.DEVICE)
        discriminator_network = discriminator_network.to(hyperparameters.DEVICE)

        created_components['generator_network'] = generator_network
        created_components['discriminator_network'] = discriminator_network

        return created_components

    @classmethod
    def __init_optimizers(cls, created_components: Dict[str, Any]) -> Dict[str, Any]:
        generator_network = created_components['generator_network']
        discriminator_network = created_components['discriminator_network']

        generator_optimizer = optim.Adam(
            params=generator_network.parameters(), 
            lr=hyperparameters.LEARNING_RATE, 
            betas=(0.0, 0.99)
        )

        discriminator_optimizer = optim.Adam(
            params=discriminator_network.parameters(), 
            lr=hyperparameters.LEARNING_RATE, 
            betas=(0.0, 0.99)
        )

        created_components['generator_optimizer'] = generator_optimizer
        created_components['discriminator_optimizer'] = discriminator_optimizer

        return created_components

    @classmethod
    def __init_scalers(cls, created_components: Dict[str, Any]) -> Dict[str, Any]:
        generator_scaler = GradScaler()
        discriminator_scaler = GradScaler()

        created_components['generator_scaler'] = generator_scaler
        created_components['discriminator_scaler'] = discriminator_scaler

        return created_components
