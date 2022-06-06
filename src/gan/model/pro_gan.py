import gan.config.hyperparameters as hyperparameters
from gan.model.builder.pro_gan_builder import ProGANBuilder
from gan.utils.checkpoints import Checkpoint


class ProGAN:
    def __init__(self) -> None:
        components = ProGANBuilder.build_by_pipeline()

        self.generator_network = components['generator_network']
        self.generator_optimizer = components['generator_optimizer']
        self.generator_scaler = components['generator_scaler']

        self.discriminator_network = components['discriminator_network']
        self.discriminator_optimizer = components['discriminator_optimizer']
        self.discriminator_scaler = components['discriminator_scaler']

        if hyperparameters.LOAD_MODEL:
            self.load_configurations()

    def load_configurations(self) -> None:
        Checkpoint.load(
            model=self.generator_network, 
            optimizer=self.generator_optimizer, 
            input_filepath=hyperparameters.PATH_GENERATOR_CHECKPOINT
        )

        Checkpoint.load(
            model=self.discriminator_network, 
            optimizer=self.discriminator_optimizer, 
            input_filepath=hyperparameters.PATH_DISCRIMINATOR_CHECKPOINT
        )

    def save_configurations(self) -> None:
        Checkpoint.save(
            model=self.generator_network, 
            optimizer=self.generator_optimizer, 
            output_filepath=hyperparameters.PATH_GENERATOR_CHECKPOINT
        )

        Checkpoint.save(
            model=self.discriminator_network, 
            optimizer=self.discriminator_optimizer, 
            output_filepath=hyperparameters.PATH_DISCRIMINATOR_CHECKPOINT
        )
