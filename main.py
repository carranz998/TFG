from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data_gathering.image_loader import ImageLoader
from src.data_processing.processing_pipeline import ProcessingPipeline
from src.data_visualization.cut_3d_view_visualization import Cut3dViewVisualization
from src.gan.hyperparameters import BATCH_SIZE, DEVICE, DISPLAY_STEP, LEARNING_RATE, LOSS_FUNCTION, N_EPOCHS, \
    Z_DIMENSION
from src.gan.vanilla_gan import VanillaGAN


raw_image_data = ImageLoader.get_image_data('20050317215037-41818.MRC15540.nii.gz')
processed_image_data = ProcessingPipeline.process(raw_image_data)
Cut3dViewVisualization.plot_cube(processed_image_data[:35, ::-1, :25])

dataset = MNIST('.', download=True, transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
vanilla_gan = VanillaGAN(DEVICE, LEARNING_RATE, Z_DIMENSION)

vanilla_gan.train(data_loader, BATCH_SIZE, DEVICE, DISPLAY_STEP, LOSS_FUNCTION, N_EPOCHS, Z_DIMENSION)
