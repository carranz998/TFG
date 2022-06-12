from dataset_operations.dataset_preprocessor import DatasetPreprocessor
from gan.training.train import train_all

if __name__ == "__main__":
    print('Preprocess data')
    DatasetPreprocessor.normalize_dataset_images()

    print('Train')
    train_all()
