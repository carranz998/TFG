from torch.utils.data import DataLoader
import torchvision.datasets
from data_preprocessing.input_image_preprocessor import InputImagePreprocessor
from utils.command_line_parser import CommandLineParser
from utils.file_management.directory_iterator import DirectoryIterator


class Application:
    def __init__(self) -> None:
        command_line_parser = CommandLineParser()
        self.command_line_parsed_args = command_line_parser.parsed_args

    def execute(self) -> None:
        abspath_directory_images = self.command_line_parsed_args.abspath_directory_images    

        dataset = torchvision.datasets.ImageFolder(root=abspath_directory_images)
        batch_size = 2
        data_loader = DataLoader(dataset, batch_size=batch_size)

        abspath_subfiles = DirectoryIterator.subtree(abspath_directory_images)

        image_axis_deleter = InputImagePreprocessor()

        for abspath_subfile in abspath_subfiles:
            image_axis_deleter.delete_axis(abspath_subfile)
            break
