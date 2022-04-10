from data_preprocessing.input_image_preprocessor import InputImagePreprocessor
from utils.command_line_parser import CommandLineParser
from utils.filepath_management import FilepathManagement


class Application:
    def __init__(self) -> None:
        command_line_parser = CommandLineParser()
        self.command_line_parsed_args = command_line_parser.parsed_args

    def execute(self) -> None:
        abspath_directory_images = self.command_line_parsed_args.abspath_directory_images
        
        input_image_preprocessor = InputImagePreprocessor()
        abspath_subfiles = FilepathManagement.get_abspath_subfiles(abspath_directory_images)

        for abspath_subfile in abspath_subfiles:
            input_image_preprocessor.delete_axis(abspath_subfile)
