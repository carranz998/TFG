import argparse
import sys


class CommandLineParser:
    def __init__(self) -> None:
        self.argument_parser = argparse.ArgumentParser()
        self.argument_parser.add_argument('--abspath_directory_images')

        self.command_line_arguments = sys.argv[1:]

    @property
    def parsed_args(self) -> argparse.Namespace:
        return self.argument_parser.parse_args(self.command_line_arguments)