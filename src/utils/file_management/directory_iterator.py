import os
from typing import Iterator

from utils.file_management.file_decompressor import FileDecompressor


class DirectoryIterator:
    @classmethod
    def expanded_subtree(cls, root_directory_path: os.PathLike) -> Iterator[os.PathLike]:
        visited_paths, unvisited_paths = set(), [root_directory_path]

        while unvisited_paths:
            directory_item_path = unvisited_paths.pop()
            if directory_item_path not in visited_paths:
                visited_paths.add(directory_item_path)

                if cls.__is_compressed_file(directory_item_path):
                    FileDecompressor.decompress_file(directory_item_path)
                    decompressed_directory_path, _ = os.path.splitext(directory_item_path)
                    directory_item_path = decompressed_directory_path

                if os.path.isdir(directory_item_path):
                    directory_items_paths = set(cls.subtree_first_level(directory_item_path))
                    directory_unvisited_paths = directory_items_paths - visited_paths
                    unvisited_paths.extend(directory_unvisited_paths)

                if os.path.isfile(directory_item_path):
                    yield directory_item_path

    @classmethod
    def subtree_first_level(cls, root_directory_path: os.PathLike | str) -> Iterator[os.PathLike | str]:
        directory_contents_basenames = os.listdir(root_directory_path)

        for directory_item_basename in directory_contents_basenames:
            directory_item_path = os.path.join(root_directory_path, directory_item_basename)

            yield directory_item_path

    @classmethod
    def __is_compressed_file(cls, item_path: os.PathLike | str) -> bool:
        if os.path.isfile(item_path):
            basename = os.path.basename(item_path)
            _, extension = os.path.splitext(basename)

            if extension == '.zip':
                return True

        return False