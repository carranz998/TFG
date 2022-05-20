import os
import zipfile


class FileDecompressor:
    @classmethod
    def decompress_file(cls, compressed_file_path: os.PathLike) -> None:
        directory_path, _ = os.path.split(compressed_file_path)

        with zipfile.ZipFile(compressed_file_path, 'r') as zip_file:
            zip_file.extractall(directory_path)

        os.remove(compressed_file_path)
