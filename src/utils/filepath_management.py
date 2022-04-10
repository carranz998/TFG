import os
from typing import Iterator


class FilepathManagement:
    @classmethod
    def get_abspath_subfiles(cls, abspath_root_folder: os.path.abspath) -> Iterator[os.path.abspath]:
        contents_root_folder = os.walk(abspath_root_folder, topdown=False)

        for abspath_subfolder, _, basenames_subfolder in contents_root_folder:
            for basename in basenames_subfolder:
                abspath_subfile = os.path.join(abspath_subfolder, basename)

                yield abspath_subfile