import functools
import os.path


@functools.lru_cache(maxsize=16)
def get_abspath(file_name: str, relpath_directory_to_inspect='resources') -> os.path.abspath:
    """
    Given the name of a file, it returns the absolute path to that file.

    :param file_name:

    :param relpath_directory_to_inspect: Directory in which the file may be located

    :return:
    """

    abspath_current_working_directory = os.getcwd()
    abspath_root_directory_to_inspect = os.path.join(abspath_current_working_directory, relpath_directory_to_inspect)
    root_directory_contents = os.walk(abspath_root_directory_to_inspect, topdown=False)

    for abspath_subdirectory, _, subdirectory_files_name in root_directory_contents:
        for subdirectory_file_name in subdirectory_files_name:
            if subdirectory_file_name == file_name:
                abspath_file = os.path.join(abspath_subdirectory, subdirectory_file_name)

                return abspath_file
