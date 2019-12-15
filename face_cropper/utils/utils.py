import os
from pathlib import Path

__all__ = ['walk_through_files']


def walk_through_files(path: str):
    for dir_path, dir_names, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            p_path = Path(file_path)
            if p_path.is_file():
                yield file_path
