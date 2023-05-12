from typing import List
from pathlib import Path

def dir_path(string) -> Path:
    """Helper for argument parsing. Ensures that the provided string is a directory."""
    path = Path(string)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(string)
    
def parse_image_dimensions(list: List[int]) -> tuple:
    if len(list) != 2:
        raise ValueError("Image dimensions must be a list of length 2.")
    list = [int(x) for x in list]
    return tuple(list)