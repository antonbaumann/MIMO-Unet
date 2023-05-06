from pathlib import Path

def dir_path(string) -> Path:
    """Helper for argument parsing. Ensures that the provided string is a directory."""
    path = Path(string)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(string)