from pathlib import Path


def get_file_name_from_path(file_path: str):
  return Path(file_path).stem