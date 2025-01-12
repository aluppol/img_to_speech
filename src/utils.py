from pathlib import Path
import pickle


def get_file_name_from_path(file_path: str):
  return Path(file_path).stem

def save_to_pkl(data, path_with_no_ext: str):
  with open(f'{path_with_no_ext}.pkl', 'wb') as f:
    pickle.dump(data, f)

def count_files_in_dir(directory: str):
  dir_path = Path(directory)
  
  if not dir_path.is_dir():
    raise ValueError(f"{directory} is not a valid directory.")
    
  return sum(1 for item in dir_path.iterdir() if item.is_file())