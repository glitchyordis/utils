from typing import Optional, List
import pathlib
import json
import numpy as np
import pandas as pd
import PySimpleGUI as gui

# general
def strtobool(val):
    """
    adapted from
    https://stackoverflow.com/questions/42248342/yes-no-prompt-in-python3-using-strtobool

    Convert a string representation of truth to true (1) or false (0).
    Raises ValueError if 'val' is anything else other than those defined as true or false.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return bool(1)
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return bool(0)
    else:
        raise ValueError("invalid truth value %r" % (val,))

# directory related
def move_files(files: List[pathlib.Path], new_dir: str):
    """
    move files from one directory to new dir
    """
    dir = pathlib.Path(new_dir)
    if not dir.exists() or not dir.is_dir():
        raise UserWarning(f"{dir = } does not exist.")
    else:
        for x in files:
            if x.exists() and x.is_file():
                x.rename(dir/x.name)

def search_files(path: str, file_name: str, exts=Optional[List[str]]):
    """
    Searches through subdir. specified in `path` for files whose name contain `file_name`.
    
    Args:
        path: path to search for file
        file_name: file_name to filter
        exts: extensions of file interested in

    Returns:
        list of filepaths for file whose name contains `file_name`

    # Example usage
    files = search_files(r'E:\Polaris Dataset sorting', '20220811', ['png', 'jpg'])
    for file in files:
        print(file)
    """
    # Create a Path object
    path = pathlib.Path(path)

    # Check if the path exists
    if not path.exists():
        print(f"The path {path} does not exist.")
        return []

    # Prepare the file name pattern
    files = []
    for ext in exts:
        pattern = f"*{file_name}*.{ext}"

        # Search for files
        files.extend(list(path.glob(f"**/{pattern}")))

    # Return the list of file paths
    return [str(file.resolve()) for file in files]

def compare_file_name(path1: str, path2: str, ext: str = "[jp][np]g"):
    """
    Navigates the subdirectories of path1 and path2 and compare filenames of files with extension 
    (jpg and png by default). 
    
    Returns: 
        file names  that are:
        - in path1 and not in path2
        - in path2 and not in path1
        - present in both paths

    Example usage: 
        # path1 = r"<my path on computer>"
        # path2 = r'<my path on computer>'

        in_path1_not_path2, in_path2_not_path1, in_both_paths = compare_files(path1, path2)
        if isinstance(in_path1_not_path2, str):
            print(in_path1_not_path2)
        elif isinstance(in_path2_not_path1, str):
            print(in_path2_not_path1)
        else:
            print(f"{len(in_path1_not_path2)} Files in path1 but not in path2:\n", in_path1_not_path2)
            print(f"{len(in_path2_not_path1)} Files in path2 but not in path1:\n", in_path2_not_path1)
            print(f"{len(in_both_paths)} Files in both paths:\n", in_both_paths)
    """
    proceed = strtobool(gui.popup(f"Confirm using {ext = }?", title = f"{__name__}", button_type=1))

    # Check if the directories exist
    if not pathlib.Path(path1).is_dir():
        return f"Directory {path1} does not exist.", None, None
    if not pathlib.Path(path2).is_dir():
        return None, f"Directory {path2} does not exist.", None
    
    if proceed:
        # Get the set of all .jpg and .png files in path1
        # files_in_path1 = {f.name for f in Path(path1).glob('*.[jp][np]g')} # this searches through dir only
        files_in_path1 = {f.name for f in pathlib.Path(path1).glob(f'**/*.{ext}')} # this searches through subfolder too

        # Get the set of all .jpg and .png files in path2
        # files_in_path2 = {f.name for f in Path(path2).glob('*.[jp][np]g')} # this searches through dir only
        files_in_path2 = {f.name for f in pathlib.Path(path2).glob(f'**/*.{ext}')} # this searches through subfolder too

        # Files in path1 but not in path2
        in_path1_not_path2 = list(files_in_path1 - files_in_path2)

        # Files in path2 but not in path1
        in_path2_not_path1 = list(files_in_path2 - files_in_path1)

        in_both_paths = list(files_in_path1 & files_in_path2)

        return in_path1_not_path2, in_path2_not_path1, in_both_paths
    else:
        raise UserWarning(f"User selected {proceed = }")

# json
class CustomJSONEncoder(json.JSONEncoder):
    """
    Generated with GPT-4. A fix for encoding a dict that may contain numpy_bool and dataframe.
    orient = "split" is to counter edge where if index is not unique, orient = "index" or "columns" causes ValueError

    Example use case:
        data = <a dictionary containing numpy.bool_ and pandas dataframe>

        with open(folder/file_name, "w") as op:
            json.dump(data, op, cls=CustomJSONEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_json(orient="split" , indent=1)
        return json.JSONEncoder.default(self, obj)