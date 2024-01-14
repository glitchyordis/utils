from typing import Optional, List, Literal, Tuple
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

def search_files(path: str, file_name: str, exts: Optional[List[str]],
                recursive: bool, file_name_match_type: Literal["any", "exact"]):
    """
    Searches through subdir. specified in `path` (if recursive set to True) for files whose name contain/ is exact match to `file_name`.
    
    Args:
        path: path to search for file
        file_name: file_name to filter (without extension)
        exts: extensions of file interested in. if None, assumes an ext already exist in filename and use it.
        recursive: if True, searches subdirectories too
        file_name_match_typ:
            "any": matches with any files that contains the strin in `file_name`
            "except": only matches files that has name == `file_name`

    Returns:
        list of filepaths for file whose name contains `file_name`

    # Example usage
    files = util.search_files(<directory to search for file>, 
                            '20231208_121138', exts=["txt", "png"],
                            recursive=False, file_name_match_type="exact")
    for file in files:
        print(file)
    """ 
    EXACT = "exact"
    ANY = "any"

    def search(recursive: bool, files: list, path: pathlib.Path, pattern: str):
        if recursive:
            files.extend(list(path.glob(f"**/{pattern}")))
        else:
            files.extend(list(path.glob(pattern)))

    assert file_name_match_type in [EXACT, ANY], "Invalid exts used."
    message = (f"{exts = }\n"
               f"Confirm using {recursive = }\n?"
               f"{file_name_match_type = }")
    if exts is None:
        if file_name_match_type != EXACT:
            file_name_match_type = EXACT
            message += (f"\n\nfile_name_match_type changed to \"{file_name_match_type}\" due to {exts = }")

    proceed = strtobool(gui.popup(message, title = f"{__name__}", button_type=1,
                                  keep_on_top=True))

    # Create a Path object
    path = pathlib.Path(path)

    # Check if the path exists
    if not path.exists():
        print(f"The path {path} does not exist.")
        return []

    # Prepare the file name pattern
    files = []
    if proceed:
        if isinstance(exts, list):
            for ext in exts:
                if file_name_match_type == ANY:
                    pattern = f"*{file_name}*.{ext}"
                elif file_name_match_type == EXACT:
                    pattern = f"{file_name}.{ext}"
                search(recursive, files, path, pattern)
        elif exts is None:
            pattern = f"{file_name}"
            search(recursive, files, path, pattern)
        else:
            raise UserWarning(f"Invalid {exts = }")

        # Return the list of file paths
        return [str(file.resolve()) for file in files]
    else:
        raise UserWarning(f"User selected {proceed = }")

def compare_file_name(path1: str, path2: str, ext: List[str], recursive: Tuple[bool, bool]):
    """
    Navigates the subdirectories of path1 and path2 and compare filenames of files with extension. 
    
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
    assert isinstance(recursive, tuple) and len(recursive)==2 and all([isinstance(x, bool) for x in recursive]), f"Invalid recursive setting. {recursive = }"
    msg = (f"Confirm using\n"
           f"{ext = }\n"
           f"{recursive = }")
    proceed = strtobool(gui.popup(msg, title = f"{__name__}", button_type=1,
                                  keep_on_top=True))

    # Check if the directories exist
    if not pathlib.Path(path1).is_dir():
        return f"Directory {path1} does not exist.", None, None
    if not pathlib.Path(path2).is_dir():
        return None, f"Directory {path2} does not exist.", None
    
    if proceed:
        files_in_path1, files_in_path2 = set(), set()

        for extension in ext:
            # Get the set of all .jpg and .png files in path1
            # files_in_path1 = {f.name for f in Path(path1).glob('*.[jp][np]g')} # this searches through dir only
            # files_in_path1 = {f.name for f in pathlib.Path(path1).glob(f'**/*.{ext}')} # this searches through subfolder too
            if recursive[0]:
                files_in_path1.update({f.name for f in pathlib.Path(path1).rglob(f'*.{extension}')})
            else:
                files_in_path1.update({f.name for f in pathlib.Path(path1).glob(f'*.{extension}')})
        
            # Get the set of all .jpg and .png files in path2
            # files_in_path2 = {f.name for f in Path(path2).glob('*.[jp][np]g')} # this searches through dir only
            if recursive[1]:
                files_in_path2.update({f.name for f in pathlib.Path(path2).rglob(f'*.{extension}')}) # this searches through subfolder too
            else:
                files_in_path2.update({f.name for f in pathlib.Path(path2).glob(f'*.{extension}')})

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