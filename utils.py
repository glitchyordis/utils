from pathlib import Path

def compare_img_name(path1: str, path2: str):
    """
    Navigates the subdirectories of path1 and path2 and compare filenames of files with extension
    jpg and png. 
    
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
    # Check if the directories exist
    if not Path(path1).is_dir():
        return f"Directory {path1} does not exist.", None
    if not Path(path2).is_dir():
        return None, f"Directory {path2} does not exist."

    # Get the set of all .jpg and .png files in path1
    # files_in_path1 = {f.name for f in Path(path1).glob('*.[jp][np]g')} # this searches through dir only
    files_in_path1 = {f.name for f in Path(path1).glob('**/*.[jp][np]g')} # this searches through subfolder too

    # Get the set of all .jpg and .png files in path2
    # files_in_path2 = {f.name for f in Path(path2).glob('*.[jp][np]g')} # this searches through dir only
    files_in_path2 = {f.name for f in Path(path2).glob('**/*.[jp][np]g')} # this searches through subfolder too

    # Files in path1 but not in path2
    in_path1_not_path2 = list(files_in_path1 - files_in_path2)

    # Files in path2 but not in path1
    in_path2_not_path1 = list(files_in_path2 - files_in_path1)

    in_both_paths = list(files_in_path1 & files_in_path2)

    return in_path1_not_path2, in_path2_not_path1, in_both_paths

