"""
The module deals with various operations on file paths.

But it does not perform any I/O operation.
"""

from dataset_models import QuerySet
import re
import os 

def list_files_in_directory(directory, match_pattern=""):
    """
    Helper method. Takes a directory name and return a list of file names (with dir names)
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            matched = True
            # If provided match_pattern, will select only the qualified file names
            if match_pattern != "":
                matched = match_pattern in file

            if matched:
                file_paths.append(os.path.join(root, file))
    return file_paths

def craft_result_path(query_set: QuerySet, results_dir, dataset_name, model):
    """
    After conducting an evaluation, you get to store the result of each set somewhere else. This method crafts a resultant filename (**xlsx format**). It takes:
    - query_set: A QuerySet instance (for parsing the test file name as the eval subset name)
    - results_dir: The directory to store the records.
    - dataset_name: The records are marked as part of a dataset.
    - model: The evaluated model name
    """
    file_path = query_set.get_path()
    query_name = "nameless"
    # query_name can be None as a QuerySet instance might be instantiated with a literal query list. See class data_model.QuerySet.
    if file_path != None:
        query_name = parse_filename_from_path(query_set.get_path())
    
    return f'{strip_trailing_slashes_from_path(results_dir)}/{dataset_name}/{model}/{sanitize_pathname(f"test-{model}-{query_name}")}.xlsx'

def craft_eval_dir_path(results_dir, dataset_name, model):
    """
    Similar to make_result_path, but only reaches the destination directory. You get:
    `results_dir/dataset_name/model`
    """
    return f'{strip_trailing_slashes_from_path(results_dir)}/{dataset_name}/{model}'

def parse_filename_from_path(file_path):
    """
    path/to/your/file.ext => file
    """
    query_name, _ = os.path.splitext(os.path.basename(file_path))
    return query_name

def sanitize_pathname(pathname):
    """
    Remove potentially problematic characters for the file system.
    """
    return re.sub(r'[^\w\-_\.]', '_', f"{pathname}")

def strip_trailing_slashes_from_path(path_str: str):
    """
    Remove any trailing slash(es) for the convenience of path concatenation.
    
    Replace backslashes with slashes, so no need to handle escaping.
    """
    stripped = path_str.replace("\\", "/").rstrip("/")
    if stripped == "":
        raise FileNotFoundError(f"The destination dir \"{stripped}\" is likely not correct.")
    return stripped