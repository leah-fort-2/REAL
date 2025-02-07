"""
The module deals with various operations on file paths.

But it does not perform any I/O operation.
"""

from dataset_models import QuerySet
import re
import os 

def list_files_in_directory(directory, match_pattern=""):
    """
    :params str directory: the directory path to look up in
    :params str match_pattern: Optional. The specific filename pattern to look for.
    :return: A list of qualified file paths in the directory. e.g. `path/to/your/file.ext`
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
    Crafts a filename (**xlsx format**) for evaluation results of query_set. e.g. `results_dir/dataset_name/model/test-model-query_set_name.xlsx`
    
    :params QuerySet query_set: A QuerySet instance (for parsing the test file name as the eval subset name)
    :params str results_dir: The desetination directory for evaluation results.
    :params str dataset_name: Mark the results as part of a dataset.
    :params str model: The evaluated model name for identifying result files.
    :return: The target filename.
    """
    file_path = query_set.get_path()
    query_name = "nameless"
    # query_name can be None as a QuerySet instance might be instantiated with a literal query list. See class data_model.QuerySet.
    if file_path != None:
        query_name = parse_filename_from_path(query_set.get_path())
    
    return f'{strip_trailing_slashes_from_path(results_dir)}/{dataset_name}/{model}/{sanitize_pathname(f"test-{model}-{query_name}")}.xlsx'

def craft_eval_dir_path(results_dir, dataset_name, model):
    """
    Similar to make_result_path, but only reaches directory level. You get:
    `results_dir/dataset_name/model`
    
    :params str results_dir: The desetination directory for evaluation results.
    :params str dataset_name: Mark the results as part of a dataset
    :params str model: The evaluated model name for identifying result files
    :return: The target evaluation result directory name.
    """
    return f'{strip_trailing_slashes_from_path(results_dir)}/{dataset_name}/{model}'

def parse_filename_from_path(file_path):
    """
    e.g. `path/to/your/file.ext` => `file`
    """
    query_name, _ = os.path.splitext(os.path.basename(file_path))
    return query_name

def sanitize_pathname(pathname):
    """
    Replace unusable characters with underscores. Usable characters: [A-Za-z0-9], hyphen, underscore, period
    
    e.g. `@ gre^t filen@me` => `__gre_t_filen_me`
    """
    return re.sub(r'[^\w\-_\.]', '_', f"{pathname}")

def strip_trailing_slashes_from_path(path_str: str):
    """
    Remove any trailing slash(es) for the convenience of path concatenation. 
    
    Replace backslashes with slashes, so no need to handle escaping.
    
    e.g. `path\\to\\somewhere\\` => `path/to/somewhere`
    """
    stripped = path_str.replace("\\", "/").rstrip("/")
    if stripped == "":
        raise FileNotFoundError(f"The destination dir \"{stripped}\" is likely not correct.")
    return stripped