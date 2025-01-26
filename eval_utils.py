import re
import os 

def list_files_in_directory(directory, match_pattern=""):
    # Helper method. Takes a directory name and return a list of file names (with dir names)
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

def sanitize_pathname(pathname):
    return re.sub(r'[^\w\-_\.]', '_', f"{pathname}")