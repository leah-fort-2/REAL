import json
import os
from typing import List, Dict

def store_to_json(filename: str, data_list: List[Dict]):
    """
    Append data to a JSON file.
    
    :params str filename: Path to the JSON file.
    :params list[dict] data_list: List of dictionaries to be written to the file.
    :return: None
    """
    if not data_list:
        return
    
    if os.path.exists(filename):
        existing_data = read_from_json(filename)
        # Merge existing with newly added
        if isinstance(existing_data, list):
            data_list = existing_data + data_list
        elif isinstance(existing_data, dict):
            data_list = [existing_data] + data_list
        else:
            raise ValueError(f"Invalid data type in file: {filename}")
        
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False)
        
def read_from_json(filename: str, fields: List[str] = []) -> List[Dict]:
    """
    Read data from a JSON file.

    :params str filename: Path to the JSON file.
    :params list[str] fields: List of fields to read. If empty, all fields are read.
    :return List[Dict]: A list of dictionaries, where each dictionary represents a JSON object from the file.
    :raise ValueError: If no data is found for the specified fields or the file is empty.
    :raise FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File \"{filename}\" not found. You are likely to read from a non-existent file.")
    data_list = []
    with open(filename, 'r', encoding='utf-8') as json_file:
        try:
            data = json.load(json_file)
            if isinstance(data, list):
                if fields:
                    data_list = [{k: item[k] for k in fields if k in item} for item in data]
                else:
                    data_list = data
            elif isinstance(data, dict):
                if fields:
                    data_list = [{k: data[k] for k in fields if k in data}]
                else:
                    data_list = [data]

        except json.JSONDecodeError:
            # Handle invalid JSON
            raise ValueError(f"Invalid JSON format in file: {filename}")

    if len(data_list) == 0:
        raise ValueError(f"No data found for any specified field(s): \"{fields}\". Either the file \"{filename}\" is empty or none of the field(s) exist.")
    return data_list
