import json
import os
from typing import List, Dict

def store_to_jsonl(filename: str, data_list: List[Dict]):
    """
    Append data to a JSONL file.
    Args:
        filename (str): Path to the JSONL file.
        data_list (List[Dict]): List of dictionaries to be written to the file.
    """
    if not data_list:
        return

    with open(filename, 'a', encoding='utf-8') as jsonl_file:
        for item in data_list:
            json.dump(item, jsonl_file, ensure_ascii=False)
            jsonl_file.write('\n')

def read_from_jsonl(filename: str, fields: List[str] = []) -> List[Dict]:
    """
    Read data from a JSONL file.

    Args:
        filename (str): Path to the JSONL file.
        fields (List[str], optional): List of fields to read. If empty, all fields are read.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a JSON object from the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If no data is found for the specified fields or the file is empty.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File \"{filename}\" not found. You are likely to read from a non-existent file.")
    data_list = []
    with open(filename, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if line:
                try:
                    json_object = json.loads(line)
                    if len(fields) == 0:
                        # Unspecified fields, read all fields
                        data_list.append(json_object)
                    else:
                        # Read only the specified fields
                        filtered_object = {field: json_object.get(field, "") for field in fields if field in json_object}
                        if filtered_object:
                            data_list.append(filtered_object)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    if len(data_list) == 0:
        raise ValueError(f"No data found for any specified field(s): \"{fields}\". Either the file \"{filename}\" is empty or none of the field(s) exist.")
    return data_list
