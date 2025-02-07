import csv
import os

def store_to_csv(filename: str, data_list: list[dict]):
    """
    :params filename: path to the csv file
    :params list[dict] data_list: list of dictionaries to be written to the csv file
    :return: None
    """
    if not data_list:
        return
    
    fieldnames = data_list[0].keys()
    
    header_flag = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if header_flag:
            writer.writeheader()
        for row in data_list:
            writer.writerow(row)

def read_from_csv(filename, fields=[]):
    """
    :params filename: path to the csv file
    :params list[str] fields: list of fields to read. If empty, all fields are read
    :return list[dict]:
    :raise ValueError: if no record is read from the file
    :raise FileNotFoundError: if the file is not found

    """
    data_list = []
    try:
        with open(filename, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if len(fields) == 0:
                # Unspecified fields, read all fields
                for row in reader:
                    data_list.append(dict(row))  # Convert OrderedDict to regular dict
            else:
                # Read only the specified fields
                header_fields = reader.fieldnames
                for row in reader:
                    filtered_row = {field: row.get(field, "") for field in fields if field in header_fields}
                    data_list.append(filtered_row)
        
        if len(data_list) == 0:
            raise ValueError(f"No available field is found in \"{filename}\". It's likely empty or not containing the fields specified.")
        
        return data_list
    except FileNotFoundError:
        raise FileNotFoundError(f"File \"{filename}\" not found. You are likely to read from a non-existent file.")
