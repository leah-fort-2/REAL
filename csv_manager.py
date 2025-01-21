import csv

def store_to_csv(data_list, filename):
    if not data_list:
        return
    
    fieldnames = data_list[0].keys()
    
    with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)

def read_from_csv(filename, query_key="query"):
    query_list = []
    try:
        with open(filename, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if query_key in row:
                    query_list.append(row[query_key])
        if len(query_list)==0:
            raise ValueError(f"Key \"{query_key}\" is not found in the csv file.")
        return query_list
    except FileNotFoundError:
        raise FileNotFoundError(f"File \"{filename}\" not found. You are likely to read from a non-existent file.")
