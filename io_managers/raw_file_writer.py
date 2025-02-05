def write_to_file(filename:str, data:str, mode="a"):
    with open(filename, encoding="utf-8", mode=mode) as file:
        file.write(data)