def write_to_file(filename:str, data:str, mode="a"):
    """
    :params filename: path to the file to be written
    :params data: a data string
    :params mode: write mode, same as in open() function. Default is "a" (append)
    :return: None
    """
    with open(filename, encoding="utf-8", mode=mode) as file:
        file.write(data)