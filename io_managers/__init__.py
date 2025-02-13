from io_managers.csv_manager import read_from_csv, store_to_csv
from io_managers.jsonl_manager import read_from_jsonl, store_to_jsonl
from io_managers.xlsx_manager import read_from_excel, store_to_excel
from io_managers.raw_file_writer import write_to_file as store_to_raw
from typing import Callable
import os

def get_reader(file_path:str) -> tuple[Callable[[str, list], list], str]:
    """
    Get reader for structured data file. Supported file format: .csv/.xlsx/.jsonl
    
    :params str file_path: `path/to/the/file/to/read.ext`
    :raise: ValueError on unsupported format (How am I supposed to read that?)
    :return: a tuple with the first element being the reader, and the second element being the parsed extension in str.
    """
    file_readers = {
        ".csv": read_from_csv,
        ".xlsx": read_from_excel,
        ".jsonl": read_from_jsonl
    }
    
    ext = os.path.splitext(file_path)[1]
    reader = file_readers.get(ext)
    if reader == None:
        raise ValueError(f"I do not know how to read from a \"{file_path}\" file. Please use the following formats: csv, xlsx, jsonl.")
    return (reader, ext)
    
def get_writer(file_path: str) -> tuple[Callable[[str, list], None], str]:
    """
    Get writer based on file extension. Structured formats with dedicated writer: .csv/.xlsx/.jsonl
    
    If unknown extension is encountered, get a writer in plain text (preserving the specified extension).
    
    No extension restrictions.
    
    :params str file_path: `path/to/the/file/to/write/to.ext`
    :return: a tuple with the first item being writer and the second item extension name. On unknown extensions, second item is None. 
    """
    file_writers = {
        ".csv": store_to_csv,
        ".xlsx": store_to_excel,
        ".jsonl": store_to_jsonl
    }
    ext = os.path.splitext(file_path)[1]
    writer = file_writers.get(ext)
    if writer == None:
        return (store_to_raw, None)
    return (writer, ext)
