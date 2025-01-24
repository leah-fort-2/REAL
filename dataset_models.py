from csv_manager import store_to_csv, read_from_csv
from xlsx_manager import store_to_excel, read_from_excel
import copy

class QuerySet:
    def __init__(self, file_path_or_query_list, field_names=[]):
        """Leave field names empty if you wish to keep all fields
        
        Accept a data file path or a query string list.
        """
        if isinstance(file_path_or_query_list, str):
            # A path to the query file is provided
            if not file_path_or_query_list.endswith('.csv') and not file_path_or_query_list.endswith('.xlsx'):
                raise ValueError(f"Reading from unsupported file format: \"{file_path_or_query_list}\". Please use CSV or XLSX.")
            self.file_path = file_path_or_query_list
            if file_path_or_query_list.endswith('.csv'):
                self.queries = read_from_csv(file_path_or_query_list, field_names)
            elif file_path_or_query_list.endswith('.xlsx'):
                self.queries = read_from_excel(file_path_or_query_list, field_names)
        elif len(file_path_or_query_list)==0:
            # The provided query list is empty
            print(f"Empty set encountered on {file_path_or_query_list}.")
        elif isinstance(file_path_or_query_list[0], dict):
            # A query dictionary is provided as is
            self.file_path = None
            self.queries = file_path_or_query_list
        else:
            # A list of query strings is provided
            self.file_path = None
            self.queries = [{"query": query} for query in file_path_or_query_list]
    
    def __len__(self):
        return len(self.queries)
    
    def get_path(self):
        return self.file_path
    
    def get_queries(self):
        return copy.deepcopy(self.queries)
    
    def get_query_list(self, query_key="query"):
        return [query[query_key] for query in self.queries]
    
    def divide(self, division_size=10):
        """Split an entire query set into divisions with remainders e.g. [10, 10, 10, 5]. 
        
        Subsets inherit all fields and file path from the parent set. 
        
        Returns (less than or equal to) division-sized query set."""
        
        def spawn_subset(queries_chunk, path):
            subset = QuerySet(queries_chunk)
            subset.file_path = path
            return subset
        
        return [spawn_subset(self.get_queries()[i:i + division_size], self.get_path()) for i in range(0, len(self.queries), division_size)]
    
class ResponseSet:
    def __init__(self, response_list: list[dict]):
        self.responses = response_list
        
    def get_responses(self):
        return self.responses
    
    def store_to(self, file_path):
        if not file_path.endswith('.csv') and not file_path.endswith('.xlsx'):
            raise ValueError(f"Storing to unsupported file format: \"{file_path}\". Please use CSV or XLSX.")
        if file_path.endswith('.csv'):
            store_to_csv(file_path, self.responses)
        elif file_path.endswith('.xlsx'):
            store_to_excel(file_path, self.responses)