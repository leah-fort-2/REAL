from csv_manager import store_to_csv, read_from_csv
from xlsx_manager import store_to_excel, read_from_excel
from jsonl_manager import store_to_jsonl, read_from_jsonl
import copy

class QuerySet:
    def __init__(self, file_path_or_query_list, field_names=[]):
        """Leave field names empty if you wish to keep all fields
        
        Accept a data file path or a query string list.
        """
        if isinstance(file_path_or_query_list, str):
            # A path to the query file is provided
            if not file_path_or_query_list.endswith('.csv') and not file_path_or_query_list.endswith('.xlsx') and not file_path_or_query_list.endswith('.jsonl'):
                raise ValueError(f"Reading from unsupported file format: \"{file_path_or_query_list}\". Please use the following: csv, xlsx, jsonl.")
            self.file_path = file_path_or_query_list
            if file_path_or_query_list.endswith('.csv'):
                self.queries = read_from_csv(file_path_or_query_list, field_names)
            elif file_path_or_query_list.endswith('.xlsx'):
                self.queries = read_from_excel(file_path_or_query_list, field_names)
            elif file_path_or_query_list.endswith('.jsonl'):
                self.queries = read_from_jsonl(file_path_or_query_list, field_names)
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
    
    def merge_keys(self, key_list_to_merge, merged_key_name, with_key_names=True):
        """
        Create a new QuerySet where specified keys are merged into a new key. The merged fields are concatenated with line breakers (\\n). The original QuerySet remains unchanged.
        
        A typical use case is to merge question and options into a single field for MCQ-type dataset evaluation.  
        """
        updated_query = self.get_queries()
        
        # The query has been validated as non-empty. Safely retrieves the first element.
        # If specified key(s) does not exist in query, raise an exception.
        for key in key_list_to_merge:
            if key not in updated_query[0]:
                key_list_stringified = str(updated_query[0].keys())
                raise KeyError(f"The specified key \"{key}\" not found in the query set.  Available keys: {key_list_stringified}")
                
        for query in updated_query:
            merged_value = "\n".join([
                f"{key}: {query[key]}" if with_key_names else f"{query[key]}"
                for key in key_list_to_merge])
            query[merged_key_name] = merged_value
        updated_query_set = QuerySet(updated_query)
        updated_query_set.file_path = self.file_path
        return updated_query_set
    
class ResponseSet:
    def __init__(self, response_list: list[dict], response_key=None):
        """
        You should not instantiate this class directly unless you know what you are doing.
        Optional: Mark response_key field names
        You can safely ignore this if score judging isn't needed. 
        """
        self.responses = response_list
        self.response_key=response_key
        
    def get_responses(self):
        return self.responses
    
    def __len__(self):
        return len(self.responses)
    
    def judge(self, answer_key="answer", eval_name="Evaluation", response_preprocessor=None, answer_preprocessor=None):
        """
        Conduct score judging using the specified answer field with optional preprocessing.
        - response_preprocessor: function<str> -> str
        - answer_preprocessor: function<str> -> str
        
        Meanwhile, add a {eval_name}_score field.
        
        Return a scoring result with the following entries:
        
        - eval_name: the evaluation name specified -> str
        - score: the number of correct answers -> int
        - full_score: the total number of questions -> int
        - accuracy: the accuracy rate -> float
        
        """
        if self.response_key == None:
            print(f"Error: The evaluation {eval_name} does not have response_key specified. Unable to proceed with score judging.")
            return None
        
        if len(self.responses) == 0:
            print(f"Warning: The evaluation {eval_name} has an empty response set.")
            return None
        
        if answer_key not in list(self.responses[0].keys()):
            print(f"Error: Evaluation {eval_name}'s answer_key {answer_key} does not seem to be an existing field.")
            
        if self.response_key not in list(self.responses[0].keys()):
            print(f"Error: Evaluation {eval_name}'s response_key {self.response_key} does not seem to be an existing field.")
        
        if response_preprocessor == None:
            response_preprocessor = lambda r: r
            
        if answer_preprocessor == None:
            answer_preprocessor = lambda r: r
        
        score = 0
        full_score = len(self)
        
        for resp_obj in self.responses:
            response = f"{resp_obj[self.response_key]}".strip()
            correct_answer = f"{resp_obj[answer_key]}".strip()
            try:
                preprocessed_response = response_preprocessor(response)
                preprocessed_answer = answer_preprocessor(correct_answer)
            except Exception:
                print(f"An error occurred in preprocessing stage: {Exception}")
                return None
            if preprocessed_answer == preprocessed_response:
                score += 1
                resp_obj.update({f"{eval_name}_score": 1})
            else:
                resp_obj.update({f"{eval_name}_score": 0})
                
        print(
            f"======\nEvaluation Report:\nEvaluation Name: {eval_name}\nAccuracy: {score}/{full_score} ({round(100*score/full_score, 1)}%)\n======\n")

        return {"eval_name": eval_name, "score": score, "full_score": full_score, "accuracy": score/full_score}

    
    def store_to(self, file_path):
        if not file_path.endswith('.csv') and not file_path.endswith('.xlsx'):
            raise ValueError(f"Storing to unsupported file format: \"{file_path}\". Please use CSV or XLSX.")
        if file_path.endswith('.csv'):
            store_to_csv(file_path, self.responses)
        elif file_path.endswith('.xlsx'):
            store_to_excel(file_path, self.responses)
        elif file_path.endswith('.jsonl'):
            store_to_jsonl(file_path, self.responses)