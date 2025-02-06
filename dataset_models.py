from io_managers.csv_manager import store_to_csv, read_from_csv
from io_managers.xlsx_manager import store_to_excel, read_from_excel
from io_managers.jsonl_manager import store_to_jsonl, read_from_jsonl
from text_preprocessors import as_is
from judgers.presets import STRICT, JUDGE_FAILED_MSG
import copy
import os
import time


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
    def __init__(self, response_list: list[dict], query_key=None, response_key=None):
        """
        You should not instantiate this class directly unless you know what you are doing.
        Optional: Mark query_key field name. Safely ignore this if score judging isn't needed.
        Optional: Mark response_key field names. Safely ignore this if score judging isn't needed. 
        """
        self.responses = response_list
        self.response_key=response_key
        self.query_key=query_key
        
    def get_responses(self):
        return self.responses
    
    def __len__(self):
        return len(self.responses)
    
    def judge(self, answer_key="answer", context_key = None, eval_name="Evaluation", response_preprocessor=as_is, answer_preprocessor=as_is, judger=STRICT):
        """
        Conduct a [0,1] acc score judging using specified answer field with optional context, preprocessing and judger method. On empty preprocessed answer/response, exclude the question from result score and total score. Return a literal score dictionary.
        - response_preprocessor: function<str> -> str
          - Preprocess the response before submitting to the judger method. e.g. mcq_preprocessor, etc. see text_preprocessors module.
        - answer_preprocessor: function<str> -> str
          - Similar to response
        - judger: function<str, str, context=None> -> float
          - Takes two strings and an optional context string (for model scoring). Output a [0,1] accuracy score. Default to STRICT.
          - If left as None, context_key will fall back to query_key. If query_key is not specified, context_key will be ignored.
        
        Score dictionary structure
        
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
            return None
            
        if self.response_key not in list(self.responses[0].keys()):
            print(f"Error: Evaluation {eval_name}'s response_key {self.response_key} does not seem to be an existing field.")
            return None
            
        # If left as None, context_key will fall back to query_key. If query_key is not specified, context_key will be ignored.
        if context_key != None:
            if context_key not in list(self.responses[0].keys()):
                print(f"Error: Evaluation {eval_name}'s optional context_key {context_key} does not seem to be an existing field.")
                return None
        elif self.query_key != None:
                context_key = self.query_key
        
        score = 0
        full_score = len(self)
        
        for resp_obj in self.responses:
            response = f"{resp_obj[self.response_key]}".strip()
            correct_answer = f"{resp_obj[answer_key]}".strip()
            # Add None check, because query_key is for providing context to model_scoring. You can still do STRICT judging without query text.
            context = ""  if context_key == None else f"{resp_obj[context_key]}".strip()
            
            # For each question, do preprocessings first.
            try:
                preprocessed_response = response_preprocessor(response)
                preprocessed_answer = answer_preprocessor(correct_answer)
            except Exception:
                # Preprocessing failed, skip the question.
                print(f"An error occurred in preprocessing stage: {str(Exception)[:50]}... Skip the question.")
                full_score -= 1
                continue
            
            # Skip questions with empty answer/response.
            if preprocessed_answer == "":
                # No valid answer field. Skip the question.
                print(f"Parsed invalid answer field. Skippped. Response: {resp_obj[self.response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
                full_score -= 1
                continue
            
            if preprocessed_response == "":
                # No valid response field to judge. Skip the question.
                print(f"Parsed invalid response field. Skippped. Response: {resp_obj[self.response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
                full_score -= 1
                continue
            
            # Score judging algorithm.
            response_score: float = judger(preprocessed_answer, preprocessed_response, context = context)
            
            if response_score == JUDGE_FAILED_MSG:
                # Score judging failed. Skip the question. Most likely stemming from model scoring.
                print(f"Score judging failed. Skipped. Response: {resp_obj[self.response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
                full_score -=1
                continue
            
            score += response_score
            resp_obj.update({f"{eval_name}_score": response_score})
                
        print(
            f"======\nEvaluation Report:\nEvaluation Name: {eval_name}\nAccuracy: {score}/{full_score} ({round(100*score/full_score, 1)}%)\n======\n")

        return {"eval_name": eval_name, "score": score, "full_score": full_score, "accuracy": score/full_score}

    
    def store_to(self, file_path):
        if not file_path.endswith('.csv') and not file_path.endswith('.xlsx'):
            raise ValueError(f"Storing to unsupported file format: \"{file_path}\". Please use CSV or XLSX.")
        
        dirname = os.path.dirname(file_path)
        if dirname != "" and not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        # Handle concurrent file writing between jobs. Default: 2 retries, 5 sec interval
        
        max_retries = 2 # set max retry count
        interval = 5 # in seconds
        retry = 0
        for _ in range(max_retries + 1): # range has exclusive upper bound
            try:
                if file_path.endswith('.csv'):
                    store_to_csv(file_path, self.responses)
                elif file_path.endswith('.xlsx'):
                    store_to_excel(file_path, self.responses)
                elif file_path.endswith('.jsonl'):
                    store_to_jsonl(file_path, self.responses)
                break
            except IOError:
                if retry < max_retries:
                    retry += 1
                    print(f"Failed to store response results to {file_path}. Retry {retry}/{max_retries} in {interval} second(s)...")
                    time.sleep(interval)
                else:
                    raise IOError(f"Failed to store response results to 
                                  {file_path} after {max_retries} retries.")
            