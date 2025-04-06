from io_managers import get_reader, get_writer
from text_preprocessors import as_is
from judgers.presets import STRICT_MATCH, JUDGE_FAILED_MSG
from request_manager.request_manager import FALLBACK_ERR_MSG
from random import shuffle
from typing import Any, Callable, Coroutine
from collections import defaultdict
import logging
import dotenv
import asyncio
import copy
import os
import time

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SCORING_BATCH_SIZE = int(os.getenv("SCORING_BATCH_SIZE", "5"))

class QuerySet:

    def _filter_fields(self, query_list: list[dict], field_names: list[str]) -> list[dict]:
        """
        Filter the fields in each dictionary in the query list to include only the specified fields. On empty field_names, return [].
        
        :params list[dict] query_list: The list of dictionaries to filter.
        :params list[str] field_names: The list of field names to retain.
        :return: A new list of dictionaries containing only the specified fields.
        """
        return [{field: query[field] for field in field_names if field in query} for query in query_list]
    
    def _append(self, query: dict[str, Any]) -> None:
        self.queries.append(query)

    def __init__(self, file_path_or_query_list: str | list[dict] | list[str], field_names=[]):
        """
        Create a query set. After instantiation, the query set is read only.
        
        :params str | list[dict] | list[str] file_path_or_query_list: a data file path,  a query object list, or a query string list.
        :params field_names:  a list of field names to read from the file. If empty, all fields are read.
        
        """
        if isinstance(file_path_or_query_list, str):
            # A path to the query file is provided
            self.queries = self._read_queries_from_file(file_path_or_query_list, field_names)
            self.file_path = file_path_or_query_list
        elif len(file_path_or_query_list) == 0:
            # The provided query list is empty
            logger.warning(f"Empty set encountered on {file_path_or_query_list}.")
            self.queries = []
            self.file_path = None
        elif isinstance(file_path_or_query_list[0], dict):
            # A query dictionary is provided as is
            if field_names:
                self.queries = self._filter_fields(file_path_or_query_list, field_names)
            else:
                self.queries = file_path_or_query_list
            self.file_path = None
        else:
            # A list of query strings is provided
            self.file_path = None
            self.queries = [{"query": query} for query in file_path_or_query_list]

    def _read_queries_from_file(self, file_path_or_query_list: str, field_names: list[str]):
        reader: Callable[[str, list], list] = None
        try:
            reader, ext = get_reader(file_path_or_query_list)
        except ValueError:
            raise ValueError(f"Reading from unsupported file format: \"{file_path_or_query_list}\".")
        
        # None safety has been ensured
        return reader(file_path_or_query_list, field_names)
        
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            # Handle single query item access
            return self.queries[key]
        elif isinstance(key, slice):
            # Handle slicing
            subset = QuerySet(self.queries[key])
            subset.file_path = self.file_path
            return subset
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Only int and slice are supported.")
    
    def get_path(self):
        """
        :return: path to the query set file. If instantiated from a list, return None.
        """
        return self.file_path
    
    def get_queries(self):
        """
        :return: a deepcopy of the query object list. Modifying this list does not affect the original query set.

        """
        return copy.deepcopy(self.queries)
    
    def get_query_list(self, query_key="query"):
        """
        :return:  a list of query strings. Modifying this list does not affect the original query set.

        """
        return [query[query_key] for query in self.get_queries()]
    
    def divide(self, division_size=10) -> list['QuerySet']:
        """
        Split an entire query set into divisions with remainders e.g. `[10, 10, 10, 5]`. Subsets inherit all fields and file path from the parent set. 
        
        :params division_size: Maximum size of each subset
        :return: a list of (less than or equal to) division-sized query subset.
        """
        return [self[i:i + division_size] for i in range(0, len(self), division_size)]
    
    def divide_by_key(self, division_key: str, ignore_unsorted=False) -> dict[str, 'QuerySet']:
        """
        Split an entire query set by a given key e.g. `category: math`. Subsets inherit all fields and file path from the parent set.
        
        :params str division_key: based on this key e.g. `category` a division will be conducted
        :params bool ignore_unsorted: in case the division_key is not found in any query object. True: keep it in "unsorted" key and continue; False: halt the division.
        :return: a dict of query sets each with a distinct identifier of the given key
        """
        return dict(self.divide_by_keys([division_key], completeness=ignore_unsorted))
        
    def divide_by_keys(self, division_keys: list[str], completeness=True) -> dict[str, 'QuerySet']:
        """
        Split an entire query set by specific keys e.g. `category: math; discipline: algebra` => `{"math/algebra": QuerySet}`. Subsets inherit all fields and file path from the parent set.
        
        :params list[str] division_keys: based on these keys e.g. `category, discipline` a division will be conducted SEQUENTIALLY
        :params bool completeness: in case any query object isn't complete with all division keys. e.g. with `division_keys=["category", "discipline"]` and `query={"category": "math"}`, True: Add the orphan query to the "unsorted" leaf of the "category" node. False: halt the division.
        :return: a flattened tree-shaped dict of distinct keys with query set objects on leaf nodes e.g. `{"math/algebra/abstract": QuerySet}`
        """
        def create_default_query_set():
            default_query_set = QuerySet([])
            default_query_set.file_path=self.get_path()
            return default_query_set
        
        all_queries_categorized = defaultdict(create_default_query_set)
        for query in self.queries:
            identifiers = []
            for i, division_key in enumerate(division_keys):
                division_value = query.get(division_key, "unsorted")
                identifiers.append(division_value)
                key_not_found = division_value == "unsorted"
                if key_not_found and completeness == False:
                    logger.error(f"Completeness is selected and specified key {division_key} is not found in the following query object. Division halted. \n{query}")
                    return
                if key_not_found or i == len(division_keys) - 1:
                    all_queries_categorized["/".join(identifiers)]._append(query)
        
        return all_queries_categorized
    
    def merge_keys(self, key_list_to_merge, merged_key_name, with_key_names=True):
        """
        Create a new QuerySet where specified keys are merged into a new key. The merged fields are concatenated with line breakers (\\n). The original QuerySet remains unchanged.
        
        A typical use case is to merge question and options into a single field for MCQ-type dataset evaluation.  
        
        :params key_list_to_merge: a list of keys to merge into the new key. The order of keys in the list is preserved in the merged field.
        :params merged_key_name: the name of the new key. If the key already exists, it will be overwritten.
        :params with_key_names: whether to include key names in the merged field. e.g. `Field1: Value1\\nField2: Value2` vs `Value1\\nValue2`. Default to True.

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
    
    def flatten_field(self, field_to_flatten, new_field_names=[]):
        """
        Suppose the field value is a list or dict. Flatten the field by assigning new/existing field names to each element in the list/dict. The original field is removed.
        
        With new_field_names=["Option_1",  "Option_2", "Option_3", "Option_4"]:
        > `{"target": ["Good", "Bad", "Bad", "Bad"]}` =>  `{"Option_1": "Good", "Option_2": "Bad",  "Option_3": "Bad", "Option_4": "Bad"}`
        
        On dict field value, new_field_names are ignored. The dict keys are used as new field names.

        > Example with dict: `{"target": {"Option_1": "Good", "Option_2": "Bad",  "Option_3": "Bad", "Option_4": "Bad"}}` => `{"Option_1": "Good", "Option_2": "Bad",  "Option_3": "Bad", "Option_4": "Bad"}`
        
        :params str field_to_flatten: Name of the field to flatten.
        :params list new_field_names: New field names for elements of list value. If the field is dict, will ignore this argument and use key names as new field names.
        :return: A new QuerySet with only flattened field updated. The original QuerySet remains unchanged.

        """
        flattened_queries = []
        for query in self.queries:
            if field_to_flatten not in query:
                logger.error(f"Field '{field_to_flatten}' not found in query: {query[:50]}")
                continue

            field_value = query[field_to_flatten]
            
            if isinstance(field_value, list):
                if not new_field_names:
                    raise ValueError("List field flattening requires specifying new field names: What fields shall the list elements be placed in after the flattening?")
                if len(field_value) > len(new_field_names):
                    logger.warning(f"Number of elements in the list ({len(field_value)}) exceeds the number of new field names ({len(new_field_names)}). Can't flatten. Field value: {field_value[:50]}.")
                if len(field_value) != len(new_field_names):
                    logger.warning(f"Number of new field names ({len(new_field_names)}) does not match the number of elements in the list ({len(field_value)}). Field value: {field_value[:50]}.")
                
                # Update the query with the new field names and list element as values.
                flattened_query = {k: v for k, v in query.items() if k != field_to_flatten}
                # Will override existing fields with the same names.
                # Zip: Match new field names as possible. Unmatched new field names are dropped.
                # ["A", "B", "C", "D"] + ["Good", "Bad", "Bad"]=> {"A": "Good", "B": "Bad", "C": "Bad"]
                flattened_query.update(dict(zip(new_field_names, field_value)))
            elif isinstance(field_value, dict):
                flattened_query = {k: v for k, v in query.items() if k != field_to_flatten}
                # Will override existing fields with the same names.
                overridden_keys = set(flattened_query.keys()) & set(field_value.keys())
                if len(overridden_keys) != 0:
                    logger.warning(f"Overriding keys {overridden_keys} in the flattened query. Field value: {field_value[:50]}. Existing keys: {flattened_query.keys()}")
                flattened_query.update(field_value)
            else:
                raise TypeError(f"You can't flatten '{field_to_flatten}' ({type(field_value)}). It must be a list or dict.")
            
            flattened_queries.append(flattened_query)
        
        flattened_set = QuerySet(flattened_queries)
        flattened_set.file_path = self.file_path
        return flattened_set
    
    
    def mcq_shuffle(self, answer_key, target_answer_key, keys_to_shuffle=["A", "B", "C", "D"], target_option_keys=["A", "B", "C", "D"]):
        """
        Return a new QuerySet with specified keys shuffled. All existing keys not updated are retained. Only applies on mcq with key name as answer e.g. {"answer": "C"} but not {"answer": "answer value"}
        
        :params answer_key: The answer key before shuffling
        :params target_answer_key: Where to store the new answer after shuffling
        :params keys_to_shuffle: The mcq option keys to shuffle. Default to ABCD but order does not matter. Must match target_option_keys in number. e.g. 4 == 4
        :params target_option_keys: WHere to store the shuffled options. Default to ABCD. The order is preserved in the new query_set. Must match target_option_keys in number. e.g. 4 == 4
        :return: A `QuerySet` object with shuffled keys.
        """
        
        existing_queries = self.get_queries()
        shuffled_queries = []
        if len(keys_to_shuffle) != len(target_option_keys):
            raise ValueError(f"The key lists before and after shuffling do not match in number. I can't shuffle {len(keys_to_shuffle)} options into {len(target_option_keys)}.")
            
        for query_obj in existing_queries:
            answer=""
            try:
                answer=query_obj[answer_key]
            except KeyError:
                raise KeyError(f"Specified answer key not found. Query: {str(query_obj)[:50]}...; Available keys: {", ".join(query_obj.keys())}")
            
            shuffle(keys_to_shuffle)
            new_query_obj = {}
            new_answer = ""
            # [A, B, C, D], [B, C, D, A] => {A: ..., B: ..., ...} 
            for target_option_key, original_option_key in zip(target_option_keys, keys_to_shuffle):
                # {A: some option}
                new_query_obj[target_option_key]=query_obj[original_option_key]
                # original option key is the answer 
                if original_option_key == answer:
                    # {answer: some option}
                    new_answer = target_option_key
            
            # To keep the order of target keys, update the answer key at last.
            new_query_obj[target_answer_key]=new_answer
            # The new query key has been populated
            shuffled_queries.append(new_query_obj)
            
        # As per now, we have a new list of query objects.
        # We will finish by updating the shuffled queries to existing queries, to keep other existing fields.
        [existing_query.update(shuffled_query) for existing_query, shuffled_query in zip(existing_queries, shuffled_queries)]
        
        shuffled_set = QuerySet(existing_queries)
        shuffled_set.file_path = self.get_path()
        return shuffled_set
    
class ResponseSet:
    def __init__(self, response_list: list[dict], query_key=None, response_key=None):
        """
        You should not instantiate this class directly unless you know what you are doing.
        
        :params query_key: Optional. Only for score judging.
        :params response_key: Optional. Only for score judging. 
        """
        self.responses = response_list
        self.response_key=response_key
        self.query_key=query_key
        
    def get_responses(self):
        """
        Returns the very list[dict] containing response objects.
        
        :return: the response list.

        """
        return self.responses
    
    def __len__(self):
        return len(self.responses)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            # Handle single query item access
            return self.responses[key]
        elif isinstance(key, slice):
            # Handle slicing
            subset = ResponseSet(self.responses[key], query_key=self.query_key, response_key=self.response_key)
            return subset
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Only int and slice are supported.")
        
    def get_response_key(self):
        return self.response_key
    
    def get_query_key(self):
        return self.query_key
    
    async def judge(self, answer_key="answer", context_key=None, eval_name="Evaluation", response_preprocessor=as_is, answer_preprocessor=as_is, judger=STRICT_MATCH, foreign_response_key=None):
        """
        Submit a [0,1] acc score judging task using specified answer field with optional context, preprocessing and judger method. Failed judgings are ignored. Return a scoring dictionary. On invalid judge parameters, returns None.
        
        - :side effects: Modifies self.responses by adding a new field "score"
        
        :params answer_key: Specify the field name for answer. Default to "answer"
        :params str context_key: Optional. Specify the field name for context. Default to None. If left as None, context_key will fall back to query_key. If query_key is not specified, context_key will be ignored.
        :params eval_name: Only for generating report on the terminal. e.g. "accountant_val" "agronomy". Default to "Evaluation"
        :params Callable[str -> str] response_preprocessor:

          - Preprocess the response before submitting to the judger method. e.g. `mcq_preprocessor`, etc. Default to `as_is`. See text_preprocessors module. 
          
        :params Callable[str -> str] answer_preprocessor:
          - Same as response, but for answer.
        :params Callable[[str, str, str], Coroutine[Any, Any, float | str]] judger:
        
          - Takes two strings and an optional context string (for model scoring). Output a [0,1] accuracy score (model scoring may return a JUDGE_FAILED_MSG: str). Default to STRICT_MATCH.
          
        :params str foreign_response_key: Default to None. Retrieve specified key value instead of the response_key set on instantiation. Default to None.

        :return dict<str, Any>: A score dictionary with following fields:

        - :eval_name: the evaluation name specified -> str
        - :score: the number of correct answers -> int
        - :full_score: the total score with failed judging excluded -> int
        - :accuracy: final score / full score -> float
        
        """
        if len(self.responses) == 0:
            logger.warning(f"The evaluation {eval_name} has an empty response set.")
            return None
        
        if not self._validate_key_names(eval_name, answer_key, context_key, foreign_response_key):
            return None
        
        # If foreign_response_key is specified, judge that instead
        response_key = foreign_response_key if foreign_response_key else self.response_key
        
        # If left as None, context_key will fall back to query_key. If query_key is not specified, context_key will be ignored.
        if context_key is None and self.query_key is not None:
            context_key = self.query_key
        
        score = 0
        full_score = len(self)
        
        semaphore = None if SCORING_BATCH_SIZE == 0 else asyncio.Semaphore(SCORING_BATCH_SIZE)
        
        for resp_obj in self.responses:
            # Receives a score delta tuple.
            score_change, full_score_change = await self._judge_single_resp_obj(resp_obj, response_key, answer_key, context_key, response_preprocessor, answer_preprocessor, judger, semaphore)
            score += score_change
            full_score += full_score_change
                
        logger.info(
            f"\n======\nEvaluation Report:\nEvaluation Name: {eval_name}\nAccuracy: {score}/{full_score} ({round(100*score/full_score, 1)}%)\n======\n")

        return {"eval_name": eval_name, "score": score, "full_score": full_score, "accuracy": score/full_score}

    def _validate_key_names(self, eval_name, answer_key, context_key, foreign_response_key):
        
        if foreign_response_key and not isinstance(foreign_response_key, str):
            logger.error(f"foreign_response_key must be a string. Got {type(foreign_response_key)} instead. ")
            return False

        response_key = foreign_response_key if foreign_response_key else self.response_key
        
        if response_key is None:
            logger.error(f"The evaluation {eval_name} does not have response_key specified. Unable to proceed with score judging.")
            return False
            
        if response_key not in list(self.responses[0].keys()):
            logger.error(f"Evaluation {eval_name}'s response_key {response_key} does not seem to be an existing field.")
            return False
                
        if answer_key not in list(self.responses[0].keys()):
            logger.error(f"Evaluation {eval_name}'s answer_key {answer_key} does not seem to be an existing field.")
            return False
        
        # If context_key is specified, need to check whether it exists in the response set.
        # If not, validation fails
        if context_key is not None:
            if context_key not in list(self.responses[0].keys()):
                logger.error(f"Evaluation {eval_name}'s optional context_key {context_key} does not seem to be an existing field.")
                return False
        
        return True
    
    async def _judge_single_resp_obj(self, resp_obj, response_key, answer_key, context_key, response_preprocessor: Callable[[str], str], answer_preprocessor: Callable[[str], str], judger: Callable[[str, str, str], Coroutine[Any, Any, float | str]], semaphore=None):
        response = resp_obj[response_key]
        correct_answer = resp_obj[answer_key]
        # context_key has been validated to be either 1) an existing key in response 2) fallback to query_key or 3) None. Ensure None safety before retrieval
        context = resp_obj[context_key] if context_key else ""
                
        # Detect cases where questions should be skipped
        def _is_skipped_pre_judging():
            # Detect failed request fallback message and skip the question
            if response == FALLBACK_ERR_MSG:
                return True
            
            # For each question, do preprocessings first.
            try:
                preprocessed_response = response_preprocessor(response)
                preprocessed_answer = answer_preprocessor(correct_answer)
            except Exception:
                # Preprocessing failed, skip the question.
                logger.error(f"An error occurred in preprocessing stage: {str(Exception)[:50]}... Skip the question.")
                return True

            # Skip questions with empty answer/response.
            if preprocessed_answer == "":
                # No valid answer field. Skip the question.
                logger.error(f"Parsed invalid answer field. Skipped. Response: {resp_obj[response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
                return True
            
            if preprocessed_response == "":
                # No valid response field to judge. Skip the question.
                logger.error(f"Parsed invalid response field. Skipped. Response: {resp_obj[response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
                return True
        
        # Tuple[score, full_score] for score changes. (0, -1) for skipped; (score, 0) for not skipped
        SKIPPED = (0, -1)
        def JUDGED(result_score):
            return (result_score,  0)
        
        if  _is_skipped_pre_judging():
            # Tuple[score, full_score] for score changes
            return SKIPPED

        # Preprocessors have been validated.
        preprocessed_response = response_preprocessor(response)
        preprocessed_answer = answer_preprocessor(correct_answer)
        # Score judging algorithm.
        if semaphore:
            async with semaphore:
                score = await judger(preprocessed_answer, preprocessed_response, context=context)
        else:
            score = await judger(preprocessed_answer, preprocessed_response, context=context)

        if score == JUDGE_FAILED_MSG:
            # Score judging failed. Most likely stemming from model scoring.
            logger.error(f"Score judging failed. Skipped. Response: {resp_obj[response_key][:50]}... ; Answer: {resp_obj[answer_key][:50]}...")
            # Skip the question in scoring
            return SKIPPED
        
        resp_obj.update({
            "judged_content": preprocessed_response,
            "score": score
            })
        return JUDGED(score)

    
    def store_to(self, file_path):
        """
        Store (append) results to file.
        
        :params file_path: The path to store the results. Support CSV, XLSX and JSONL format.

        """
        writer, ext = get_writer(file_path)
        if ext == None:
            raise ValueError(f"Storing to unsupported file format: \"{file_path}\". Please use csv, xlsx or jsonl.")
        
        dirname = os.path.dirname(file_path)
        if dirname != "" and not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        # Handle concurrent file writing between jobs. Default: 2 retries, 5 sec interval
        max_retries = 2 # set max retry count
        interval = 5 # in seconds
        retry = 0
        
        for _ in range(max_retries + 1): # range has exclusive upper bound
            try:
                writer(file_path, self.responses)
                break
            except IOError:
                if retry < max_retries:
                    retry += 1
                    logger.error(f"Failed to store response results to {file_path}. Retry {retry}/{max_retries} in {interval} second(s)...")
                    time.sleep(interval)
                else:
                    raise IOError(f"Failed to store response results to {file_path} after {max_retries} retries.")
