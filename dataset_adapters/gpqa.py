import os
from dataset_models import QuerySet, ResponseSet
from worker import Worker
from typing import Callable
from pathfinders import list_files_in_directory, craft_result_path, craft_eval_dir_path, parse_filename_from_path
from judgers.presets import STRICT_MATCH
from resultfile_logger import log_resultfile
import asyncio
from random import shuffle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JUDGER=STRICT_MATCH

async def conduct_gpqa(dataset_dir: str, worker: Worker, response_preprocessor: Callable[[str], str], results_dir="results", score_output_path="model_results.xlsx", test_mode=False, enable_metrics=False):
    """
    Conduct a gpqa test. Before evaluation, create a worker instance.
    
    :params dataset_dir: The dataset should have the following directory structure:
    
    `$DATASET_NAME$ > $SUBSET_NAME$`
    For GPQA, no subset_name since it only has one main test file `gpqa_main.csv`.
    
    - Evaluation format supported: depend on ResponseSet.store_to method
    
    :params worker: The industrious worker.
    :params Callable[[str], str] response_preprocessor: Preprocess model responses before they go to the court. Select on your need.
    :params results_dir: Store evaluation results in this directory. e.g. results/gpqa/athene-v2-chat/test-athene-v2-chat-gpqa_main.xlsx
    :params score_output_path: Store a score summary. Format supported: same as "Evaluation format supported".
    :params test_mode: only first 3 questions from first subset under dataset_dir will be evaluated. Only for debug purposes.
    :params bool enable_metrics: Whether to read "usage" key from response body. Only available when the server enabled metrics.
    """
    DATASET_NAME = "gpqa"
    MODEL = worker.get_params()["model"]
    
    # # !!!Override the model name to bypass the vllm inline model loading!!!
    # worker.request_params.model = "gemma-3-27b-it-gptq-4.0bit-128g"
    
    # Check if both dataset_dir and results_dir exist
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory is not found: {dataset_dir}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {results_dir}")
    
    original_query_key = "Question"
    original_answer_key = "Correct Answer"
    original_option_keys = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
            
    target_query_key = "query"
    target_answer_key = "answer"
    target_option_keys = ["A", "B", "C", "D"]
    
    async def task(query_set: QuerySet):
        response_set = await worker(query_set, "query").invoke(enable_metrics=enable_metrics)

        subset_path = query_set.get_path()
        # this get_path method can return None when query_set is instantiated with a literal query string list. However, this wouldn't happen in dataset evaluation. No need for None safety validation.
        
        # Use query set path basename as eval name as each subset should have its distinctive name.
        score_result = await response_set.judge(answer_key=target_answer_key, 
                                          eval_name=f"{parse_filename_from_path(subset_path)}",
                                          response_preprocessor = response_preprocessor,
                                          judger=JUDGER)
        
        # Store response with score info updated in response_set
        response_set.store_to(craft_result_path(query_set, results_dir, DATASET_NAME, MODEL))
        
        score_result.update({
            "dataset": DATASET_NAME,
            "model": MODEL
            })
        if enable_metrics:
            score_result.update({"total_output_tokens": sum([query["output_tokens"] for query in response_set.get_responses()])})
        ResponseSet([score_result]).store_to(score_output_path)
        
    # Create QuerySet instances from dataset paths (in this case, only one for GPQA)
    datasets = list_files_in_directory(dataset_dir, ".csv")
    
    # Test mode: Only the first subset will be evaluated.
    if test_mode:
        preview_eval_counts([QuerySet(subset_path) for subset_path in datasets])
        datasets = [datasets[0]]
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)
        
    for i, subset_path in enumerate(datasets):
        selected_keys = [original_query_key, original_answer_key, *original_option_keys]
        
        # Test mode: Only the first 3 queries will be evaluated.
        raw_dataset = QuerySet(subset_path, field_names=selected_keys)[:3] if test_mode else QuerySet(subset_path, field_names=selected_keys)

        # gpqa has the following data structure:
        # {... "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", ...} : dict
        # They will be shuffled into new mcq fields namely ABCD, with Correct Answer parsed to corresponding letter too
        shuffled_dataset = create_shuffled_gpqa_query_set(raw_dataset, original_answer_key, target_answer_key, original_option_keys, target_option_keys)
        
        # merge mcq keys into a unified query
        dataset: QuerySet = shuffled_dataset.merge_keys([original_query_key, *target_option_keys], target_query_key)
        dataset.file_path = raw_dataset.get_path()
        
        # Create a hint message
        dataset_size = len(dataset)
        print(f"Conducting test: {dataset.get_path()} ({dataset_size})")
        # Task pool has been deprecated. Execute tasks synchronously. Each task is still done asynchronously with batch_size in .env file.
        await task(dataset)
        
        # Very long haul! Add 120 sec break
        # However, no need to break after the last task.
        if i < len(datasets) - 1:
            await asyncio.sleep(120)
            
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "mcq",
            "judging_method": response_preprocessor.__name__
        }
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir, params=params)
    log()

def create_shuffled_gpqa_query_set(gpqa_dataset: QuerySet, original_answer_key: str, target_answer_key: str, original_option_keys: list[str], target_option_keys: list[str]):
    """
    Takes a gpqa dataset, shuffle the options, return the shuffled query set. Old query set remains unchanged.
    
    :params gpqa_dataset: The GPQA query_set object
    :params original_answer_key: In GPQA's case it's "Correct Answer"
    :params target_answer_key: Which key to store the updated answer marker after shuffling
    :params original_option_keys: In GPQA's case it's ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    :params target_option_keys: The keys to store the shuffled options. ["A", "B", "C", "D"] for typical mcq. The order of target options is retained.
    """
    # Extract the query object list. Will do some hacky manipulation before creating a temporary merged_queryset
    existing_queries = gpqa_dataset.get_queries()
    shuffled_queries = []
        
    for query_obj in existing_queries:
        shuffle(original_option_keys)
        
        new_query_obj = {}
        new_answer= ""
        # [A, B, C, D], shuffled(["Correct Answer", ...]) => {A: ..., B: ..., ...} 
        for target_option_key, original_option_key in zip(target_option_keys, original_option_keys):
            # {A: some option}
            new_query_obj[target_option_key]=query_obj[original_option_key]
            # The option key is "Correct Answer"
            if original_option_key == original_answer_key:
                # {answer: some option}
                new_answer=target_option_key
                
        # To keep the order of target keys, update the answer key at last.
        new_query_obj[target_answer_key]=new_answer
        shuffled_queries.append(new_query_obj)
        
    # As per now, we have a new list of query objects.
    # We will finish by updating the shuffled queries to existing queries, to keep other existing fields.   
    [existing_query.update(shuffled_query) for existing_query, shuffled_query in zip(existing_queries, shuffled_queries)]
        
    shuffled_dataset = QuerySet(existing_queries)
    # We are calling an internal property outside the class. This isn't the best practice, but it won't harm the performance, so use it for now.
    shuffled_dataset.file_path = gpqa_dataset.get_path()
    return shuffled_dataset

def preview_eval_counts(query_set_list: list[QuerySet]):
    preview_message = f"""
    ======EVAL CHECKLIST======
    |\tEval name\t|\tSize\t|
    {"\n".join([f"|\t{query_set.get_path()}\t|\t{len(query_set)}|" for query_set in query_set_list])}
    ============
    """
    logger.info(preview_message)