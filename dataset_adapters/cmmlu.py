import os
from tqdm import tqdm
from dataset_models import QuerySet, ResponseSet
from typing import Callable
from worker import Worker
from pathfinders import list_files_in_directory, craft_result_path, craft_eval_dir_path, parse_filename_from_path
from judgers.presets import STRICT_MATCH
from resultfile_logger import log_resultfile
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JUDGER=STRICT_MATCH

async def conduct_cmmlu(dataset_dir: str, worker: Worker, response_preprocessor: Callable[[str], str], results_dir="results", score_output_path="model_results.xlsx", shuffled=False, subset_max_size=0, test_mode=False):
    """
    Conduct a cmmlu test. Before evaluation, create a worker instance.
    
    :params dataset_dir: The dataset should have the following directory structure:
    
    `$DATASET_NAME$ > $SUBSET_NAME$`
    as in `cmmlu/agronomy.csv`
    - Evaluation format supported: depend on ResponseSet.store_to method
    
    :params worker: The industrious worker.
    :params Callable[[str], str] response_preprocessor: Preprocess model responses before they go to the court. Select on your need.
    :params results_dir: Store evaluation results in this directory. e.g. results/cmmlu/athene-v2-chat/test-athene-v2-chat-agronomy.xlsx
    :params score_output_path: Store a score summary. Format supported: same as "Evaluation format supported".
    :params shuffled: Each query evaluates with shuffled options. 
    :params int subset_max_size: 0 (default) = eval all entries; 50 = the first 50 entries of each subfield, etc.
    :params test_mode: only first 10 questions from first subset under dataset_dir will be evaluated. Only for debug purposes.
    """
    DATASET_NAME = "cmmlu"
    MODEL = worker.get_params()["model"]
    ANSWER_KEY = "Answer"
    QUERY_KEY = "Question"
    # Check if both dataset_dir and results_dir exist
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory is not found: {dataset_dir}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {results_dir}")
    
    async def task(query_set: QuerySet):
        response_set = await worker(query_set, QUERY_KEY).invoke()

        subset_path = query_set.get_path()
        # this get_path method can return None when query_set is instantiated with a literal query string list. However, this wouldn't happen in dataset evaluation. No need for None safety validation.
        
        # Use query set path basename as eval name as each subset should have its distinctive name.
        score_result = await response_set.judge(answer_key=ANSWER_KEY, 
                                          eval_name=f"{parse_filename_from_path(subset_path)}",
                                          response_preprocessor = response_preprocessor,
                                          judger=JUDGER)
        
        # Store response with score info updated in response_set
        response_set.store_to(craft_result_path(query_set, results_dir, DATASET_NAME, MODEL))
        
        score_result.update({"dataset": DATASET_NAME, "model": MODEL})
        ResponseSet([score_result]).store_to(score_output_path)
        
    # Create QuerySet instances from dataset paths
    subset_paths = list_files_in_directory(dataset_dir, ".csv")
    datasets = [QuerySet(subset_path) for subset_path in subset_paths]
    
    # Test mode: Only the first subset will be evaluated.
    if test_mode:
        preview_eval_counts(datasets)
        datasets = [datasets[0]]
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)
        
    tasks = []
    
    for raw_dataset in datasets:
        # The original cmmlu test set contains 5 mcq fields. Need to merge them into one.
        # Test mode: Only the first 10 queries will be evaluated.
        if test_mode:
            raw_dataset = raw_dataset[:10]
        else:
            if subset_max_size > 0:
                raw_dataset = raw_dataset[:subset_max_size]
                
        # Keys are merged into a question field, overwriting the existing field
        dataset = raw_dataset.merge_keys([QUERY_KEY, "A", "B", "C", "D"], "Question")
        
        if shuffled:
            raw_dataset = raw_dataset.mcq_shuffle(ANSWER_KEY, ANSWER_KEY)
        # Create a hint message
        dataset_size = len(dataset)
        print(f"Conducting test: {dataset.get_path()} ({dataset_size})")
        # Task pool has been deprecated. Execute tasks synchronously. Each task is still done asynchronously with batch_size in .env file.
        tasks.append(task(dataset))
            
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{DATASET_NAME}: Task Completion Progress", position=0):
        await completed_task
            
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "mcq",
            "judging_method": response_preprocessor.__name__,
            "subset_max_size": subset_max_size
        }
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir, params=params)
    log()
    
def preview_eval_counts(query_set_list: list[QuerySet]):
    preview_message = f"""
    ======EVAL CHECKLIST======
    |\tEval name\t|\tSize\t|
    {"\n".join([f"|\t{query_set.get_path()}\t|\t{len(query_set)}|" for query_set in query_set_list])}
    ============
    """
    logger.info(preview_message)