import os
from dataset_models import QuerySet, ResponseSet
from worker import Worker
from pathfinders import list_files_in_directory, craft_result_path, craft_eval_dir_path, parse_filename_from_path
from judgers.presets import STRICT_MATCH
from text_preprocessors import mcq_preprocessor
from resultfile_logger import log_resultfile
import asyncio

RESPONSE_PREPROCESSOR=mcq_preprocessor
JUDGER=STRICT_MATCH

async def conduct_ceval(dataset_dir: str, worker: Worker, results_dir="results", score_output_path="model_results.xlsx", shuffled=False, test_mode=False):
    """
    Conduct a ceval test. Before evaluation, create a worker instance.
    
    :params dataset_dir: The dataset should have the following directory structure: 
    
    `$DATASET_NAME$ > $SUBSET_NAME$`
    as in `ceval/accountant.csv`
    - Evaluation format supported: depend on ResponseSet.store_to method
    
    :params worker: The industrious worker.
    :params results_dir: Store evaluation results in this directory. e.g. results/ceval/athene-v2-chat/test-athene-v2-chat-accountant.xlsx
    :params score_output_path: Store a score summary. Format supported: same as "Evaluation format supported".
    :params shuffled: Each query evaluates with shuffled options. 
    :params test_mode: only first 10 questions from first subset under dataset_dir will be evaluated. Only for debug purposes.
    """
    DATASET_NAME = "ceval"
    MODEL = worker.get_params()["model"]
    ANSWER_KEY = "answer"
    QUERY_KEY = "question"
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
                                          response_preprocessor = RESPONSE_PREPROCESSOR,
                                          judger=JUDGER)
        
        # Store response with score info updated in response_set
        response_set.store_to(craft_result_path(query_set, results_dir, DATASET_NAME, MODEL))

        score_result.update({"dataset": DATASET_NAME, "model": MODEL})
        ResponseSet([score_result]).store_to(score_output_path)
        
    # Create QuerySet instances from dataset paths
    datasets = list_files_in_directory(dataset_dir, ".csv")
    
    # Test mode: Only the first subset will be evaluated.
    if test_mode:
        datasets = [datasets[0]]
        
    for i, subset_path in enumerate(datasets):
        # The original ceval test set contains 5 mcq fields. Need to merge them into one.
        # Test mode: Only the first 10 queries will be evaluated.
        raw_dataset = QuerySet(subset_path)[:10] if test_mode else QuerySet(subset_path)
        # Keys are merged into a question field, overwriting the existing field
        dataset = raw_dataset.merge_keys([QUERY_KEY, "A", "B", "C", "D"], "question")
        
        if shuffled:
            raw_dataset = raw_dataset.mcq_shuffle(ANSWER_KEY, ANSWER_KEY)
        # Create a hint message
        dataset_size = len(dataset)
        print(f"Conducting test: {dataset.get_path()} ({dataset_size})")
        # Task pool has been deprecated. Execute tasks synchronously. Each task is still done asynchronously with batch_size in .env file.
        await task(dataset)
        
        # Very long haul! Add 120 sec break
        # However, no need to break after the last task.
        if i < len(datasets) - 1:
            await asyncio.sleep(60)

            
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "mcq",
            "judging_method": RESPONSE_PREPROCESSOR.__name__
        }
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir, params=params)
    log()
    
def make_system_prompt():
    return f"你是一位审题专家，请根据选择题内容，根据对应的专业知识，在A/B/C/D四个选项中，选出最合适的选项。直接给出选项前对应的字母，不要给出任何其他内容。"