import os
from dataset_models import QuerySet, ResponseSet
from worker import Worker
from dataset_adapters.pathfinders import list_files_in_directory, craft_result_path, craft_eval_dir_path, parse_filename_from_path
from dataset_adapters.preset_preprocessors import mcq_cot_preprocessor
from dataset_adapters.resultfile_logger import log_resultfile
import asyncio

async def conduct_ceval(dataset_dir: str, worker: Worker, results_dir: str, score_output_path="model_results.xlsx", test_mode=False):
    """
    Conduct a ceval test. Before evaluation, create a worker instance.
    
    - dataset_dir: The dataset should have the following directory structure: 
    
    `$DATASET_NAME$ > $SUBSET_NAME$`
    as in `ceval/accountant.csv`
    - Evaluation format supported: depend on ResponseSet.store_to method
    
    - results_dir: Store evaluation results in this directory. e.g. results/ceval/athene-v2-chat/test-athene-v2-chat-accountant.xlsx
    
    - score_output_path: Store a score summary. Format supported: same as "Evaluation format supported".
    
    - Test mode: only the first subset under dataset_dir will be evaluated. Only for debug purposes.
    
    """
    DATASET_NAME = "ceval"
    MODEL = worker.get_params()["model"]
    # Check if both dataset_dir and results_dir exist
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory is not found: {dataset_dir}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {results_dir}")
    
    async def task(query_set: QuerySet):
        response_set = await worker(query_set, "question").invoke()
        
        subset_path = query_set.get_path()
        # this get_path method can return None when query_set is instantiated with a literal query string list. However, this wouldn't happen in dataset evaluation. No need for None safety validation.

        # Use query set path basename as eval name as each subset should have its distinctive name.
        score_result = response_set.judge(answer_key="answer", 
                                          eval_name=f"{parse_filename_from_path(subset_path)}", 
                                          response_preprocessor = mcq_cot_preprocessor)
        
        # Store response with score info updated in response_set
        response_set.store_to(craft_result_path(query_set, results_dir, DATASET_NAME, MODEL))

        score_result.update({"dataset": DATASET_NAME, "model": MODEL})
        ResponseSet([score_result]).store_to(score_output_path)
        
    # Create QuerySet instances from dataset paths
    datasets = list_files_in_directory(dataset_dir, ".csv")
    
    # Test mode: Only the first subset will be evaluated.
    if test_mode:
        datasets = [datasets[0]]
        
    for subset_path in datasets:
        # The original ceval test set contains 5 mcq fields. Need to merge them into one.
        raw_dataset = QuerySet(subset_path)
        # Keys are merged into a question field, overwriting the existing field
        dataset = raw_dataset.merge_keys(["question", "A", "B", "C", "D"], "question")
        
        # Create a hint message
        dataset_size = len(dataset)
        print(f"Conducting test: {dataset.get_path()} ({dataset_size})")
        # Task pool has been deprecated. Execute tasks synchronously. Each task is still done asynchronously with batch_size in .env file.
        await task(dataset)
        
        # Very long haul! Add 120 sec break
        await asyncio.sleep(120)

            
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir)
    log()
    
def make_system_prompt():
    return f"你是一位审题专家，请根据选择题内容，根据对应的专业知识，在A/B/C/D四个选项中，选出最合适的选项。直接给出选项前对应的字母，不要给出任何其他内容。"