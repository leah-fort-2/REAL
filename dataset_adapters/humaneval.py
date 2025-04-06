from worker import Worker
from dataset_models import QuerySet, ResponseSet
import os
import logging
from external_eval_methods.humaneval_eval.evaluation import humaneval_eval_raw_pass
from pathfinders import craft_result_path, craft_eval_dir_path
from resultfile_logger import log_resultfile
from text_preprocessors import clean_humaneval_preprocessor, clean_humaneval_cot_preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HumanEval has dedicated preprocessing methods. clean_humaneval_preprocessor is for non-cot eval and the other one, you bet.
RESPONSE_PREPROCESSOR = clean_humaneval_preprocessor

async def conduct_humaneval(humaneval_file_path: str, worker: Worker, results_dir="results", score_output_path="model_results.xlsx", test_mode=False):
    """
    Conduct humaneval evaluation through its test file.

    :params humaneval_file_path: The humaneval test file path.
    :params worker: The industrious worker.
    :params str results_dir: Store result file in this directory. Default to: results
    :params bool test_mode: only the first 10 questions are tested. Only for debug purposes.
    """
    # humaneval specific settings, do not modify
    QUERY_KEY = "prompt"
    RESPONSE_KEY = "completion"
    DATASET_NAME = "humaneval"
    MODEL = worker.get_params()["model"]
    # Check if both query_file_path and output_dir exist
    if not os.path.isfile(humaneval_file_path):
        raise FileNotFoundError(f"Speficied humaneval file is not found: {humaneval_file_path}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {results_dir}")
    
    query_set = QuerySet(humaneval_file_path)
    
    if test_mode:
        query_set = query_set[:10]
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)
        
    
    logger.info(f"Conducting test: {humaneval_file_path} ({len(query_set)})")
    response_set: ResponseSet = await worker(query_set, query_key=QUERY_KEY, response_key=RESPONSE_KEY).invoke()
    
    # Score judging for IFEval
    score_entry = humaneval_eval_raw_pass(response_set, humaneval_file_path, response_preprocessor=RESPONSE_PREPROCESSOR)
    score_entry.update({"dataset": DATASET_NAME, "model": MODEL})
    
    # Store (Append to) result file
    result_filename = craft_result_path(query_set, results_dir, DATASET_NAME, MODEL, file_ext="xlsx")
    response_set.store_to(result_filename)
    # Store ifeval score to score_output_path
    ResponseSet([score_entry]).store_to(score_output_path)
    
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "humaneval",
            "judging_method": humaneval_eval_raw_pass.__name__
        }
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir, params=params)
    log()
    

def make_system_prompt():
    return """You are a coder. Complete the following code block according to the docstring with proper indentation. Provide ONLY the completion without additional content.
"""

def make_prompt_suffix():
    return """# YOUR COMPLETION STARTS HERE
"""