from worker import Worker
from dataset_models import QuerySet, ResponseSet
import os
import logging
from external_eval_methods.instruction_following_eval.evaluation_main import ifeval_judge_strict
from pathfinders import craft_result_path, craft_eval_dir_path
from text_preprocessors import as_is, remove_think_tags
from resultfile_logger import log_resultfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IfEval does not need response preprocessing (use as_is) unless you are evaluating reasoning models.
RESPONSE_PREPROCESSOR=remove_think_tags

async def conduct_ifeval(ifeval_src_file_path: str, worker: Worker, results_dir="results", score_output_path="model_results.xlsx", test_mode=False):
    """
    Conduct IFEval through its test file.

    :params ifeval_src_file_path: Ifeval test file path.
    :params worker: The industrious worker.
    :params str results_dir: Store result file in this directory. Default to: results
    :params score_output_path: Store a score summary. Format supported: same as "Evaluation format supported".
    :params bool test_mode: only the first 10 questions will be evaluated. Only for debug purposes.
    """
    # ifeval specific settings, do not modify
    QUERY_KEY = "prompt"
    RESPONSE_KEY = "response"
    DATASET_NAME = "IFEval"
    MODEL = worker.get_params()["model"]
    # Check if both query_file_path and output_dir exist
    if not os.path.isfile(ifeval_src_file_path):
        raise FileNotFoundError(f"Speficied ifeval file is not found: {ifeval_src_file_path}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {results_dir}")
    
    query_set = QuerySet(ifeval_src_file_path)
    
    if test_mode:
        query_set = query_set[:10]
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)
        
    
    logger.info(f"Conducting test: {ifeval_src_file_path} ({len(query_set)})")
    response_set: ResponseSet = await worker(query_set, query_key=QUERY_KEY, response_key=RESPONSE_KEY).invoke()
    
    # Score judging for IFEval
    score_entry = ifeval_judge_strict(response_set, ifeval_src_file_path, response_preprocessor=RESPONSE_PREPROCESSOR)
    score_entry.update({"dataset": DATASET_NAME, "model": MODEL})
    
    # Store (Append to) result file
    result_filename = craft_result_path(query_set, results_dir, DATASET_NAME, MODEL, file_ext="jsonl")
    response_set.store_to(result_filename)
    # Store ifeval score to score_output_path
    ResponseSet([score_entry]).store_to(score_output_path)
    
    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "others",
            "judging_method": ifeval_judge_strict.__name__
        }
        eval_dir = craft_eval_dir_path(results_dir, DATASET_NAME, MODEL)
        log_resultfile(DATASET_NAME, worker, eval_dir, params=params)
    log()