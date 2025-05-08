import asyncio
from typing import Callable
from tqdm import tqdm
from dataset_models import QuerySet, ResponseSet
from judgers.presets import STRICT_MATCH
from pathfinders import craft_eval_dir_path, sanitize_pathname, strip_trailing_slashes_from_path
from resultfile_logger import log_resultfile
from worker import Worker
import os
import logging
# from qwq_blacklist import blacklist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JUDGER = STRICT_MATCH

async def conduct_supergpqa(supergpqa_file_path: str, worker: Worker, response_preprocessor: Callable[[str], str], results_dir="results", score_output_path="model_results.xlsx", subset_max_size=0, test_mode=False, enable_metrics=False):
    """
    Conduct an superGPQA evaluation. Before calling the method, create a worker.

    :params str dataset_dir: The mmlu pro dataset path (Prepare it as csv/xlsx/jsonl file. Por favor, do it yourself, I'm not going to add parquet support.)
    :params Worker worker: The worker to dispatch
    :params Callable[[str], str] response_preprocessor: Preprocess model responses before they go to the court. Select on your need.
    :params str results_dir: In which directory to store the responses + evaluation results. Default to "results" => "results/supergpqa/athene-v2-chat/test-athene-v2-chat-supergpqa.xlsx"
    :params str score_output_path: Where to store the evaluation summary. Default to "model_results.xlsx"
    :params int subset_max_size: 0 (default) = eval all entries; 50 = the first 50 entries of each category, etc.
    :params bool test_mode: If enabled, only the first 3 subsets (in alphabetical order) will be tested. Output files will be placed under test/ directory. For debug purposes. Default to False.
    :params bool enable_metrics: Whether to read "usage" key from response body. Only available when the server enabled metrics.
    """
    DATASET_NAME = "supergpqa"
    MODEL = worker.get_params()["model"]
    ANSWER_KEY = "answer_letter"
    QUERY_KEY = "question"
    OPTIONS_KEY = "options"
    CATEGORY_KEY = "discipline"
    CATEGORY_SUB_KEY_1 = "field"
    CATEGORY_SUB_KEY_2 = "subfield"
    # {CATEGORY_KEY}/{CATEGORY_SUB_KEY_1}/{CATEGORY_SUB_KEY_2}
    
    # Check if both dataset_dir and results_dir exist
    if not os.path.isfile(supergpqa_file_path):
        raise FileNotFoundError(
            f"Speficied supergpqa file is not found:  {supergpqa_file_path}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Destination results directory is not found: {results_dir}")

    giant_query_set = QuerySet(supergpqa_file_path)
    
    # Split the query set by identifiers ("discipline/field/subfield") first.
    query_sets_by_identifiers = giant_query_set.divide_by_keys([CATEGORY_KEY, CATEGORY_SUB_KEY_1, CATEGORY_SUB_KEY_2], completeness=True)
    # Add test mode prefix to output path and dir 
    if test_mode:
        preview_eval_counts(query_sets_by_identifiers)
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)

    if subset_max_size > 0:
        [query_sets_by_identifiers.update(
            {identifier: subset[:subset_max_size]}
            )
         for identifier, subset in query_sets_by_identifiers.items()]
        
    id_query_set_pairs_sorted_in_alphabetical_order = sorted(list(query_sets_by_identifiers.items()), key=lambda tup: tup[0])
    if test_mode:
        id_query_set_pairs_sorted_in_alphabetical_order = id_query_set_pairs_sorted_in_alphabetical_order[:3]

    async def task(identifier: str, query_set: QuerySet):
        """
        1. Create temporary MCQ subset
        
        2. Do the request and sequential judge work
        
        3. Aggregate responses to the above global dict by categories
        
        4. Store responses by categories. 
        """
        # Query structure: [
        #     {... "question": ..., "options": [...], "answer": ..., ...}
        # ]:
        mcq_query_set = make_mcq_from_query_set(query_set, query_key=QUERY_KEY, options_key=OPTIONS_KEY)
        response_set = await worker(mcq_query_set, query_key=QUERY_KEY).invoke(enable_metrics=enable_metrics)
        score_summary = await response_set.judge(
            answer_key=ANSWER_KEY,
            eval_name=identifier,
            response_preprocessor=response_preprocessor)
        # Score has been annotated in each response object.
        responses = response_set.get_responses()

        # Store the category.
        ResponseSet(responses).store_to(
                craft_category_path(
                    results_dir,
                    DATASET_NAME,
                    MODEL,
                    identifier,
                    file_ext="jsonl")
            )
        
        # Calculate score for each identifier.
        score_summary.update({
            "dataset": DATASET_NAME,
            "model": MODEL
            })
        if enable_metrics:
            score_summary.update({"total_output_tokens": sum([query["output_tokens"] for query in response_set.get_responses()])})
        ResponseSet([score_summary]).store_to(score_output_path)
    
    tasks = []
    for identifier, query_set in id_query_set_pairs_sorted_in_alphabetical_order:
        tasks.append(task(identifier, query_set))

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
    
def make_mcq_from_query_set(query_set: QuerySet, query_key:str, options_key: str):
    """
    Create a typical MCQ query set from mmlu pro dataset.
    
    Source: `[... "options": "[]"]`
    
    query_key will be overwritten by the new mcq query field. e.g. `Q? \\nA. Answer\\nB. Answer...`
    """
    new_queries = query_set.get_queries()
    [query_obj.update(
        {query_key: f"{query_obj[query_key]}\n"
            + "\n".join(
                [f"{LETTER}. {CONTENT}" for LETTER, CONTENT
                    in zip(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    query_obj[options_key])])}
    ) for query_obj in new_queries]
    new_query_set = QuerySet(new_queries)
    new_query_set.file_path = query_set.get_path()
    return new_query_set

def craft_category_path(results_dir: str, dataset_name: str, model: str, category: str, file_ext):
        return f"{strip_trailing_slashes_from_path(results_dir)}/{dataset_name}/{model}/{sanitize_pathname(f"test-{model}-{category}")}.{file_ext}"
    
        
def preview_eval_counts(query_sets_by_identifiers: dict[str, QuerySet]):
    preview_message = f"""
    ======EVAL CHECKLIST======
    |\tEval name\t|\tSize\t|
    {"\n".join([f"|\t{category}\t|\t{len(query_set)}|" for category, query_set in query_sets_by_identifiers.items()])}
    ============
    """
    logger.info(preview_message)