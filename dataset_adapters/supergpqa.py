import asyncio
from typing import Tuple
from tqdm import tqdm
from dataset_models import QuerySet, ResponseSet
from judgers.presets import STRICT_MATCH
from pathfinders import craft_eval_dir_path, sanitize_pathname, strip_trailing_slashes_from_path
from resultfile_logger import log_resultfile
from text_preprocessors import mcq_cot_preprocessor_for_bad_if
from worker import Worker
from collections import defaultdict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESPONSE_PREPROCESSOR = mcq_cot_preprocessor_for_bad_if
JUDGER = STRICT_MATCH

async def conduct_supergpqa(supergpqa_file_path: str, worker: Worker, results_dir="results", score_output_path="model_results.xlsx", test_mode=False):
    """
    Conduct an superGPQA evaluation. Before calling the method, create a worker.

    :params str dataset_dir: The mmlu pro dataset path (Prepare it as csv/xlsx/jsonl file. Por favor, do it yourself, I'm not going to add parquet support.)
    :params Worker worker: The worker to dispatch
    :params str results_dir: In which directory to store the responses + evaluation results. Default to "results" => "results/supergpqa/athene-v2-chat/test-athene-v2-chat-supergpqa.xlsx"
    :params str score_output_path: Where to store the evaluation summary. Default to "model_results.xlsx"
    :params bool test_mode: If enabled, only the first 10 queries will be evaluated. For debug purposes. Default to False.
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
    
    # Add test mode prefix to output path and dir 
    if test_mode:    
        results_dir = os.path.join("test/", results_dir)
        score_output_path = os.path.join("test/", score_output_path)

    giant_query_set = QuerySet(supergpqa_file_path)
    
    # 10k+ data entries. Splitting it into 200 chunks
    # If test_mode, only test the first 10 queries
    query_set_chunks = [giant_query_set[:10]] if test_mode else giant_query_set.divide(
        100)

    all_responses_categorized = defaultdict(list)

    async def task(query_set: QuerySet):
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
        response_set = await worker(mcq_query_set, query_key=QUERY_KEY).invoke()
        await response_set.judge(
            answer_key=ANSWER_KEY,
            eval_name=DATASET_NAME,
            response_preprocessor=RESPONSE_PREPROCESSOR)
        # Score has been annotated in each response object.
        responses = response_set.get_responses()

        # Classify responses by category
        responses_categorized = defaultdict(list)
        [responses_categorized[f"{response_obj[CATEGORY_KEY]}/{response_obj[CATEGORY_SUB_KEY_1]}/{response_obj[CATEGORY_SUB_KEY_2]}"].append(
            response_obj) for response_obj in responses]
        
        # Merge to all categorized responses
        [all_responses_categorized[category].extend(response_list)
         for category, response_list in responses_categorized.items()]
        
        # Store' em by category.
        [ResponseSet(response_list).store_to(
                craft_category_path(
                    results_dir,
                    DATASET_NAME,
                    MODEL,
                    category,
                    file_ext="jsonl")
            )
            for category, response_list in responses_categorized.items()]

    for i, query_set_chunk in tqdm(enumerate(query_set_chunks), total=len(query_set_chunks), desc=f"{DATASET_NAME}: Evaluation Progress"):
        await task(query_set_chunk)

        if i < len(query_set_chunks)-1:
            # Very long haul! Take a break between each chunk.
            await asyncio.sleep(60)
            
    # MMLU pro is a gigantic dataset: Score for each subset can only be calculated when the categorization is finished.
    score_summaries: list[dict] = calculate_scores_by_categories(
        all_responses_categorized,
        DATASET_NAME,
        MODEL
        )
    ResponseSet(score_summaries).store_to(score_output_path)

    # Initialize a RESULTFILE in evaluation results directory.
    def log():
        params = {
            "test_set_type": "mcq",
            "judging_method": RESPONSE_PREPROCESSOR.__name__
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

def calculate_scores_by_categories(responses_by_categories: dict[str, list[dict]], dataset_name: str, model: str):
    logger.info(f"Begin score calculation for {dataset_name}@{model}")
    def calculate_category_score(response_list: list[dict]) -> Tuple[float, float]:
        category_score = 0
        category_total = 0
        for response_obj in response_list:
            response_score = response_obj.get("score", None)
            if response_score is not None:
                category_score += response_score
                category_total += 1
        return (category_score, category_total)
    
    scores_by_categories = []
    
    for category, response_list in responses_by_categories.items():
        category_score, category_total_score = calculate_category_score(response_list)
        category_summary = {"eval_name": category, 
            "score": category_score,
            "full_score": category_total_score,
            "accuracy": category_score/category_total_score,
            "dataset": dataset_name,
            "model": model}
        logger.info(
            f"\n======\nEvaluation Report:\nEvaluation Name: {category}\nAccuracy: {category_score}/{category_total_score} ({round(100*category_score/category_total_score, 1)}%)\n======\n")
        
        scores_by_categories.append(category_summary)
    
    return scores_by_categories

def make_system_prompt() -> str:
    return f"You are a professional exam question verifier. Answer the given Multiple Choice Question with your expertise in the corresponding domain. Think step by step between <think> and </think> tags. After thinking, present ONLY the correct option letter without any additional content."

def make_prompt_suffix() -> str:
    return f"\nASSISTANT: Let's think step by step: "