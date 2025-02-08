from worker import Worker
from dataset_models import QuerySet, ResponseSet
from pathfinders import parse_filename_from_path, sanitize_pathname
import os
from judgers.presets import STRICT_MATCH
from text_preprocessors import as_is

# judging_algorithm: Choose a score judging algorithm. Presets: `STRICT_MATCH` (A == A), `TEXT_SIMILARITY` (Edit distance ratio), or `MODEL_JUDGING` (Use a judge model, configure in .env file).
JUDGING_ALGORITHM = STRICT_MATCH
# judging_preprocessor: Transform the response before putting it to score judging. Default: as_is (keep as is)
JUDGING_PREPROCESSOR = as_is

QUERY_KEY = "query"
ANSWER_KEY = "answer"

async def run_test(test_file_path: str, workers: list[Worker], output_dir="results/custom_tests",test_mode=False):
    """
    Conduct a custom evaluation with a test data file. Support xlsx, csv and jsonl.

    :params test_file_path: A file containing query strings. One query per line. Currently xlsx/csv/jsonl. Depend on ResponseSet.store_to method.
    :params list[Worker] workers: The workers to dispatch.
    :params str output_dir: Store result file in this directory. Default to: results/custom_tests
    :params bool test_mode: only the first subset under dataset_dir will be tested. Only for debug purposes.
    """
    
    # Check if both query_file_path and output_dir exist
    if not os.path.isfile(test_file_path):
        raise FileNotFoundError(f"Speficied query file is not found: {test_file_path}")
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {output_dir}")
    
    if not ANSWER_KEY:
        raise ValueError(f"Answer_key is required for score judging. Got {ANSWER_KEY}.")
    
    # Example file contains an MCQ test set. Need to merge the question and options into one query.
    query_set = QuerySet(test_file_path).merge_keys([QUERY_KEY, "A", "B", "C", "D"], QUERY_KEY)
    
    if test_mode:
        query_set = query_set[:10]
            
    QUERY_SET_NAME = parse_filename_from_path(test_file_path)
    # For aggregated output
    aggregated_response_list = []
    
    for i, worker in enumerate(workers):
        model_name = worker.get_params()["model"]
        model_response_key = f"{model_name}_response"
        response_set = await worker(query_set, query_key=QUERY_KEY, response_key=model_response_key).invoke()
        response_list = response_set.get_responses()
        if i == 0:
            # First job, initialize response list, no need to aggregate
            aggregated_response_list = response_list
            continue
            
        if i > 0:
            # Sequential update on existing job response set
            [existing_item.update(
                {
                    model_response_key: aggregating_item[model_response_key]  
                }
                ) for existing_item, aggregating_item in zip(aggregated_response_list, response_list)]
            
    aggregated_response_set = ResponseSet(aggregated_response_list, query_key=QUERY_KEY)
    
    for worker in workers:
        model_name = worker.get_params()["model"]
        model_response_key = f"{model_name}_response"
        model_eval_name = f"{model_name}"
        await aggregated_response_set.judge(answer_key=ANSWER_KEY,
                                      context_key=QUERY_KEY,
                                      eval_name=model_eval_name,
                                      response_preprocessor=JUDGING_PREPROCESSOR,
                                      judger=JUDGING_ALGORITHM,
                                      foreign_response_key=model_response_key)
    
    output_path = os.path.join(output_dir, sanitize_pathname(f"{QUERY_SET_NAME}_results.xlsx"))
    
    aggregated_response_set.store_to(output_path)