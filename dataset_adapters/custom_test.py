from worker import Worker
from dataset_models import QuerySet, ResponseSet
from pathfinders import parse_filename_from_path, sanitize_pathname
import os
from judgers.presets import STRICT_MATCH
from text_preprocessors import as_is

async def run_test(test_file_path: str, workers: list[Worker], output_dir="results/custom_tests",test_mode=False, judging_algorithm=STRICT_MATCH, judging_preprocessor=as_is, query_key="query", answer_key="answer"):
    """
    Conduct a custom evaluation with a test data file. Support xlsx, csv and jsonl.

    :params test_file_path: A file containing query strings. One query per line. Currently xlsx/csv/jsonl. Depend on ResponseSet.store_to method.
    :params list[Worker] workers: The workers to dispatch.
    :params str output_dir: Store result file in this directory. Default to: results/custom_tests
    :params bool test_mode: only the first subset under dataset_dir will be tested. Only for debug purposes.
    :params judging_algorithm: The judging algorithm used for score judging. Preset: STRICT_MATCH (for A == A), TEXT_SIMILARITY (based on minimal editing steps), and MODEL_SCORING (submitted to a judger model).
    :params judging_preprocessor: Preprocess the response before score judging. Default to as_is.
    :params query_key: The key for evaluation queries. Set as your query file requires.
    :params answer_key: Conduct score judging based on this key. Required for score judging.
    """
    
    # Check if both query_file_path and output_dir exist
    if not os.path.isfile(test_file_path):
        raise FileNotFoundError(f"Speficied query file is not found: {test_file_path}")
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {output_dir}")
    
    if not answer_key:
        raise ValueError(f"Answer_key is required for score judging. Got {answer_key}.")
    
    query_set = QuerySet(test_file_path)
    
    if test_mode:
        query_set = query_set[:10]
            
    QUERY_SET_NAME = parse_filename_from_path(test_file_path)
    # For aggregated output
    aggregated_response_list = []
    
    for i, worker in enumerate(workers):
        model_name = worker.get_params()["model"]
        model_response_key = f"{model_name}_response"
        response_set = await worker(query_set, query_key=query_key, response_key=model_response_key).invoke()
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
            
    aggregated_response_set = ResponseSet(aggregated_response_list, query_key=query_key)
    
    for worker in workers:
        model_name = worker.get_params()["model"]
        model_response_key = f"{model_name}_response"
        model_eval_name = f"{model_name}"
        await aggregated_response_set.judge(answer_key=answer_key,
                                      context_key=query_key,
                                      eval_name=model_eval_name,
                                      response_preprocessor=judging_preprocessor,
                                      judger=judging_algorithm,
                                      foreign_response_key=model_response_key)
    
    output_path = os.path.join(output_dir, sanitize_pathname(f"{QUERY_SET_NAME}_results.xlsx"))
    
    aggregated_response_set.store_to(output_path)