from worker import Worker
from dataset_models import QuerySet, ResponseSet
from pathfinders import parse_filename_from_path, sanitize_pathname
import os

# Specify which key to query. Default to "query".
QUERY_KEY="query"

async def batch_query(query_file_path: str, workers: list[Worker], output_dir="results/batch_query", test_mode=False):
    """
    Conduct a naive batch query without score judging. 
    
    :params query_file_path: A file containing query strings. One query per line.
    
    - Evaluation format supported: Currently xlsx/csv/jsonl. Depend on ResponseSet.store_to method
    
    :params list[Worker] workers: The workers to dispatch.
    :params output_dir: Store result file in this directory. Default to: results/batch_query
    :params test_mode: only the first subset under dataset_dir will be evaluated. Only for debug purposes.
    """
    
    # Check if both query_file_path and output_dir exist
    if not os.path.isfile(query_file_path):
        raise FileNotFoundError(f"Speficied query file is not found: {query_file_path}")
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Destination results directory is not found: {output_dir}")
    
    # Example file contains an MCQ test set. Need to merge the question and options into one query.
    query_set = QuerySet(query_file_path).merge_keys([QUERY_KEY, "A", "B", "C", "D"], QUERY_KEY)
        
    if test_mode:
        query_set = query_set[:10]
            
    QUERY_SET_NAME = parse_filename_from_path(query_file_path)
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
                    model_response_key: aggregating_item[QUERY_KEY]  
                }
                ) for existing_item, aggregating_item in zip(aggregated_response_list, response_list)]
            
    aggregated_response_set = ResponseSet(aggregated_response_list, query_key=QUERY_KEY)
    
    output_path = os.path.join(output_dir, sanitize_pathname(f"{QUERY_SET_NAME}_responses.xlsx"))
    
    aggregated_response_set.store_to(output_path)