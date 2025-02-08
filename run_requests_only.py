from dataset_adapters.batch_query import batch_query
from worker import RequestParams, Worker
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

"""
    Run requests only, without score judging
    
    - query_file_path
    - workers: Evaluation multiple models together against the file. Read README for more worker details.
    - output_dir: The directory to store the result file. e.g. "results/batch_query/example_query_file_responses.xlsx"
    - test_mode: Only run first 10 queries from the file. For debuf purposes. Default to False.
    
    Moreover, you need to specify the following test-specific fields in batch_query.py
    
    - query_key: The key for evaluation queries. Default to "query".
"""

# Worker parameters
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "yi-lightning"
SYSTEM_PROMPT = "你是一位审题专家，请根据选择题内容，根据对应的专业知识，在A/B/C/D四个选项中，选出最合适的选项。直接给出选项前对应的字母，不要给出任何其他内容。"

# Test set parameters
# Specify which key to query. Default to "query".
QUERY_KEY="query"

QUERY_FILE_PATH = "example_query_file.xlsx"
OUTPUT_DIR = "results/batch_query"

async def run_requests_only():

    worker_profile={
        "model": MODEL,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "max_tokens": 128,
        "system_prompt": SYSTEM_PROMPT
    }
    
    industrious_worker = Worker(RequestParams(**worker_profile))
    
    await batch_query(QUERY_FILE_PATH, [industrious_worker], output_dir=OUTPUT_DIR, test_mode=True, query_key=QUERY_KEY)
    
if __name__ == "__main__":
    asyncio.run(run_requests_only())