from worker import RequestParams, Worker
from dotenv import load_dotenv
import asyncio
import os
from dataset_adapters.ceval import conduct_ceval, make_system_prompt as make_ceval_system_prompt

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "yi-lightning"

async def main():
    """
    A demonstration on how to run typical dataset evaluations.
    """
    
    # Locate datasets (not packed within)
    CEVAL_DIR = "datasets/ceval/formal_ceval/val"
    # CMMLU_DIR = ...
    # MMLU_DIR = ...
    
    # 
    # Worker creation
    # 
    ceval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=128,
        system_prompt=make_ceval_system_prompt()
    )
    # cmmlu_worker_profile = RequestParams(...)
    # mmlu_worker_profile = RequestParams(...)
    
    ceval_worker = Worker(ceval_worker_profile)
    # cmmlu_worker = Worker(cmmlu_worker_profile)
    # mmlu_worker = Worker(mmlu_worker_profile)
    
    # 
    # Start evaluation
    # 
    await conduct_ceval(CEVAL_DIR, ceval_worker, test_mode=True)

    # Demo: Organize eval tasks for multiple models/datasets
    # 
    # tasks = [] 
    # async def subtask():
    #     await conduct_ceval(DATASET_DIR, ceval_worker)
    #     await conduct_cmmlu(DATASET2_DIR, cmmlu_worker)
    # 
    # tasks.append(subtask())
    # tasks.append(conduct_mmlu(DATASET3_DIR, mmlu_worker))
    # 
    # await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
