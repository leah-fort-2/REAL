from worker import RequestParams, Worker
from dotenv import load_dotenv
import asyncio
import os
from dataset_adapters.ceval import conduct_ceval, make_system_prompt as make_ceval_system_prompt
from dataset_adapters.cmmlu import conduct_cmmlu, make_system_prompt as make_cmmlu_system_prompt
from dataset_adapters.mmlu import conduct_mmlu, make_system_prompt as make_mmlu_system_prompt
from dataset_adapters.gpqa import conduct_gpqa, make_system_prompt as make_gpqa_system_prompt, make_suffix as make_gpqa_suffix
from dataset_models import QuerySet

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

BASE_URL = "http://192.168.1.9:6006/v1"
API_KEY = os.getenv("API_KEY")
MODEL = "calmerys-78b-orpo-v0.1-exl2-3.0bpw"

async def main():
    """
    A demonstration on how to run typical dataset evaluations.
    """
    
    # Locate datasets (not packed within)
    CEVAL_DIR = "datasets/ceval/formal_ceval/val"
    CMMLU_DIR = "datasets/cmmlu/test"
    MMLU_DIR = "datasets/mmlu/test"
    GPQA_DIR = "datasets/GPQA"
    
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
    cmmlu_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=128,
        system_prompt=make_cmmlu_system_prompt()
    )
    mmlu_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=128,
        system_prompt=make_mmlu_system_prompt()
    )
    gpqa_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=128,
        system_prompt=make_gpqa_system_prompt(),
        prompt_suffix=make_gpqa_suffix()
    )
    
    ceval_worker = Worker(ceval_worker_profile)
    cmmlu_worker = Worker(cmmlu_worker_profile)
    mmlu_worker = Worker(mmlu_worker_profile)
    gpqa_worker = Worker(gpqa_worker_profile)
    
    # 
    # Start evaluation
    # 
    await conduct_gpqa(GPQA_DIR, gpqa_worker)
    await conduct_ceval(CEVAL_DIR, ceval_worker)
    await conduct_cmmlu(CMMLU_DIR, cmmlu_worker)
    await conduct_mmlu(MMLU_DIR, mmlu_worker)

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