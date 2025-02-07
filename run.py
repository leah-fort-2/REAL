from worker import RequestParams, Worker
from dotenv import load_dotenv
import asyncio
import os
from dataset_adapters.ceval import conduct_ceval, make_system_prompt as make_ceval_system_prompt
from dataset_adapters.cmmlu import conduct_cmmlu, make_system_prompt as make_cmmlu_system_prompt
from dataset_adapters.mmlu import conduct_mmlu, make_system_prompt as make_mmlu_system_prompt

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
BACKUP_URL = "http://192.168.1.9:6007/v1"
MODEL = "deepseek-r1-distill-qwen-32b-exl2-4.5bpw"
# Change: timeout 60s => 144s in api_actions; mcq_preprocessor => mcq_cot_preprocessor in dataset_adapters.preset_proprocesssors; add break in adapters
DATASET_NAME = "ceval"
DATASET2_NAME = "cmmlu"
DATASET3_NAME = "mmlu"

async def main():
    
    # ceval
    DATASET_DIR = "datasets/ceval/formal_ceval/val"
        
    # cmmlu
    DATASET2_DIR = "datasets/cmmlu/test/"
    
    # mmlu
    DATASET3_DIR = "datasets/mmlu/test/"
    
    ceval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        system_prompt=make_ceval_system_prompt()
    )
    cmmlu_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        system_prompt=make_cmmlu_system_prompt()
    )
    mmlu_worker_profile = RequestParams(
        base_url=BACKUP_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        system_prompt=make_mmlu_system_prompt()
    )
    
    ceval_worker = Worker(ceval_worker_profile)
    cmmlu_worker = Worker(cmmlu_worker_profile)
    mmlu_worker = Worker(mmlu_worker_profile)
    
    tasks = [] 
    async def subtask():
        await conduct_ceval(DATASET_DIR, ceval_worker)
        await conduct_cmmlu(DATASET2_DIR, cmmlu_worker)
    
    tasks.append(subtask())
    tasks.append(conduct_mmlu(DATASET3_DIR, mmlu_worker))
    
    await asyncio.gather(*tasks)
    


if __name__ == "__main__":
    asyncio.run(main())
