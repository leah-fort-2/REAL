from dataset_adapters.ifeval import conduct_ifeval
from worker import RequestParams, Worker
from dotenv import load_dotenv
import asyncio
import os
from dataset_adapters.ceval_val import conduct_ceval_val, make_system_prompt as make_ceval_system_prompt
from dataset_adapters.cmmlu import conduct_cmmlu, make_system_prompt as make_cmmlu_system_prompt
from dataset_adapters.mmlu import conduct_mmlu, make_system_prompt as make_mmlu_system_prompt
from dataset_adapters.gpqa import conduct_gpqa, make_system_prompt as make_gpqa_system_prompt, make_suffix as make_gpqa_suffix
from dataset_adapters.humaneval import conduct_humaneval, make_system_prompt as make_humaneval_system_prompt, make_prompt_suffix as make_humaneval_suffix
from dataset_adapters.mmlu_pro import conduct_mmlu_pro, make_system_prompt as make_mmlu_pro_system_prompt
from dataset_adapters.supergpqa import conduct_supergpqa, make_system_prompt as make_supergpqa_system_prompt, make_prompt_suffix as make_supergpqa_suffix

load_dotenv()

"""
In this file, a workflow is demonstrated with 4 models evaluated in one go. 

You may run them sequentially, or if resources allow it, concurrently.

prompt specifics are configured in the respective adapter file, as they do be quite "specific". However, these are just my recommendation. Use whatever you like here.
"""

# Load custom BASE_URL and API_KEY from .env file, or specify them at runtime

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

async def main():
    
    # Locate datasets (not packed within)
    CEVAL_DIR = "datasets/ceval/formal_ceval/val"
    CMMLU_DIR = "datasets/cmmlu/test"
    MMLU_DIR = "datasets/mmlu/test"
    GPQA_DIR = "datasets/GPQA"
    IFEVAL_PATH="datasets/IFEval/data/input_data.jsonl"
    HUMANEVAL_PATH = "datasets/humaneval/human-eval-v2-20210705.jsonl"
    MMLU_PRO_PATH = "datasets/mmlu_pro/data/test.jsonl"
    SUPERGPQA_PATH = "datasets/SuperGPQA/SuperGPQA-all.jsonl"
    
    # 
    # Worker creation
    # 
    ceval_val_worker_profile = RequestParams(
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
    ifeval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048
    )
    humaneval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=512,
        system_prompt=make_humaneval_system_prompt(),
        prompt_suffix=make_humaneval_suffix()
    )
    mmlu_pro_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=128,
        system_prompt=make_mmlu_pro_system_prompt()
    )
    # Using cot in supergpqa evaluation. Use the following non-cot prompt as you need (which is aligned to mmlu settings)
    def temp_system_prompt() -> str:
        return f"You are a professional exam question verifier. Answer the given Multiple Choice Question with your expertise in the corresponding domain. Present ONLY the correct option letter without any additional content."
    supergpqa_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        system_prompt=make_supergpqa_system_prompt(),
        prompt_suffix=make_supergpqa_suffix()
    )
    
    ifeval_worker = Worker(ifeval_worker_profile)
    ceval_worker = Worker(ceval_val_worker_profile)
    cmmlu_worker = Worker(cmmlu_worker_profile)
    mmlu_worker = Worker(mmlu_worker_profile)
    gpqa_worker = Worker(gpqa_worker_profile)
    mmlu_pro_worker = Worker(mmlu_pro_worker_profile)
    humaneval_worker = Worker(humaneval_worker_profile)
    supergpqa_worker = Worker(supergpqa_worker_profile)
    
    # 
    # Start evaluation
    # 
    await conduct_humaneval(HUMANEVAL_PATH, humaneval_worker)
    await conduct_gpqa(GPQA_DIR, gpqa_worker)
    await conduct_ifeval(IFEVAL_PATH, ifeval_worker)
    await conduct_ceval_val(CEVAL_DIR, ceval_worker)
    await conduct_cmmlu(CMMLU_DIR, cmmlu_worker)
    await conduct_mmlu(MMLU_DIR, mmlu_worker)
    await conduct_mmlu_pro(MMLU_PRO_PATH, mmlu_pro_worker)
    await conduct_supergpqa(SUPERGPQA_PATH, supergpqa_worker)

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