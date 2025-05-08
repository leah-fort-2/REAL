from tqdm import tqdm
from dataset_adapters.gpqa import conduct_gpqa
from dataset_adapters.mmlu import conduct_mmlu
from worker import RequestParams, Worker
from dotenv import load_dotenv
import asyncio
import os
from dataset_adapters.ifeval import conduct_ifeval
from dataset_adapters.ceval import conduct_ceval
from dataset_adapters.ceval_val import conduct_ceval_val
from dataset_adapters.cmmlu import conduct_cmmlu
from dataset_adapters.humanevalplus import conduct_humanevalplus, make_system_prompt as make_humanevalplus_system_prompt, make_prompt_suffix as make_humanevalplus_suffix
from dataset_adapters.mmlu_pro import conduct_mmlu_pro
from dataset_adapters.supergpqa import conduct_supergpqa
from prompts import make_en_system_prompt as make_system_prompt, make_zh_system_prompt as make_zh_system_prompt
from text_preprocessors import mcq_search_preprocessor

load_dotenv()

"""
In this file, a workflow is demonstrated with 4 models evaluated in one go. 

You may run them sequentially, or if resources allow it, concurrently.

prompt specifics are configured in the respective adapter file, as they do be quite "specific". However, these are just my recommendation. Use whatever you like here.
"""

# Load custom BASE_URL and API_KEY from .env file, or specify them at runtime

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "deepseek-chat"

async def main():
    
    # Locate datasets (not packed within)
    CEVAL_DIR = "datasets/ceval/formal_ceval/test"
    CEVAL_VAL_DIR = "datasets/ceval/formal_ceval/val"
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
    ceval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_zh_system_prompt(),
        # chat_template_args={"enable_thinking": False} # custom argument for qwen3
    )
    ceval_val_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_zh_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    mmlu_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    gpqa_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    cmmlu_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_zh_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    ifeval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        # chat_template_args={"enable_thinking": False}
    )
    humaneval_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_humanevalplus_system_prompt(),
        prompt_suffix=make_humanevalplus_suffix(),
        # chat_template_args={"enable_thinking": False}
    )
    mmlu_pro_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    supergpqa_worker_profile = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=8192,
        system_prompt=make_system_prompt(),
        # chat_template_args={"enable_thinking": False}
    )
    
    ceval_worker = Worker(ceval_worker_profile)
    ceval_val_worker = Worker(ceval_val_worker_profile)
    mmlu_worker = Worker(mmlu_worker_profile)
    gpqa_worker = Worker(gpqa_worker_profile)
    ifeval_worker = Worker(ifeval_worker_profile)
    cmmlu_worker = Worker(cmmlu_worker_profile)
    mmlu_pro_worker = Worker(mmlu_pro_worker_profile)
    humaneval_worker = Worker(humaneval_worker_profile)
    supergpqa_worker = Worker(supergpqa_worker_profile)
    
    # 
    # Start evaluation
    # 
    async def subtask1():
        # ceval is not operational at this moment as the dataset does not provide with it the answer field. Do not use.
        await conduct_ceval(CEVAL_DIR, ceval_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask2():
        await conduct_ceval_val(CEVAL_VAL_DIR, ceval_val_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask3():
        await conduct_mmlu(MMLU_DIR, mmlu_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask4():
        await conduct_gpqa(GPQA_DIR, gpqa_worker, test_mode=True, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask5():
        await conduct_humanevalplus(HUMANEVAL_PATH, humaneval_worker, test_mode=True, enable_metrics=True)
    async def subtask6():
        await conduct_ifeval(IFEVAL_PATH, ifeval_worker, test_mode=True, enable_metrics=True)
    async def subtask7():
        await conduct_cmmlu(CMMLU_DIR, cmmlu_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask8():
        await conduct_mmlu_pro(MMLU_PRO_PATH, mmlu_pro_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)
    async def subtask9():
        await conduct_supergpqa(SUPERGPQA_PATH, supergpqa_worker, test_mode=True, subset_max_size=1, response_preprocessor=mcq_search_preprocessor, enable_metrics=True)

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
    tasks = [subtask2(), subtask3(), subtask4(), subtask5(), subtask6(), subtask7(), subtask8(), subtask9()]
    # Display a task completion progress bar
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Task completion progress", position=0):
        await completed_task

if __name__ == "__main__":
    asyncio.run(main())