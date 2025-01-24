from worker import RequestParams, Worker
from dataset_models import QuerySet
from dotenv import load_dotenv
import asyncio
import os
from eval_utils import sanitize_pathname

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "anthropic.claude-3.5-sonnet-v2"

async def main():
    # Example usage with deepseek. But you can use any api provider:    

    # Step 1: Instantiate a QuerySet object
    
    # Only csv and xlsx files supported
    # Prepare query_set
    ES_QUERY_SET = QuerySet("loc_es.xlsx")
    PT_QUERY_SET = QuerySet("loc_pt.xlsx")
    FR_QUERY_SET = QuerySet("loc_fr.xlsx")

    # Step 2: Create a RequestParams object. It's like a reusable worker profile.
    
    pt_system_prompt=make_prompt("Portuguese (Brazil)")
    es_system_prompt=make_prompt("Spanish (Spain)")
    fr_system_prompt=make_prompt("French (France)")
    pt_params = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        system_prompt=pt_system_prompt
    )
    es_params = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        system_prompt=es_system_prompt
    )
    fr_params = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        system_prompt=fr_system_prompt
    )
    
    # Create a worker (QuerySet-> ResponseSet)
    pt_worker = Worker(pt_params)
    fr_worker = Worker(fr_params)
    es_worker = Worker(es_params)
    # Step 3: Organize tasks
    
    async def task(worker, query_set):
        query_set_name = query_set.get_path()
        output_path=sanitize_pathname(f"eval_result-{query_set_name}")
        print(f"Testing: {query_set_name}. Dataset size: {len(query_set)}.")
        
        async def subtask(chunk, output_path, query_key):
            responses = await worker(chunk, query_key).invoke()
            responses.store_to(output_path)
        
        subtask_list = []
        for chunk in query_set.divide(10):
            subtask_list.append(subtask(chunk, output_path, query_key="query"))
            
        await asyncio.gather(*subtask_list)
    # Step 4: Hit and run!
    await asyncio.gather(
        task(es_worker, ES_QUERY_SET)
    )
    await asyncio.gather(
        task(pt_worker, PT_QUERY_SET), 
    )
    await asyncio.gather(
        task(fr_worker, FR_QUERY_SET)
    )
    
def make_prompt(target_locale_str):
    return f"You are a professional localization engineer and your language pair is Simplified Chinese => {target_locale_str}. You are working on a question set translation. A Chinese source wil be provided for contextual understanding only, followed by a localization text in the target language. You are not allowed to review or revise the source."

if __name__ == "__main__":
    asyncio.run(main())
