from worker import RequestParams, Worker
from dataset_models import QuerySet
from dotenv import load_dotenv
import asyncio
import os
from eval_utils import list_files_in_directory, sanitize_pathname

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

async def main():
    # Example usage with deepseek. But you can use any api provider:    

    # Step 1: Instantiate a QuerySet object
    
    # Only csv and xlsx files supported
    DATASET_DIR="dataset"
    dataset_paths = list_files_in_directory(DATASET_DIR, ".csv")

    # Prepare query_set
    query_set = QuerySet(dataset_paths[0])
    # The dataset contains options, so concatenate options with the question to create an updated temp query set
    # A "query" key is inserted into each query dict.
    updated_query_set = query_set.merge_keys(["question", "A", "B", "C", "D"], "query")
    # You can use a local list instead
    
    # query_list = [
    #     "What's the city with the lowest annual average temperature in the world?",
    #     "Which city has the highest population density in the world?",
    #     "What is the capital of France?",
    #     "Who wrote 'Romeo and Juliet'?",
    #     "What is the largest ocean on Earth?",
    #     "Which country is known as the Land of the Rising Sun?",
    #     "What is the currency of Japan?",
    #     "Who painted the Mona Lisa?",
    #     "What is the tallest mountain in the world?"
    # ]
    # query_set = QuerySet(query_list)

    # Step 2: Create a RequestParams object. It's like a reusable worker profile.
    
    solo_system_prompt="You are a helpful assistant. Select the correct or the most appropriate option."
    MODEL = "qwen2.5-coder-instruct"
    solo_params = RequestParams(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        system_prompt=solo_system_prompt
    )
    
    # Create a worker (QuerySet-> ResponseSet)
    solo_worker = Worker(solo_params)
    # Step 3: Organize tasks
    
    async def task():        
        query_set_name = updated_query_set.get_path()
        output_path=sanitize_pathname(f"eval_result-{query_set_name}")
        print(f"Testing: {query_set_name}. Dataset size: {len(updated_query_set)}。")
        
        async def subtask(chunk, output_path, query_key):
            responses = await solo_worker(chunk, query_key).invoke()
            responses.store_to(output_path)
        
        subtask_list = []
        for chunk in updated_query_set.divide(10):
            subtask_list.append(subtask(chunk, output_path, query_key="query"))
            
        await asyncio.gather(*subtask_list)
    # Step 4: Hit and run!
    await asyncio.gather(task(query_set))

if __name__ == "__main__":
    asyncio.run(main())