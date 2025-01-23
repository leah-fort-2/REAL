from worker import RequestParams, Worker
from dataset_models import QuerySet
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

LINGYI_BASE_URL = os.getenv('LINGYI_BASE_URL')
LINGYI_API_KEY = os.getenv('LINGYI_API_KEY')

async def main():
    # Example usage with deepseek. But you can use any api provider:    

    # Step 1: Instantiate a QuerySet object
    
    # Only csv and xlsx files supported
    # query_set=QuerySet("example_queries.csv")

    # You can use a local list instead
    
    query_list = [
        "What's the city with the lowest annual average temperature in the world?",
        "Which city has the highest population density in the world?",
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the largest ocean on Earth?",
        "Which country is known as the Land of the Rising Sun?",
        "What is the currency of Japan?",
        "Who painted the Mona Lisa?",
        "What is the tallest mountain in the world?"
    ]
    query_set = QuerySet(query_list)

    # Step 2: Create a RequestParams object. It's like a reusable worker profile.
    
    deepseek_params = RequestParams(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-chat"
        # prompt_prefix="Hello! ",
        # prompt_suffix="If you see this line, use 233 at the end of your response to hint me."
    )
    
    # Create a worker (QuerySet-> ResponseSet)
    deepseek_worker = Worker(deepseek_params)

    # Step 3: Organize tasks
    
    async def deepseek_task():
        
        # You can use chunked query_set to create sub jobs.
        for div in query_set.divide(10):
            # A worker can be called with a QuerySet instance.
            # Custom query_key available. Default to "query".
            result = await deepseek_worker(div, query_key="query").invoke()
            # Note: Will append to existing file
            result.store_to("Book1.csv")
    
    # Step 4: Hit and run!
    
    await asyncio.gather(deepseek_task())
    
if __name__ == "__main__":
    asyncio.run(main())