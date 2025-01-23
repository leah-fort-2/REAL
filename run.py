from worker import RequestParams, Worker
from dataset_models import QuerySet
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

async def main():
    # Example usage with deepseek. But you can use any api provider:    

    # Step 1: Instantiate a QuerySet object
    
    # Only csv and xlsx files supported
    query_set=QuerySet("loc_pt.xlsx")

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
    
    solo_system_prompt="You are a professional localization engineer and your language pair is Simplified Chinese => Portuguese (Brazil)."
       
    
    solo_params = RequestParams(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-reasoner",
        system_prompt=solo_system_prompt
    )
    
    # Create a worker (QuerySet-> ResponseSet)
    deepseek_solo_worker = Worker(solo_params)
    # Step 3: Organize tasks
    
    async def solo_task():
        result = await deepseek_solo_worker(query_set).invoke()
        result.store_to("solo_results_pt.xlsx")
       
    # Step 4: Hit and run!
    await asyncio.gather(solo_task())

if __name__ == "__main__":
    asyncio.run(main())