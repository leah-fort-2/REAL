from worker import RequestParams, work
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Load custom BASE_URL and API_KEY from .env file

DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

async def main():
    # Example usage with deepseek. But you can use any api provider:
    
    deepseek_params = RequestParams(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-chat"
    )
    
    # Create tasks
    # Currently only support xlsx/csv input and output format. Specifying query key is possible and optional.
    deepseek_task = work(deepseek_params, input_path="queries.xlsx", output_path="deepseek_response.xlsx")
    
    # Run all tasks concurrently and wait for them to complete
    await asyncio.gather(deepseek_task)
    
if __name__ == "__main__":
    asyncio.run(main())