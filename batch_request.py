import os
from dotenv import load_dotenv

load_dotenv()

MAX_CONCURRENT_REQUESTS = int(os.getenv('BATCH_SIZE'))

import asyncio
import aiohttp
from api_actions import do_request_on, extract_content
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_requests(request_list: list[str]):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_request(session, semaphore, request) for request in request_list]
        responses = await asyncio.gather(*tasks)
    return responses

async def process_single_request(session, semaphore, request):
    async with semaphore:
        try:
            response = await do_request_on(session, request)
            if response:
                content = extract_content(response)
                logger.info(f"Processed request: {request[:50]}...")
                return content
            else:
                logger.error(f"Failed to process request: {request[:50]}...")
                return None
        except Exception as e:
            logger.error(f"Error processing request: {request[:50]}... Error: {str(e)}")
            return None
