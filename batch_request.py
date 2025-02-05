import os
from dotenv import load_dotenv

load_dotenv()

MAX_CONCURRENT_REQUESTS = int(os.getenv('BATCH_SIZE', "10"))

import asyncio
import aiohttp
from api_actions import do_request_on, extract_content, extract_content_with_reasoning
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_requests(request_list: list[str], **request_params):
    # Returns a list of response strings
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_request(session, semaphore, request, **request_params) for request in request_list]
        responses = await asyncio.gather(*tasks)
    return responses

async def process_single_request(session, semaphore, request, **request_params):
    async with semaphore:
        try:
            if request == "":
                logger.warning(f"I found an empty query, but will proceed requesting with it.")
            response = await do_request_on(session, request, **request_params)
            if response:
                # Deepseek reasoner now supports outputting its reflection process. You can use extract_content to get the response content only.
                # content = extract_content_with_reasoning(response)
                content = extract_content(response)
                logger.info(f"Processed request: {request[:50]}...")
                return content
            else:
                logger.error(f"Failed to process request: {request[:50]}... Response: {response}")
                return None
        except Exception as e:
            logger.error(f"Error processing request: {request[:50]}... Error: {str(e)}")
            return None