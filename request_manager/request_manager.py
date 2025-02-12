import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from request_manager.api_actions import do_request_on, extract_content, extract_content_with_reasoning
import logging

load_dotenv()

MAX_CONCURRENT_REQUESTS = int(os.getenv('BATCH_SIZE', "5"))
FALLBACK_ERR_MSG = "Unknown error in processing request"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_batch(request_list: list[str], request_params: dict):
    """
    Process a list of request strings asynchronously, with a semaphore of size BATCH_SIZE set in .env file.
    
    :param request_list: A list of request strings
    :param request_params: Request parameters in body e.g. temperature
    :return: a list of response strings, or error messages
    """
    semaphore = None if MAX_CONCURRENT_REQUESTS == 0 else asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    batch_total = len(request_list)
    async with aiohttp.ClientSession() as session:
        tasks = [_process_request(request, request_params, session, semaphore=semaphore, request_id=f"{i}/{batch_total}") for i, request in enumerate(request_list)]
        responses = await asyncio.gather(*tasks)
    return responses

async def single_request(request: str, request_params: dict):
    """
    Process a single request independently. Concurrency is managed externally.
    
    :param request: A request string
    :param request_params: Request parameters e.g. temperature
    :return: a response string or error message
    """
    async with aiohttp.ClientSession() as session:
        return await _process_request(request, request_params, session)

async def _process_request(request, request_params, session, semaphore=None, request_id=None):
    """
    Process a single request as part of a batch operation, where a session and a semaphore are managed externally. Returns a message content string.
    
    Delegate api request and content parsing to api_actions module. 
    
    :param request: The text of the request to process
    :param request_params: Additional parameters for the request
    :param session: An active aiohttp.ClientSession
    :param semaphore: An optional semaphore of size BATCH_SIZE, set in .env file. Leave it as None for single requests
    :return: Processed content or error message
    """
    async def _do_request():
        try:
            if request == "":
                logger.warning(f"I found an empty query, but will proceed requesting with it.")
            response = await do_request_on(session, request, **request_params)
            if response:
                # Deepseek reasoner now supports outputting its reflection process. You can use extract_content to get the response content only.
                # content = extract_content_with_reasoning(response)
                content = extract_content(response)
                logger.info(f"Processed request{f" {request_id}" if request_id else ""}: {request[:50]}...")
                return content
            else:
                logger.error(f"Failed to process request: {request[:50]}... Response: {response}")
                return FALLBACK_ERR_MSG
        except Exception as e:
            logger.error(f"Error processing request: {request[:50]}... Error: {str(e)}")
            return FALLBACK_ERR_MSG
        
    if semaphore:
        async with semaphore:
            return await _do_request()
    else:
        return await _do_request()