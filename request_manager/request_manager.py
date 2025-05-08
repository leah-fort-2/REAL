import os
from typing import Tuple
from dotenv import load_dotenv
import asyncio
import aiohttp
from request_manager.api_actions import do_request_on, extract_content
import logging

load_dotenv()

MAX_CONCURRENT_REQUESTS = int(os.getenv('BATCH_SIZE', "5"))
FALLBACK_ERR_MSG = "Unknown error in processing request"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestResourceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RequestResourceManager, cls).__new__(cls)
            cls._instance.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) if MAX_CONCURRENT_REQUESTS > 0 else None
        return cls._instance

    def get_semaphore(self) -> asyncio.Semaphore:
        return self._instance.semaphore

async def process_batch(request_list: list[str], request_params: dict, enable_metrics=False) -> list[dict[str, str] | dict[str, int | str]]:
    """
    Process a list of request strings asynchronously, with a semaphore of size BATCH_SIZE set in .env file.
    
    :param request_list: A list of request strings
    :param request_params: Request parameters in body e.g. temperature
    :param bool enable_metrics: Default to False. Whether to include usage in results
    :return: a list of response strings, or error messages
    """
    semaphore = RequestResourceManager().get_semaphore()
    batch_total = len(request_list)
    async with aiohttp.ClientSession() as session:
        tasks = [_process_request(request, request_params, session, semaphore=semaphore, request_id=f"{i + 1}/{batch_total}", enable_metrics=enable_metrics) for i, request in enumerate(request_list)]
        responses = await asyncio.gather(*tasks)
    return responses

async def single_request(request: str, request_params: dict) -> dict[str, str] | dict[str, int | str]:
    """
    Process a single request independently. Concurrency is managed externally.
    
    :param request: A request string
    :param request_params: Request parameters e.g. temperature
    :return: a response string or error message
    """
    async with aiohttp.ClientSession() as session:
        return await _process_request(request, request_params, session)

async def _process_request(request, request_params, session, semaphore=None, request_id=None, enable_metrics=False) -> dict[str, str] | dict[str, int | str]:
    """
    Process a single request as part of a batch operation, where a session and a semaphore are managed externally. Returns a message content string.
    
    Delegate api request and content parsing to api_actions module. 
    
    :param request: The text of the request to process
    :param request_params: Additional parameters for the request
    :param session: An active aiohttp.ClientSession
    :param semaphore: An optional semaphore of size BATCH_SIZE, set in .env file. Leave it as None for single requests
    :param bool enable_metrics: Default to False. Whether to include usage in results
    :return: Processed content or error message
    """
    async def _do_request():
        FALLBACK = {"content": FALLBACK_ERR_MSG}
        if enable_metrics:
            FALLBACK.update({"prompt_tokens": 0, "completion_tokens": 0})

        try:
            if request == "":
                logger.warning(f"I found an empty query, but will proceed requesting with it.")
            response = await do_request_on(session, request, **request_params)
            if response:
                result = extract_content(response, enable_metrics)
                logger.info(f"Processed request{f" {request_id}" if request_id else ""}: {request[:50]}...")
                return result
            else:
                logger.error(f"The following request received a void response: {request[:50]}... Response: {response}")
                return FALLBACK
        except Exception as e:
            logger.error(f"Error processing request: {request[:50]}... Error: {str(e)}")
            return FALLBACK
        
    if semaphore:
        async with semaphore:
            return await _do_request()
    else:
        return await _do_request()