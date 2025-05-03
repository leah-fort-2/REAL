# Configure logging
import logging
import asyncio
from typing import Tuple
from aiohttp import ClientTimeout
from aiohttp import ClientTimeout, ClientError
import dotenv
import os

dotenv.load_dotenv()

TIMEOUT = int(os.getenv("TIMEOUT", 144))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def do_request_on(session, request_text, **request_params):
    """
    Send post request to OAI api in async fashion. An aiohttp session is managed externally.
    
    :params session:  an aiohttp.ClientSession object
    :params request_text:  a single request string sent to API
    :param request_params: Request parameters in body e.g. temperature
    :return: Coroutine -> response dict | None
    """
    API_URL = request_params["api_url"]
    API_KEY = request_params["api_key"]
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    request_body = make_request_body(request_text, **request_params)
    timeout = ClientTimeout(total=TIMEOUT)


    for attempt in range(MAX_ATTEMPTS):
        try:
            async with session.post(API_URL, json=request_body, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    body = await response.json()
                    if body != None:
                        return body
                    
                    # Server-side issue, returns empty body with 200 code
                    if attempt < MAX_ATTEMPTS:
                        logger.warning(f"API returned 200 but with empty response body. Response: {await response.text()}.  Attempt {attempt + 1} of {MAX_ATTEMPTS}")
                    else:
                        logger.error(f"API returned 200 but with empty response body. Response: {await response.text()}")
                        return None
                
                # Non 200 code
                else:
                    if attempt < MAX_ATTEMPTS: 
                        logger.warning(f"API request failed with status {response.status}. Response: {await response.text()}. Attempt {attempt + 1} of {MAX_ATTEMPTS}")
                        await asyncio.sleep(1)  # Wait for 1 second before retrying
                    else:
                        logger.error(f"API request failed with status {response.status}. Response: {await response.text()}")
                        return None
        
        # Request timeout
        except asyncio.TimeoutError:
            if attempt < MAX_ATTEMPTS:
                logger.warning(f"API request timed out after {TIMEOUT} seconds. Attempt {attempt + 1} of {MAX_ATTEMPTS}")
                await asyncio.sleep(1)  # Wait for 1 second before retrying
            else:
                logger.error(f"API request timed out after {MAX_ATTEMPTS} attempts")
                return None
        
        # Client Error
        except ClientError as e:
            if attempt < MAX_ATTEMPTS:
                logger.warning(f"API request error: {str(e)}. Attempt {attempt + 1} of {MAX_ATTEMPTS}")
            else:
                logger.error(f"API request failed after {MAX_ATTEMPTS} attempts")
            return None
        
        # Unknown error
        except Exception as e:
            logger.error(f"An error occurred during the API request: {str(e)}")
            if attempt < MAX_ATTEMPTS:
                logger.warning(f"API request failed with status {response.status}. Response: {await response.text()}. Attempt {attempt + 1} of {MAX_ATTEMPTS}")
            else:
                logger.error(f"API request failed after {MAX_ATTEMPTS} attempts")
                return None
    
        # Wait before retrying
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(1)
    
    return None  # This line should never be reached, but it's here for completeness
    
def make_request_body(request_str, **request_params):
    """
    Construct the request body.
    
    :params request_str:  a single request string sent to API
    :params request_params: Request parameters in body e.g. temperature
    """
    params = {}
    try:
        MODEL = request_params.pop("model")
    except KeyError:
        raise KeyError(f"No model information is found in request params. Params content: {request_params}")
    params.update({"model": MODEL})
    # By this moment, model should no more exist in request_params
    
    float_params = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
    int_params = ["max_tokens", "top_k"]

    key_updated = {}
    for key in float_params:
        if key in request_params:
            key_updated[key] = float(request_params.pop(key))
    for key in int_params:
        if key in request_params:
            key_updated[key] = int(request_params.pop(key))
    params.update(key_updated)
    # By this moment, float_params and int_params should no more exist in request_params
    
    if ("top_p" in params or "top_k" in params) and "temperature" in params and params["temperature"]==0:
        logger.info(f"Detect temperature {params['temperature']} and top_p {params.get('top_p', None)} / top_k {params.get('top_k', None)}. In case of 0 temperature, top_p and top_k will be ignored. To use nucleus sampling, set temperature as a non-zero value.")
        params.pop("top_p", None)
        params.pop("top_k", None)
    
    prefix = request_params.pop("prompt_prefix", "")
    suffix = request_params.pop("prompt_suffix", "")
    system_prompt = request_params.pop("system_prompt", "")
    messages=[]
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{prefix}{request_str}{suffix}"})
    # By this moment, prefix, suffix and system prompt should no more exist in request_params

    request_params.pop("base_url")
    request_params.pop("api_url")
    request_params.pop("api_key")
    # Shouldn't be present in actual request

    request = {"messages": messages}
    for key, value in params.items():
        request[key] = value
        
    # Add remaining request_params
    for key, value in request_params.items():
        request[key] = value
    # Patch in 2.1.2: Forbid stream output
    request.update({"stream": False})
    
    return request

NONE_CONTENT_ERROR_MSG = "Received None content."

def extract_content(OAI_response):
    """
    :params OAI_response: response dict from an OpenAI compatible API. Should be structured as:
    ```{'choices': [
            {'message': 
                {'content': '...'}
            }]
        }
    """
    try:
        msg = OAI_response['choices'][0]['message']['content']
        if msg is None:
            raise ValueError
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract content from API response: {OAI_response}")
        msg = ''
    except ValueError:
        msg = NONE_CONTENT_ERROR_MSG
    return msg
    
def extract_usage(OAI_response) -> Tuple[str | None, int, int]:
    """
    :params OAI_response: response dict from an OpenAI compatible API. Should be structured as:
    ```{'choices': [
            {'message': 
                {'content': str}
            }],
        'usage': 
            {
                'prompt_tokens': int,
                'completion_tokens': int
            }
        }
    ```
    :returns: Tuple[message, prompt_tokens, completion_tokens]
    """
    try:
        msg: str | None = OAI_response['choices'][0]['message']['content']
        if msg is None:
            raise ValueError
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract content from API response: {OAI_response}")
        msg = ''
    except ValueError:
        msg = NONE_CONTENT_ERROR_MSG
    try:
        prompt_tokens = OAI_response['usage']['prompt_tokens']
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract prompt token usage from API response: {OAI_response}")
        prompt_tokens = 0
    try:
        completion_tokens = OAI_response['usage']['completion_tokens']
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract completion token usage from API response: {OAI_response}")
        completion_tokens = 0

    return (msg, prompt_tokens, completion_tokens)

def extract_content_with_reasoning(o1_response):
    """
    :params o1_response: response dict from an OAI API with `reasoning_content` output. Should be structured as:
    ```{'choices': [
            {'message': 
                {
                    'reasoning_content': '...',
                    'content': '...'}
            }]
        }
    """ 
    try:
        reasoning_content = o1_response['choices'][0]['message']['reasoning_content']
        reasoning_content = "\n".join([f"> {line}" for line in reasoning_content.split("\n")])
    except Exception as e:
        logger.error(f"Failed to extract reasoning content: {str(e)}")
        reasoning_content = ""
    # Extract main message string
    try:
        content = o1_response['choices'][0]['message']['content']
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract content from API response: {o1_response}")
        return ''
    
    return f"{reasoning_content}\n{content}"