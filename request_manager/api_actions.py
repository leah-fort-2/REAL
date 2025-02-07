# Configure logging
import logging
import asyncio
from aiohttp import ClientTimeout
from aiohttp import ClientTimeout, ClientError

# Configure logging
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
    TIMEOUT_SECS = 144 # Set timeout
    timeout = ClientTimeout(total=TIMEOUT_SECS)
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            async with session.post(API_URL, json=request_body, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                if attempt < max_attempts: 
                    logger.warning(f"API request failed with status {response.status}. Response: {await response.text()}. Attempt {attempt + 1} of {max_attempts}")
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                else:
                    logger.error(f"API request failed with status {response.status}. Response: {await response.text()}")
                    return None
                        
        except asyncio.TimeoutError:
            if attempt < max_attempts:
                logger.warning(f"API request timed out after {TIMEOUT_SECS} seconds. Attempt {attempt + 1} of {max_attempts}")
                await asyncio.sleep(1)  # Wait for 1 second before retrying
            else:
                logger.error(f"API request timed out after {max_attempts} attempts")
                return None
        except ClientError as e:
            if attempt < max_attempts:
                logger.warning(f"API request error: {str(e)}. Attempt {attempt + 1} of {max_attempts}")
            else:
                logger.error(f"API request failed after {max_attempts} attempts")
            return None
        except Exception as e:
            logger.error(f"An error occurred during the API request: {str(e)}")
            if attempt < max_attempts:
                logger.warning(f"API request failed with status {response.status}. Response: {await response.text()}. Attempt {attempt + 1} of {max_attempts}")
            else:
                logger.error(f"API request failed after {max_attempts} attempts")
                return None
    
        # Wait before retrying
        if attempt < max_attempts:
            await asyncio.sleep(1)
    
    return None  # This line should never be reached, but it's here for completeness
    
def make_request_body(request_str, **request_params):
    """
    Construct the request body.
    
    :params request_str:  a single request string sent to API
    :params request_params: Request parameters in body e.g. temperature
    """
    MODEL = request_params["model"]
    params = {"model": MODEL}
    float_params = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
    int_params = ["max_tokens"]
    params.update({key: float(request_params[key]) for key in float_params if key in request_params})
    params.update({key: int(request_params[key]) for key in int_params if key in request_params})
    
    prefix = request_params.get("prompt_prefix", "")
    suffix = request_params.get("prompt_suffix", "")
    system_prompt = request_params.get("system_prompt", "")
    messages=[]
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{prefix}{request_str}{suffix}"})
    request = {"messages": messages}
    for key, value in params.items():
        request[key] = value
    
    return request

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
        return OAI_response['choices'][0]['message']['content']
    except (TypeError, KeyError, IndexError):
        logger.error(f"Failed to extract content from API response: {OAI_response}")
        return ''
    
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
