# Configure logging
async def do_request_on(session, request_text, **request_params):
    API_URL = request_params["api_url"]
    API_KEY = request_params["api_key"]
    MODEL = request_params["model"]
    TEMPERATURE = request_params["temperature"]
    TOP_P = request_params["top_p"]
    MAX_TOKENS = request_params["max_tokens"]
    FREQUENCY_PENALTY = request_params["frequency_penalty"]
    PRESENCE_PENALTY = request_params["presence_penalty"]
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # create request params object
    params = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY
    }
        
    request_body = make_request_body(request_text, **params)
    async with session.post(API_URL, json=request_body, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            return None

def make_request_body(request_str, **request_params):
    request = {
        "messages": [
            {
                "role": "user",
                "content": request_str
            }
        ]}
    for key, value in request_params.items():
        request[key] = value
    
    return request
    
def extract_content(OAI_response):
    try:
        return OAI_response['choices'][0]['message']['content']
    except (TypeError, KeyError, IndexError):
        return ''