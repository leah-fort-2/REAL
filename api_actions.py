# Configure logging
async def do_request_on(session, request_text, **request_params):
    API_URL = request_params["api_url"]
    API_KEY = request_params["api_key"]
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
        
    MODEL = request_params["model"]
    params = {"model": MODEL}
    float_params = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
    int_params = ["max_tokens"]
    params.update({key: float(request_params[key]) for key in float_params if key in request_params})
    params.update({key: int(request_params[key]) for key in int_params if key in request_params})
    
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