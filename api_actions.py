import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("BASE_URL")+"/chat/completions"
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = float(os.getenv("TEMPERATURE"))
TOP_P = float(os.getenv("TOP_P"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))

# Configure logging
async def do_request_on(session, request_text):    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    request_body = make_request_body(request_text)
    async with session.post(API_URL, json=request_body, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            return None

def make_request_body(request_str):
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": request_str
            }
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS
    }
    
def extract_content(OAI_response):
    try:
        return OAI_response['choices'][0]['message']['content']
    except (TypeError, KeyError, IndexError):
        return ''