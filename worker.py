from batch_request import process_requests
from csv_manager import store_to_csv, read_from_csv
from xlsx_manager import store_to_excel, read_from_excel

import os
from dotenv import load_dotenv

load_dotenv()

# Read fallback parameters from .env
DEFAULT_BASE_URL = os.getenv("BASE_URL")
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE"))
DEFAULT_TOP_P = float(os.getenv("TOP_P"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
DEFAULT_FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY"))
DEFAULT_PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY"))

class RequestParams:
    def __init__(self, 
                 base_url=DEFAULT_BASE_URL, 
                 api_key=DEFAULT_API_KEY, 
                 model=DEFAULT_MODEL,
                 temperature=DEFAULT_TEMPERATURE,
                 top_p=DEFAULT_TOP_P,
                 max_tokens=DEFAULT_MAX_TOKENS,
                 frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
                 presence_penalty=DEFAULT_PRESENCE_PENALTY
                 ):
        self.api_url = base_url + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
    def get_params(self):
        return {
            "api_url": self.api_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

async def work(request_params: RequestParams, input_path="queries.csv", query_key="query", output_path="responses.csv"):
    if input_path.endswith('.csv'):
        request_list = read_from_csv(input_path, query_key)
    elif input_path.endswith('.xlsx'):
        request_list = read_from_excel(input_path, query_key)
    else:
        raise ValueError(f"Reading from unsupported file format: \"{input_path}\". Please use CSV or XLSX.")    
    # request_list = ["hi", "how are you? "]
    
    if not output_path.endswith('.csv') and not output_path.endswith('.xlsx'):
        raise ValueError(f"Storing to unsupported file format: \"{output_path}\". Please use CSV or XLSX.")
    
    params = request_params.get_params()
    response_texts = await process_requests(request_list, **params)
    
    responses = []
    
    for request, response_text in zip(request_list, response_texts):
        responses.append({"request": request, "response": response_text})

    if output_path.endswith('.csv'):
        store_to_csv(responses, output_path)
    else:
        # output_path ends with '.xlsx'
        store_to_excel(responses, output_path)