from batch_request import process_requests
from dataset_models import ResponseSet

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
    """
        Request parameters for OpenAI compatible APIs. If not specified, use global settings in .env file.
        - api_url
        - api_key
        - model
        - temperature
        - max_tokens
        - top_p
        - frequency_penalty
        - presence_penalty        
    """
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
        
class Worker:
    def __init__(self, request_params: RequestParams):
        self.request_params = request_params
        
    def get_params(self):
        return self.request_params.get_params()
    
    async def invoke(self, query_set, query_key="query"):
        queries = query_set.get_queries()
        
        # Acquire a query string list
        try:
            # Case 1: query_set is initialized from a file...
            query_list = [query[query_key] for query in queries]
        except TypeError:
            # Case 2: or a list of query strings
            query_list = queries
        
        # Launch requests
        params = self.get_params()
        response_list = await process_requests(query_list, **params)
        
        # Post works
        try:
            # Case 1: update the original query_list
            for query, response in zip(queries, response_list):
                query.update({"response": response})
        except AttributeError:
            # Case 2: str has no update attribute
            # Construct a new dict for each query and response pair
            queries = [
                {
                    "query": query,
                    "response": response
                } for query, response in zip(queries, response_list)]
        return ResponseSet(queries)