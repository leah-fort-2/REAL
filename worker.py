from batch_request import process_requests
from dataset_models import QuerySet, ResponseSet

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
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DEFAULT_PROMPT_PREFIX = os.getenv("PROMPT_PREFIX")
DEFAULT_PROMPT_SUFFIX = os.getenv("PROMPT_SUFFIX")

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
                 presence_penalty=DEFAULT_PRESENCE_PENALTY,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 prompt_prefix=DEFAULT_PROMPT_PREFIX,
                 prompt_suffix=DEFAULT_PROMPT_SUFFIX
                 ):
        self.api_url = base_url + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        
    def get_params(self):
        return {
            "api_url": self.api_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "system_prompt": self.system_prompt,
            "prompt_prefix": self.prompt_prefix,
            "prompt_suffix": self.prompt_suffix
        }
        
class Worker:
    class Job:
        """
        You are not supposed to directly create a Job instance. Use Worker instance to create one.
        """
        def __init__(self, worker, query_set_or_template, query_key="query"):
            self.worker = worker
            self.query_set_or_template = query_set_or_template
            self.query_key = query_key
        
        def __len__(self):
            return len(self.query_set_or_template)
        
        def get_query_set(self):
            return self.query_set_or_template
        
        def get_worker(self):
            return self.worker
        
        async def invoke(self):
            worker = self.worker
            query_key = self.query_key
            # queries: list({query_key: <query_str>})
            queries = self.query_set_or_template.get_queries()
            
            # Acquire a query string list
            query_list = [query[query_key] for query in queries]
            
            # Launch requests
            params = worker.get_params()
            response_list = await process_requests(query_list, **params)
            
            # Post works
            for query, response in zip(queries, response_list):
                query.update({"response": response})
            return ResponseSet(queries)
        
    def __init__(self, request_params: RequestParams):
        self.request_params = request_params
        
    def __call__(self, query_set: QuerySet, query_key="query"):
        return self.Job(self, query_set, query_key)
        
    def get_params(self):
        return self.request_params.get_params()