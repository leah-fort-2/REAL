from request_manager.request_manager import process_batch
from dataset_models import QuerySet, ResponseSet
from typing import Union, Dict, Any

import os
from dotenv import load_dotenv

load_dotenv()

# Read fallback parameters from .env
DEFAULT_BASE_URL = os.getenv("BASE_URL")
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("MODEL")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE")) if os.getenv("TEMPERATURE") else None
DEFAULT_TOP_P = float(os.getenv("TOP_P"))  if os.getenv("TOP_P") else None
DEFAULT_TOP_K = int(os.getenv("TOP_K")) if os.getenv("TOP_K") else None
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS"))  if os.getenv("MAX_TOKENS") else None
DEFAULT_FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY"))  if os.getenv("FREQUENCY_PENALTY") else None
DEFAULT_PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY"))  if os.getenv("PRESENCE_PENALTY") else None
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY"))  if os.getenv("REPETITION_PENALTY") else None
# When system prompt is left blank or None, no system message will be added to the session.
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DEFAULT_PROMPT_PREFIX = os.getenv("PROMPT_PREFIX")
DEFAULT_PROMPT_SUFFIX = os.getenv("PROMPT_SUFFIX")

class RequestParams:
    """
    A class to manage and validate request parameters for API calls.

    This class handles the initialization, validation, and storage of various
    parameters used in API requests. It ensures type safety for predefined
    attributes and allows for dynamic addition of custom attributes.
    Attributes:
        base_url (str): The base URL for the API.
        api_key (str): The API key for authentication.
        model (str): The model to be used for the request.
        temperature (Union[float, int]): The sampling temperature.
        top_p (Union[float, int]): The nucleus sampling parameter.
        max_tokens (int): The maximum number of tokens to generate.
        frequency_penalty (Union[float, int]): The frequency penalty parameter.
        presence_penalty (Union[float, int]): The presence penalty parameter.
        api_url (str): (Generated automatically) The full API URL, constructed from base_url.

    The class also allows for addition of custom attributes not listed above.
    """
    _attribute_types = {
        'base_url': str,
        'api_key': str,
        'model': str,
        'temperature': Union[float, int],
        'top_p': Union[float, int],
        'top_k': int,
        'max_tokens': int,
        'frequency_penalty': Union[float, int],
        'presence_penalty': Union[float, int],
        "repetition_penalty": Union[float, int],
        'api_url': str
    }

    def __init__(self, **kwargs):
        """
        Initialize the RequestParams instance.

        :param kwargs: Keyword arguments for setting initial attribute values.
                       If not provided, default values from environment variables are used.
        """
        self.base_url = self._validate_type('base_url', kwargs.get('base_url', DEFAULT_BASE_URL))
        self.api_key = self._validate_type('api_key', kwargs.get('api_key', DEFAULT_API_KEY))
        self.model = self._validate_type('model', kwargs.get('model', DEFAULT_MODEL))
        self.temperature = self._validate_type('temperature', kwargs.get('temperature', DEFAULT_TEMPERATURE))
        self.top_p = self._validate_type('top_p', kwargs.get('top_p', DEFAULT_TOP_P))
        self.top_k = self._validate_type('top_k', kwargs.get('top_k', DEFAULT_TOP_K))
        self.max_tokens = self._validate_type('max_tokens', kwargs.get('max_tokens', DEFAULT_MAX_TOKENS))
        self.frequency_penalty = self._validate_type('frequency_penalty', kwargs.get('frequency_penalty', DEFAULT_FREQUENCY_PENALTY))
        self.presence_penalty = self._validate_type('presence_penalty', kwargs.get('presence_penalty', DEFAULT_PRESENCE_PENALTY))
        self.repetition_penalty = self._validate_type('repetition_penalty', kwargs.get('repetition_penalty', DEFAULT_REPETITION_PENALTY))
        
        self.api_url = self._validate_type('api_url', f"{self.base_url}/chat/completions")
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def _validate_type(self, name: str, value: Any) -> Any:
        """
        Validate the type of a given value against the expected type.

        :param name: The name of the attribute being validated.
        :param value: The value to validate.
        :return: The validated value.
        :raises TypeError: If the value does not match the expected type.
        """
        if value is not None:
            expected_type = self._attribute_types.get(name)
            if expected_type and not isinstance(value, expected_type):
                    raise TypeError(f"{name} must be of type {expected_type}")
        return value

    def __setattr__(self, name: str, value: Any):
        """
        Set an attribute with type validation.

        :param name: The name of the attribute to set.
        :param value: The value to set the attribute to.
        :raises TypeError: If the value does not match the expected type for predefined attributes.
        """
        self._validate_type(name, value)
        super().__setattr__(name, value)

    def get_params(self) -> Dict[str, Any]:
        """
        Get a dictionary of all non-None parameters.

        :return: A dictionary containing all attributes with non-None values.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __getattr__(self, name: str) -> Any:
        """
        Custom attribute access method to raise an AttributeError for non-existent attributes.

        :param name: The name of the attribute to get.
        :raises AttributeError: If the attribute does not exist.
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str):
        """
        Delete an attribute if it exists.

        :param name: The name of the attribute to delete.
        :raises AttributeError: If the attribute does not exist.
        """
        if hasattr(self, name):
            super().__delattr__(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class Worker:
    class Job:
        """
        You are not supposed to directly create a Job instance. Use Worker instance to create one.
        """
        def __init__(self, worker, query_set: QuerySet, query_key: str, response_key: str):
            self.worker = worker
            self.query_set = query_set
            self.query_key = query_key
            self.response_key = response_key
        
        def __len__(self):
            return len(self.query_set)
        
        def get_query_set(self):
            """
            :return Queryset query_set: The very QuerySet object to start this job.
            
            """
            return self.query_set
        
        def get_worker(self):
            """
            :return Worker worker: The very Worker object to start this job.

            """
            return self.worker
        
        async def invoke(self):
            """
            Start this job.
            """
            worker = self.worker
            query_key = self.query_key
            response_key = self.response_key
            # queries: list({query_key: <query_str>})
            queries = self.query_set.get_queries()
            
            # Acquire a query string list
            query_string_list = [query[query_key] for query in queries]
            
            # Launch requests
            params: dict[str, Any] = worker.get_params()
            response_list = await process_batch(query_string_list, params)
            
            # Post works
            for query, response in zip(queries, response_list):
                query.update({response_key: response})
            return ResponseSet(queries, query_key=query_key, response_key=response_key)
        
    def __init__(self, request_params: RequestParams):
        self.request_params = request_params
        
    def __call__(self, query_set: QuerySet, query_key="query", response_key="response"):
        """
        Create a `Job` instance with specified query_key and response_key. Call `invoke()` to start when needed.
        
        :params QuerySet query_set: The query set to evaluate with.
        :params str query_key: Field to evaluate in the query set. Default to "query".
        :params str response_key: Field to store response to. Default to "response".
        :return Job: A job instance.

        """
        return self.Job(self, query_set, query_key, response_key)
        
    def get_params(self):
        """
        :return dict[str, Any]: The parameters to initialize this worker. Modifying this dict does not affect the worker.
 
        """
        return self.request_params.get_params()