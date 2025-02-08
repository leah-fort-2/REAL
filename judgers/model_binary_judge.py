from dotenv import load_dotenv
import os
import asyncio
from request_manager.request_manager import single_request
from text_preprocessors import model_binary_scoring_cot_preprocessor, model_binary_scoring_preprocessor
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_judge_prompt():
    return "You are an high-level exam judge. You will be provided with a response text, a question text and a list of reference answer candidates, all of which are regarded as acceptable. Do not output anything other than a numeric score.\nScoring standards:\n- Either 1 (=correct) or 0 (=not correct)\n- To score as 1, the response must\n  > unambiguously bear correct answer(s), meanwhile\n  > does not include incorrect information.\n- Partial incorrectness = 0 (=incorrect). No mid value.\n- Repetition don't count as incorrect. Response truncation (last line cuts off) is safely ignored. Non-essential extra details in response text doesn't affect rating.\n- Unintelligible response = 0 (=incorrect)."

scoring_parameters = {"base_url": None,
                            "api_key": None,
                            "model": None,
                            "temperature": 0,
                            # Using temperature, top_p is omitted
                            "max_tokens": 1024,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "system_prompt": make_judge_prompt(),
                            "prompt_prefix": "",
                            "prompt_suffix": ""}
                                       
async def model_scoring(response:str, answer: str, context: str):
    # Yeah, validation first.
    validate_scoring_model_setting()
    validate_scoring_api_base_url()
    validate_scoring_api_key()
    
    SCORING_MODEL = os.getenv("SCORING_MODEL")
    SCORING_API_BASE_URL = os.getenv("SCORING_API_BASE_URL")
    SCORING_API_KEY = os.getenv("SCORING_API_KEY")
    if not SCORING_API_BASE_URL:
        logging.error(f"SCORING_API_BASE_URL is {SCORING_API_BASE_URL}. Configure in .env file first before using model scoring.")
    if not SCORING_API_KEY:
        logging.error(f"SCORING_API_KEY is {SCORING_API_KEY}. Configure in .env file first before using model scoring.")
    if not SCORING_MODEL:
        logging.error(f"SCORING_MODEL is {SCORING_MODEL}. Configure in .env file first before using model scoring.")
    
    scoring_parameters.update({
        "model": os.getenv("SCORING_MODEL"),
        "api_url": os.getenv("SCORING_API_BASE_URL")+"/chat/completions",
        "api_key": os.getenv("SCORING_API_KEY")
        })
    
    scoring_query = make_scoring_query(response, answer, context)
    
    scoring_result = await single_request(scoring_query, scoring_parameters)
    # Reminder: request_manager module is None safe. If the api request failed, a FALLBACK_ERR_MSG is returned.
    
    # Parse the binary score. Choose to use either cot or non-cot preprocessor.
    # When a numeric score can't be parsed, an empty string is returned. 
    score = model_binary_scoring_preprocessor(scoring_result)
    
    # We are using binary scoring, so int instead of float.
    return score if score == "" else int(score)
    
    
def validate_scoring_model_setting():
    env_scoring_model = os.getenv("SCORING_MODEL")
    if env_scoring_model == None:
        raise ValueError("To use model scoring, SCORING_MODEL must be configured in .env file. This could be because some workflow used MODEL_SCORING judger unintentionally during score judging.")
    
def validate_scoring_api_base_url():
    env_scoring_api_base_url = os.getenv("SCORING_API_BASE_URL")
    if env_scoring_api_base_url == None:
        raise ValueError("To use model scoring, SCORING_API_BASE_URL must be configured in .env file. This could be because some workflow used MODEL_SCORING judger unintentionally during score judging.")
    
def validate_scoring_api_key():
    env_scoring_api_key = os.getenv("SCORING_API_KEY")
    if env_scoring_api_key == None:
        raise ValueError("To use model scoring, SCORING_API_KEY must be configured in .env file. This could be because some workflow used MODEL_SCORING judger unintentionally during score judging.")
    
def make_scoring_query(response, answer, context):
    return f"""Response: `{response}`
Question: `{context}`
Answer: `{answer}`
"""