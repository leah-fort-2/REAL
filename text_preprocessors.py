"""
Response/answer preprocessors for the score judging feature `judge` in ResponseSet module.

Processors are wrappers of function pipelines. Each pipeline takes a string (in this case the response text) and a series of string manipulation methods.

When empty string is encountered at any stage of the pipeline, the workflow is halted with warning, and an empty string is returned immediately.
"""

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THINK_FAILED_MSG = "Thinking process failed."

def as_is(response: str):
    """
    No preprocessing, no validation, just "as-is".
    """
    return response

def mcq_preprocessor(response: str):
    """
    Simple mcq preprocessor that uses the first letter then uppercase it
    
    e.g. ` A. answer` => `A`
    """
    return preprocess_pipeline(response, 
                               strip_then_use_first_letter_then_uppercase)

def mcq_cot_preprocessor(response: str):
    """
    - :Incomplete cot: THINK_FAILED_MSG (`Thinking process failed.`)
    - :Regular: The first letter uppercased `A. answer` => `A`
    - :Valid cot but Empty conclusion: ""
    """
    def catch_bad_cot_or_fallback(s: str):
        if s == THINK_FAILED_MSG:
            # remove_think_tags will return "" when cot process is not completed or the model outputs nothing as the result
            return THINK_FAILED_MSG
        return strip_then_use_first_letter_then_uppercase(s)
    
    return preprocess_pipeline(response, 
                               remove_think_tags, 
                               catch_bad_cot_or_fallback)
    
def mcq_cot_preprocessor_for_bad_if(response:str):
    """
    - :Incomplete cot: THINK_FAILED_MSG ("Thinking process failed.")
    
    - :Regular: The last alphabetical character uppercased `The correct answer is: A.` => `A`
    
    - :Last letter is not among A/B/C/D: Try search `/Answer: [A-Da-d]/`: `Answer: D\\nThe correct answer` => `D`
    
    - :No match: ""
    
    - :Valid cot but Empty conclusion: ""
    """
    
    def catch_bad_cot_then_strip_with_fallback(s: str):
        if s == THINK_FAILED_MSG:
            # remove_think_tags will return "" when cot process is not completed or the model outputs nothing as the result
            return "Thinking process failed."
        
        state = strip_whitespace_then_asterisk_then_last_letter_then_uppercase(s)
        # e.g.  "Answer: A" -> "A"
        if state not in ["A", "B", "C", "D"]:
            state = search_for_answer(s)
            # e.g. "Answer: A\nThe above is the answer." => "A"
        return state
    return preprocess_pipeline(response, 
                               remove_think_tags,
                               catch_bad_cot_then_strip_with_fallback)
    
def model_binary_scoring_cot_preprocessor(response: str) -> str:
    """
    Parse a binary score from a scoring cot model message into a string. e.g. "1" "0"
    
    On failure to parse, return "".
    """
    # Parse a scoring model message into a binary int score.
    state = remove_think_tags(response)
    if state == THINK_FAILED_MSG:
        return ""
    state = strip_whitespace_then_asterisk_then_last_number(state)
    if state not in ["1", "0"]:
        return ""
    return state

def model_binary_scoring_preprocessor(response: str) -> str:
    """
    Binary score parsing for non-cot model.
    
    On failure to parse, return "".
    """
    first_char = strip_then_use_first_character(response)
    if first_char.isdigit():
        return first_char
    last_char = strip_then_use_last_character(response)
    if last_char.isdigit():
        return last_char
    return ""    

def strip_then_use_first_letter_then_uppercase(s: str):
    # Remove non-alphabetical characters
    state=re.sub("[^A-Za-z]", "", s)
    try:
        return state.strip()[0].upper()
    except IndexError:
        # e.g. " " => "" => no [0] can be retrieved
        return ""

def strip_then_use_first_character(s: str):
    state=re.sub("\\s", "", s)
    try:
        return state.strip()[0]
    except IndexError:
        # e.g. " " => "" => no [0] can be retrieved
        return ""

def strip_then_use_last_character(s: str):
    state=re.sub("\\s", "", s)
    try:
        return state.strip()[-1]
    except IndexError:
        # e.g. " " => "" => no [0] can be retrieved
        return ""
    
def strip_whitespace_then_asterisk_then_last_letter_then_uppercase(s: str):
    # Remove non-alphabetical characters
    state=re.sub("[^A-Za-z]", "", s)
    try:
        return state.strip().replace("*", "").rstrip()[-1].upper()
    except IndexError:
        # e.g. " " => "" => no [-1] can be retrieved
        return ""

def strip_whitespace_then_asterisk_then_last_number(s: str):
    # Remove non-numeric characters
    state=re.sub("[^\\d]", "", s)
    try:
        last_char = state.strip().replace("*", "").rstrip()[-1]
        return last_char if last_char.isdigit() else ""
    except IndexError:
        # e.g. " " => "" => no [-1] can be retrieved
        return ""
    
def search_for_answer(s: str):
    pattern = f"[Aa]nswer:.*?([A-Da-d])"
    match = re.search(pattern, s, flags=re.DOTALL)
    if match == None:
        logger.error(f"Failed to parse answer from response: {s[:50]}")
        return ""
    return match.group(1).upper()
    
def remove_think_tags(s: str):
    removed = re.sub("<[Tt]hink>.*</[Tt]hink>", "", s, flags=re.DOTALL)
    
    # If no cot is parsed at all, return a fallback message string.
    if len(removed) == len(s):
        logger.warning(f"Encountered malformed cot section. The cot is likely incomplete. Will fall back to think failed msg.")
        return THINK_FAILED_MSG
    return removed
    
def preprocess_pipeline(str_to_preprocess: str, *preprocessors):
    """
    Execute a series of preprocessors with empty string detection.
    """
    state = str_to_preprocess
    for func in preprocessors:
        if state != "":
            state = func(state)
            continue
        logger.warning("Encountered empty response. It's likely the model returned an empty response!")
        return ""
    return state


if __name__ == "__main__":
    msg = """
    <think>
Okay, 
1
    """
    print(model_binary_scoring_cot_preprocessor(msg))
    
    msg2="1"
    
    print(model_binary_scoring_preprocessor(msg2))
    print(model_binary_scoring_preprocessor(msg))