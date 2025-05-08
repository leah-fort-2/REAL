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
ANSWER_FAILED_MSG = "No valid answer field was found. Fall back to false."
from request_manager.api_actions import NONE_CONTENT_ERROR_MSG

def as_is(response: str):
    """
    No preprocessing, no validation, just "as-is".
    """
    return response

def mcq_search_preprocessor(response: str):
    """
    mcq preprocessor that works with \<answer\> \</answer\> tags.
    
    - :None response content: NONE_CONTENT_ERROR_MSG ('Received None content.')
    - :Malformed answer (No valid \<answer\> field): ANSWER_FAILED_MSG ("No valid answer field was found. Fall back to false.")
    """
    def _extract_answer(s: str):
        # Remove latex boxed statement e.g. `\boxed{A}`
        unboxed = s.replace("\\boxed", " ")

        pattern = f"<[Aa]nswer>([^\\w]*?)([A-Za-z]+).*</[Aa]nswer>"
        match = re.search(pattern, unboxed, flags=re.DOTALL)
        if match != None:
            # Only pick if the first letter of the group is independent. e.g. `Answer: A` but not `Answer: Shark` 
            state = pick_first_letter_if_independent(match.group(2))
            if state:
                return state.upper()
        return ANSWER_FAILED_MSG
    
    return preprocess_pipeline(response, _extract_answer)

def mcq_preprocessor(response: str):
    """
    Simple mcq preprocessor that uses the first letter if independent, then uppercase it.
    
    An independent letter has no adjacent alphanumeric characters. A well-formated MCQ response starts with and only contains a letter. Considering formatting, at least the first/last alphabetical character should be independent (`**A**` is possible).
    
    - :None response content: NONE_CONTENT_ERROR_MSG ('Received None content.')
    - :Unrecognizable independent letters: ""
    - :Others: ` A. answer` => `A`
    """
    
    def _handle_both_ends(s: str):
        first_independent_letter = pick_first_letter_if_independent(s)
        return first_independent_letter if first_independent_letter else pick_last_letter_if_independent(s)
    
    return preprocess_pipeline(response, 
                               _handle_both_ends,
                               lambda s: s.upper())

def mcq_cot_preprocessor(response: str):
    """
    Similar to a regular mcq preprocessor, but add cot truncation and bad cot handling.
    
    - :None response content: NONE_CONTENT_ERROR_MSG ('Received None content.')
    - :Incomplete cot: THINK_FAILED_MSG (`Thinking process failed.`)
    - :Valid cot but Empty conclusion: THINK_FAILED_MSG
    - :Otherwise: as in mcq_preprocessor
    """
    def _handle_both_ends(s: str):
        first_independent_letter = pick_first_letter_if_independent(s)
        return first_independent_letter if first_independent_letter else pick_last_letter_if_independent(s)
    
    return preprocess_pipeline(response, 
                               remove_think_tags, 
                               _handle_both_ends,
                               lambda s: s.upper())
    
def mcq_cot_preprocessor_for_bad_if(response:str):
    """
    Similar to mcq_cot_preprocessor, but cherrypick the last letter instead. Applies to models with bad instruction following (stubborn in giving analysis).
    
    - :None response content: NONE_CONTENT_ERROR_MSG ('Received None content.')
    - :Incomplete cot: THINK_FAILED_MSG ("Thinking process failed.")
    - :Valid cot but Empty conclusion: THINK_FAILED_MSG
    - :Multiple answers (among ABCD) were given: `A, B, C` => [removed]
    - :Regular: The last alphabetical character uppercased `The correct answer is: A.` => `A`
    - :No match: ""
    """
    
    def remove_multiple_choices(s: str):
        """
        Answer: A/B/C/D (X)
        """
        return re.sub("[ABCDabcd]\\W{0,2}[ABCDabcd](\\W{0,2}[ABCDabcd]){0,2}(\\W|$)", "", s)
    
    def _catch_bad_cot_pipeline(s: str):
        """
        These conditional makes the component methods not individually chainable to pipeline:
        
        - :ERROR MSG flags input: immediately return the flag
        
        - :letter selection with fallbacks:
        
        - search_for_answer decision upon no independent letter is found from both ends 

        """
        state = pick_last_letter_if_independent(s)
        # e.g.  "Answer: B" -> "B" 
        
        state = state if state else pick_first_letter_if_independent(s)
        # Try to catch if the first letter is the answer
        
        return state.upper() if state else search_for_answer(s)
        # e.g. "Answer: B\nThe above is the answer." => "B"

    return preprocess_pipeline(response, 
                               remove_multiple_choices,
                               remove_think_tags,
                               _catch_bad_cot_pipeline)

def clean_humaneval_preprocessor(response: str) -> str:
    """
    Deals with code completion with surrounding code block syntax(```python ```)
    """
    return response.strip("`").lstrip("python")

def clean_humaneval_cot_preprocessor(response: str) -> str:
    """
    Equivalent of clean_humaneval_preprocessor, but for models supporting cot, as deepseek r1 distill models.
    """
    def _catch_bad_cot_and_clean(s: str) -> str:
        return s.strip("`").lstrip("python")
    
    return preprocess_pipeline(response,
                               remove_think_tags,
                               _catch_bad_cot_and_clean)
    
def model_binary_scoring_cot_preprocessor(response: str) -> str:
    """
    Parse a binary score from a scoring cot model message into a string. e.g. "1" "0"
    
    On failure to parse, return "".
    """
    # Parse a scoring model message into a binary int score.
    def _catch_bad_cot_pipeline(s: str):
        # Requires remove_think_tags first
        s = remove_think_tags(s)
        if s in (THINK_FAILED_MSG, NONE_CONTENT_ERROR_MSG):
            return "" # In scoring, no need to return flag
        
        state = pick_first_numbers_if_independent(s)
        if not filter_non_binary_scores(state):
            # The first numeric substring is not independent e.g. amici1000
            # Use last numeric substring
            last_numeric_substring = pick_last_numbers_if_independent(state)
            return last_numeric_substring if filter_non_binary_scores(last_numeric_substring) else ""
        return state
    
    return preprocess_pipeline(response, 
                               _catch_bad_cot_pipeline)

def model_binary_scoring_preprocessor(response: str) -> str:
    """
    Binary score parsing for non-cot model.
    
    On failure to parse, return "".
    """
    def _catch_non_binary_score_pipeline(s: str):
        # Requires pick_first_numbers_if_independent first
        if not filter_non_binary_scores(s):
            # The first numeric substring is not independent e.g. amici1000
            # Use last numeric substring
            last_numeric_substring = pick_last_numbers_if_independent(s)
            return last_numeric_substring if filter_non_binary_scores(last_numeric_substring) else ""
        return s

    return preprocess_pipeline(response, 
                               pick_first_numbers_if_independent, 
                               _catch_non_binary_score_pipeline)

def pick_first_letter_if_independent(s: str) -> str:
    """
    Pick the first letter if it's independent, otherwise returns "". An improved version of naive `str.strip()[0]` for MCQ responses.
    
    An independent letter has no adjacent alphanumeric characters.
    
    Examples
    - :A: `A` (typical, independent)
    - :\\*\\*A\\*\\*: `A` (non-word adjacent characters, still independent)
    - :$\\boxed{A}$: `A` (boxed formatting, captured and independent)
    - :3.5\\n A: `A` (not adjacent to a number, still independent)
    
    Counter examples
    - :The correct answer is A.: "" (T has h after it, not independent)
    - :10A should be the answer. B: "" (A is adjacent to 0, not independent)
    """
    
    unboxed = s.replace("\\boxed", " ")
    
    # Pick the first letter with immediately adjacent letters. 
    # For obvious reason, the previous character should not be alphabatical.
    pattern = "^[^A-Za-z]*?([^A-Za-z]?)([A-Za-z])(.?)"
    match = re.search(pattern, unboxed)
    
    # None match = no alphabetical characeter found
    if match != None:
        # Are adjacent characters alphanumeric?
        if not match.group(1).isalnum() and not match.group(3).isalnum():
            return match.group(2)
        
    return ""

def pick_last_letter_if_independent(s: str) -> str:
    """
    Pick the last letter if it's independent, otherwise returns "". An improved version of naive `str.strip()[-1]` for MCQ responses.
    
    An independent letter has no adjacent alphanumeric characters.
    
    Examples
    - :`A`: `A` (typical, independent)
    - :`**A**`: `A` (non-word adjacent characters, still independent)
    - :`$\\boxed{A}$`: `A` (boxed formatting - but does not affect matching since we pick the last letter, not the first)
    - :`Answer: A: 3.5`: `A` (not adjacent to a number, still independent)
    
    Counter examples
    - :This probably needs more clarification.: "" (n has o before it, not independent)
    - :The most probable cell is A10.: "" (A is adjacent to 1, not independent)
    """

    # Pick the last letter with immediately adjacent letters.
    # For obvious reason, the following character should not be alphabatical.
    pattern = "(.?)([A-Za-z])([^A-Za-z]?)[^A-Za-z]*?$"
    match = re.search(pattern, s)
    
    # None match = no alphabetical characeter found
    if match != None:
        # Are adjacent characters alphanumeric?
        if not match.group(1).isalnum() and not match.group(3).isalnum():
            return match.group(2)
        
    return ""

def pick_first_numbers_if_independent(s: str) -> str:
    """
    Pick the first numbers as a substring if it's independent. Support decimal point and "," as separator (removed in output). An improved version of naive `str.strip()[0]` for numeric answer responses.
    
    The numeric substring is independent if it does not have adjacent alphabetic characters.
    
    Examples:
    - `Found 3,242 jobs.`: `3242` (typical)
    - `Untested cases: 3.25%`: `3.25` (percentage is not supported, as it's not parsable by float)
    - `The string1 is not well written.`: "" (no independent number substring is found.)
    - "It's a string without any number.": "" (obviously)
    """
    
    # Will match on
    # - The last numbers with
    #   - optional percentage mark
    #   - shrinkable "[0-9]+[,.]" group, as in 1,000,000 where "1," and "000," form two extra groups
    #   - optional minus sign
    pattern = "^[^0-9]*?([^0-9]?)(-?([0-9]+[.,])*[0-9]+)([^0-9]?)"
    match = re.search(pattern, s)
    
    # None match = no alphabetical characeter found
    if match != None:
        # Are adjacent characters alphabetic?
        # Group note: in "Score: 2,214,529,240(Highest)"
        # - group(1) == " "
        # - group(2) == "2,214,529,240"
        # - group(3) == "529,"
        # - group(4) == "("
        if not match.group(1).isalpha() and not match.group(4).isalpha():
            return match.group(2).replace(",","")
    return ""

def pick_last_numbers_if_independent(s: str) -> str:
    """
    Pick the last numbers as a substring if it's independent. Support decimal point and "," as separator (removed in output). An improved version of naive `str.strip()[-1]` for numeric answer responses.
    
    The numeric substring is independent if it does not have adjacent alphabetic characters.
    
    Examples:
    - `Found 3,242 jobs.`: `3242` (typical)
    - `Untested cases: 3.25%`: `3.25` (percentage is not supported, as it's not parsable by float)
    - `The string1 is not well written.`: "" (no independent number substring is found.)
    - "It's a string without any number.": "" (obviously)
    """
    # Will match on
    # - The last numbers with
    #   - optional percentage mark
    #   - shrinkable "[0-9]+[,.]" group, as in 1,000,000 where "1," and "000," form two extra groups
    #   - optional minus sign
    pattern = "([^0-9]?)(-?([0-9]+[.,])*[0-9]+)([^0-9]?)[^0-9]*?$"
    match = re.search(pattern, s)
    
    # None match = no alphabetical characeter found
    if match != None:
        # Are adjacent characters alphabetic?
        # Group note: in "Score: 2,214,529,240(Highest)"
        # - group(1) == " "
        # - group(2) == "2,214,529,240"
        # - group(3) == "529,"
        # - group(4) == "("
        if not match.group(1).isalpha() and not match.group(4).isalpha():
            return match.group(2).replace(",","")
    return ""

@DeprecationWarning
def search_for_answer(s: str):
    # Deprecated.
    # Remove latex boxed statement e.g. `\boxed{A}`
    unboxed = s.replace("\\boxed", " ")
    
    # Allow multiple non-word characters before the first alphabetical (A-Za-z) group after "Answer:".
    # Model might use "**A**" or adding arbitrary spaces around the letter.
    # This is not matched: `Answer: not sure`
    pattern = f"[Aa]nswer:([^\\w]*?)([A-Za-z]+)"
    match = re.search(pattern, unboxed)
    if match != None:
        # Only pick if the first letter of the group is independent. e.g. `Answer: A` but not `Answer: Shark` 
        state = pick_first_letter_if_independent(match.group(2))
        if state:
            return state.upper()
        
    pattern_zh = f"答案[：:]([^\\w]*?)([A-Za-z]+)"
    match_zh = re.search(pattern_zh, unboxed)
    if match_zh != None:
        # Same as above.
        state2 = pick_first_letter_if_independent(match_zh.group(2))
        if state2:
            return state2.upper()

    logger.error(f"Failed to parse answer from response: {s[:50]}")
    return ""
    
def remove_think_tags(s: str):
    removed = re.sub(".*</[Tt]hink>", "", s, flags=re.DOTALL)
    
    # If no cot is parsed at all, return a fallback message string.
    if len(removed) == len(s):
        logger.warning(f"Encountered malformed cot section. The cot is likely incomplete. Will fall back to think failed msg.")
        return THINK_FAILED_MSG
    if len(removed.strip()) == 0:
        logger.warning("Despite cot being complete, no answer was made. Will fall back to think failed msg.")
        return THINK_FAILED_MSG
    return removed.lstrip("\n")

def filter_bad_option_letters(s: str):
    """
    A reverse filter for [A-D].
    
    :return: [A-D] or ""
    """
    return s if s in "ABCD" else ""

def filter_non_binary_scores(s: str):
    """
    A reverse filter for ["0", "1"].
    
    :return:  "0" | "1" | ""
    """
    
    return s if s in "01" else ""
    
def preprocess_pipeline(str_to_preprocess: str, *preprocessors):
    """
    Execute a series of preprocessors with empty string detection.
    """
    state = str_to_preprocess
    for func in preprocessors:
        if state in (THINK_FAILED_MSG, ANSWER_FAILED_MSG, NONE_CONTENT_ERROR_MSG):
            return state
        if state != "":
            state = func(state)
            continue
        logger.warning(f"Processed response is empty. Preprocessing halted.")
        return ""
    return state