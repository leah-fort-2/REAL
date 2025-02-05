"""
Response/answer preprocessors for the score judging feature `judge` in ResponseSet module.

Processors are wrappers of function pipelines. Each pipeline takes a string (in this case the response text) and a series of string manipulation methods.

When empty string is encountered at any stage of the pipeline, the workflow is halted with warning, and an empty string is returned immediately.
"""

import re

def mcq_preprocessor(response: str):
    return preprocess_pipeline(response, 
                               strip_then_use_first_letter_then_uppercase)

def mcq_cot_preprocessor(response: str):
    return preprocess_pipeline(response,
                               remove_think_tags,
                               strip_then_use_first_letter_then_uppercase)
    
def mcq_cot_preprocessor_for_bad_if(response:str):
    def strip_with_fallback(s: str):
        state = strip_whitespace_then_asterisk_then_last_letter_then_uppercase(s)
        # e.g.  "Answer: A" -> "A"
        if state not in ["A", "B", "C", "D"]:
            state = search_for_answer(s)
            # e.g. "Answer: A\nThe above is the answer." => "A"
        return state
    return preprocess_pipeline(response, 
                               remove_think_tags,
                               strip_with_fallback)

def strip_then_use_first_letter_then_uppercase(s: str):
    # Remove non-alphabetical characters
    state=re.sub("[^A-Za-z]", "", s)
    try:
        return state.strip()[0].upper()
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

def search_for_answer(s: str):
    pattern = f"[Aa]nswer:.*?([A-Za-z])"
    match = re.search(pattern, s, flags=re.DOTALL)
    if match == None:
        print(f"Failed to parse answer from response: {s}")
        return ""
    return match.group(1).upper()
    
def remove_think_tags(s: str):
    removed = re.sub("<[Tt]hink>.*</[Tt]hink>", "", s, flags=re.DOTALL)
    if len(removed) == len(s):
        print(f"Encountered malformed cot section. Can't judge. Will be judged as blank.")
        return ""
    return removed
    
def preprocess_pipeline(str_to_preprocess: str, *preprocessors: list):
    state = str_to_preprocess
    for func in preprocessors:
        if state != "":
            state = func(state)
            continue
        print("Encountered empty response. It's likely the model returned an empty response!")
        return ""
    return state