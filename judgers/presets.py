"""
Preset judger methods (synchronous) for comparing a response with a correct answer.

Takes two strings and an optional context string, outputs a [0, 1] float score describing accuracy.

Does not care about preprocessing. That is delegated to preprocessor module.
"""

from judgers.model_binary_judge import model_scoring
import asyncio

# A failed flag used in score judging module in dataset_model
JUDGE_FAILED_MSG = "Judge failed."

async def STRICT_MATCH(response: str, answer: str, context="") -> float:
    """
    Strictly compare response and answer. No context.
    
    Recommended scenarios: mcq, closed-ended question
    """
    # Use asyncio.to_thread to submit as async task
    # Why? Because model_scoring is async! Need to maintain a unified interface.
    return await asyncio.to_thread(_STRICT_MATCH, response, answer, context=context)

def _STRICT_MATCH(response: str, answer: str, context="") -> float:
    return float(response == answer)
    
async def TEXT_SIMILARITY(response: str, answer: str, context="") -> float:
    """
    Calculate a similarity ratio @[0,1]. 0 = totally different, 1 = identical
    
    No context.
    """
    return await asyncio.to_thread(_TEXT_SIMILARITY, response, answer, context=context)

def _TEXT_SIMILARITY(response: str, answer: str, context="") -> float:
    ROWS, COLUMNS = len(response), len(answer)
    
    operation_matrix = [[0] * (COLUMNS + 1) for _ in range(ROWS + 1)]
    #       0   1   2   3   4   5
    #       ""  h   e   l   l   o
    # 0 ""  0   0   0   0   0   0
    # 1 w   0   0   0   0   0   0
    # 2 o   0   0   0   0   0   0
    # 3 r   0   0   0   0   0   0
    # 4 l   0   0   0   0   0   0
    # 5 d   0   0   0   0   0   0
    
    for r in range(ROWS + 1):
        operation_matrix[r][0] = r
    #       0   1   2   3   4   5
    #       ""  h   e   l   l   o
    # 0 ""  0   0   0   0   0   0
    # 1 w   1   0   0   0   0   0
    # 2 o   2   0   0   0   0   0
    # 3 r   3   0   0   0   0   0
    # 4 l   4   0   0   0   0   0
    # 5 d   5   0   0   0   0   0
    
    for c in range(COLUMNS + 1):
        operation_matrix[0][c] = c
    #       0   1   2   3   4   5
    #       ""  h   e   l   l   o
    # 0 ""  0   1   2   3   4   5
    # 1 w   1   0   0   0   0   0
    # 2 o   2   0   0   0   0   0
    # 3 r   3   0   0   0   0   0
    # 4 l   4   0   0   0   0   0
    # 5 d   5   0   0   0   0   0

    # Dynamic Programming
    for r in range(1, ROWS + 1):
        for c in range(1, COLUMNS + 1):
            if response[r-1] == answer[c-1]:
                operation_matrix[r][c] = operation_matrix[r-1][c-1]
                # No operation needed, cost not incremented
                    #       0   1   2   3   4   5
                    #       ""  h   e   l   l   o
                    # 0 "" <0>  1   2   3   4   5
                    # 1 h   1  <0>  0   0   0   0
                    # 2 a   2   0   0   0   0   0
                    # 3 r   3   0   0   0   0   0
                    # 4 r   4   0   0   0   0   0
                    # 5 o   5   0   0   0   0   0
            else:
                operation_matrix[r][c] = min(
                    # Find the previous edit step, with minimal cost
                    
                    # Consider "hello" as initial state and "world" the resultant state. Operations go right-to-left and top-to-bottom.
                    operation_matrix[r-1][c], # insert "w"
                    #       0   1   2   3   4   5
                    #       ""  h   e   l   l   o
                    # 0 ""  0   1   2   3  <4>  5
                    # 1 w   1   1   2   3  [0]  0
                    # 2 o   2   0   0   0   0   0
                    # 3 r   3   0   0   0   0   0
                    # 4 l   4   0   0   0   0   0
                    # 5 d   5   0   0   0   0   0
                    operation_matrix[r][c-1], # delete "l"
                    #       0   1   2   3   4   5
                    #       ""  h   e   l   l   o
                    # 0 ""  0   1   2   3   4   5
                    # 1 w   1   1   2  <3> [0]  0
                    # 2 o   2   0   0   0   0   0
                    # 3 r   3   0   0   0   0   0
                    # 4 l   4   0   0   0   0   0
                    # 5 d   5   0   0   0   0   0
                    operation_matrix[r-1][c-1] # substitute "l" with "w"
                    #       0   1   2   3   4   5
                    #       ""  h   e   l   l   o
                    # 0 ""  0   1   2  <3>  4   5
                    # 1 w   1   1   2   3  [0]  0
                    # 2 o   2   0   0   0   0   0
                    # 3 r   3   0   0   0   0   0
                    # 4 l   4   0   0   0   0   0
                    # 5 d   5   0   0   0   0   0
                    ) + 1 # Accumulate current step cost
    
    # Resultant state:
    #       0   1   2   3   4   5
    #       ""  h   e   l   l   o
    # 0 ""  0   1   2   3   4   5
    # 1 w   1   1   2   3   4   5
    # 2 o   2   2   2   3   4   4
    # 3 r   3   3   3   3   4   5
    # 4 l   4   4   4   3   3   4
    # 5 d   5   5   5   4   4  [4]
    # 
    # matrix[5][5] shows the edit distance.
    # To calculate the similarity, the maximum distance is max(ROWS, COLUMNS). We draw a simple ratio.
    
    return 1 - operation_matrix[ROWS][COLUMNS] / max(ROWS, COLUMNS)

async def MODEL_SCORING(response: str, answer: str, context="") -> float | str:
    """
    Score response with a judge model.
    
    Recommended scenarios: open-ended question, response model with bad IF
    """
    score = await model_scoring(response, answer, context)
    
    if score == "":
        return JUDGE_FAILED_MSG
    return float(score)