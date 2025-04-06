def make_en_cot_system_prompt():
    return "Answer the MCQ (only one option is correct). Think step by step first in <think> and </think>. In the conclusion section, present the correct option letter between <answer> and </answer>. "

def make_en_reasoning_suffix():
    return "\nASSISTANT: Let's think step by step: "

def make_en_system_prompt():
    return "Answer the MCQ (only one option is correct). In your response, present the correct option letter between <answer> and </answer>. "

def make_en_cod_system_prompt():
    return "Answer the MCQ (only one option is correct). Think step by step while keep only a minimum draft of each step. In the conclusion section, present the correct option letter between <answer> and </answer>. "

def make_zh_cot_system_prompt():
    return "请回答一道单项选择题（有唯一正确答案）。先在<think></think>中逐步思考，再在<answer>和</answer>之间输出正确的选项字母。"

def make_zh_reasoning_suffix():
    return "\nASSISTANT: 好的，让我一步步思考解决这个问题："

def make_zh_system_prompt():
    return "请回答一道单项选择题（有唯一正确答案），并在<answer>和</answer>之间输出正确的选项字母。"

def make_zh_cod_system_prompt():
    return "请回答一道单项选择题（有唯一正确答案）。先在<think></think>中逐步思考，每步仅保留草稿；再在<answer>和</answer>之间输出正确的选项字母。"