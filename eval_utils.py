import re
import os 

def list_files_in_directory(directory, match_pattern=""):
    # Helper method. Takes a directory name and return a list of file names (with dir names)
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            matched = True
            # If provided match_pattern, will select only the qualified file names
            if match_pattern != "":
                matched = match_pattern in file

            if matched:
                file_paths.append(os.path.join(root, file))
    return file_paths

def judge(response_list, eval_name="Test set"):
    # response_list: a list containing response objects (dict), each comprising:
    # ...
    # correct_answer
    # response
    # ...
    # For each question, a score is judged with a simple comparison between correct_answer and response
    # Adding all score up gives the final score of a set.

    score = 0
    full_score = len(response_list)

    for resp_obj in response_list:
        correct_answer = str(resp_obj['correct_answer']).strip()
        response = str(resp_obj['response']).strip()
        try:
            if correct_answer[0] == response[0]:
                score += 1
        except IndexError:
            print(f"An index error has occurred in score judging. This is likely due to empty answer field(s): {correct_answer}, {response}. The answer will be judged as 0 point.")
            pass

    print(
        f"======\nEvaluation Report:\nEvaluation Name: {eval_name}\nAccuracy: {score}/{full_score} ({100*round(score/full_score, 3)}%)\n======\n")

    return {"eval_name": eval_name, "pass_rate": round(score/full_score, 3), "score": score, "full_score": full_score}

def sanitize_pathname(pathname):
    return re.sub(r'[^\w\-_\.]', '_', f"eval_result-{pathname}")