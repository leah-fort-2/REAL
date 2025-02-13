# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
import logging as global_logging

from dataset_models import ResponseSet
from external_eval_methods.instruction_following_eval import instructions_registry

from io_managers import get_reader
from request_manager.request_manager import FALLBACK_ERR_MSG

global_logging.basicConfig(level=global_logging.INFO)
logger = global_logging.getLogger(__name__)
SCORING_BATCH_SIZE = int(os.getenv("SCORING_BATCH_SIZE", "5"))

_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename: str):
  """Read inputs from an ifeval jsonl test file."""
  inputs = []
  reader, _ = get_reader(input_jsonl_filename)
  
  literal_list = reader(input_jsonl_filename)

  for example in literal_list:
    inputs.append(
        InputExample(key=example["key"],
                      instruction_id_list=example["instruction_id_list"],
                      prompt=example["prompt"],
                      kwargs=example["kwargs"]))

  return inputs

def read_prompt_list_from_obj(input_obj_list):
  """
  Read a list[InputExample] directly from the parsed ifeval entry list.
  
  Parse beforehand.
  """
  inputs = []
  for obj in input_obj_list:
    inputs.append(
        InputExample(key=obj["key"],
                      instruction_id_list=obj["instruction_id_list"],
                      prompt=obj["prompt"],
                      kwargs=obj["kwargs"]))


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  response = prompt_to_response[inp.prompt]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["prompt"]] = example["response"]
  return return_dict


def print_report(outputs):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  print(f"prompt-level: {prompt_correct / prompt_total}")
  print(f"instruction-level: {instruction_correct / instruction_total}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # The IFEval input entries parsed from the dataset file
  # key, instruction_id_list, prompt, kwargs
  eval_entries = read_prompt_list(_INPUT_DATA.value)
  # A prompt: response dict
  prompt_to_response = read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  # test if accuracy
  for test_if, output_file_name in [
      (test_instruction_following_strict, "eval_results_strict"),
      (test_instruction_following_loose, "eval_results_loose"),
  ]:
    logger.info(f"Generating {output_file_name}...")
    outputs = []
    for eval_obj in eval_entries:
      outputs.append(test_if(eval_obj, prompt_to_response))
    if_score = [o.follow_all_instructions for o in outputs]
    accuracy = sum(if_score) / len(outputs)
    logger.info(f"Accuracy: {accuracy}")

    output_file_name = os.path.join(
        _OUTPUT_DIR.value, output_file_name + ".jsonl"
    )
    write_outputs(output_file_name, outputs)
    logger.info(f"Generated: {output_file_name}")

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    print_report(outputs)
    
def ifeval_judge_strict(response_set: ResponseSet, ifeval_eval_file_path: str):
  """
  Still an adapter style score judging method. Paired with dataset_model module to incorporate into the workflow.
  
  - :side effects: Modifies response_set.responses by adding a new field `ifeval_strict_score`
  
  :params response_set: A ResponseSet instance, obtained from a finished Worker.Job
  :params ifeval_eval_file_path: Locate the IFEval jsonl source file.
  :return: Returns a score briefing dict.
  """
  EVAL_NAME = "ifeval"
  # The IFEval input entries parsed from the dataset file
  # key, instruction_id_list, prompt, kwargs
  eval_entries = read_prompt_list(ifeval_eval_file_path)
  
  query_key = response_set.get_query_key()
  response_key = response_set.get_query_key()
  query_key = response_set.get_query_key()
  response_key = response_set.get_query_key()
  responses = response_set.get_responses()
  # A prompt: response dict
  prompt_to_response = {}
  [prompt_to_response.update(
    {resp_obj[query_key] # The prompt
        : resp_obj[response_key]}) # The response
    {resp_obj[query_key] # The prompt
        : resp_obj[response_key]}) # The response
        for resp_obj in responses]

  # test if accuracy
  for test_if, test_mode_name in [
      (test_instruction_following_strict, "strict")
      # (test_instruction_following_loose, "eval_results_loose"), # Not using loose here.
  ]:
    logger.info(f"Conduct score judging: {test_mode_name}...")
    outputs : list[OutputExample] = [] 
    # Get score OutputExample obj for each eval_obj in eval_entries.
    for eval_obj, resp_obj in zip(eval_entries, responses):
      if resp_obj[response_key]!=FALLBACK_ERR_MSG:
        output_example: OutputExample = test_if(eval_obj, prompt_to_response)
        outputs.append(output_example)
        # Score calculation
        score: bool = output_example.follow_all_instructions
        resp_obj.update({
          f"{EVAL_NAME}_{test_mode_name}_score": 1 if score else 0}
        )
    # Finished updating all resp objs in ResponseSet.
    
    # Total score and acc calculation
    result_score = sum([o.follow_all_instructions for o in outputs])
    full_score = len(outputs)
    accuracy = result_score/full_score
    logger.info(
            f"\n======\nEvaluation Report:\nEvaluation Name: {EVAL_NAME}\nAccuracy: {result_score}/{full_score} ({round(100*result_score/full_score, 1)}%)\n======\n")

    # Return a score briefing object
    return {"eval_name": EVAL_NAME, "score": result_score, "full_score": full_score, "accuracy": accuracy}