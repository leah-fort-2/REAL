"""
Initialize a RESULTFILE with (default field values):
- date
- model: (worker's parameter)
- backend: (null)
- quantized: (null)
- quantization_bits: (null)

# model settings
- temperature: (worker's parameter)
- top_p: (worker's parameter)
- repetition_penalty: (worker's parameter)

# prompt settings
- system_prompt: (worker's parameter)
- prompt_prefix: (worker's parameter)
- prompt_suffix: (worker's parameter)

# score judging specifics
- test_set_name: (null)
- test_set_type: (null)
- judging_method: (null)
"""

import os
from datetime import datetime
from worker import Worker
from io_managers.raw_file_writer import write_to_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_resultfile(dataset_name, worker: Worker, log_dir: str, params: dict):
    """
    Write a `RESULTFILE` to the specified directory. Append.
    
    :params str dataset_name: The dataset name to be logged.
    :params Worker worker: A worker instance. Its parameters will be logged as evaluation parameters.
    :params str log_dir: Where to store the log file.
    :params params: Specify test_set_type: str and judging_method: str in RESULTFILE

    """
    # Default values
    default_values = {
        'date': datetime.now().strftime('%m-%d-%Y'),
        'model': 'local-model',
        'backend': 'null',
        'quantized': 'null',
        'quantization_bits': 'null',
        'temperature': 'null',
        'top_p': 'null',
        'max_tokens': 'null',
        'frequency_penalty': 'null',
        'presence_penalty': 'null',
        'system_prompt': '',
        'prompt_prefix': '',
        'prompt_suffix': '',
        'test_set_name': dataset_name,
        'test_set_type': 'mcq',
        'judging_method': 'null'
    }
    
    # Update default values with provided parameter values
    values = worker.get_params()
    final_values = {**default_values, **values, **params}
        
    # Create the RESULTFILE content
    resultfile_content = f"""date: {final_values['date']}
model: {final_values['model']}
backend: {final_values['backend']}
quantized: {final_values['quantized']}
quantization_bits: {final_values['quantization_bits']}

# model settings
temperature: {final_values['temperature']}
top_p: {final_values['top_p']}
max_tokens: {final_values['max_tokens']}
frequency_penalty: {final_values['frequency_penalty']}
presence_penalty: {final_values['presence_penalty']}

# prompt settings
system_prompt: "{escape(final_values['system_prompt'])}"
prompt_prefix: "{escape(final_values['prompt_prefix'])}"
prompt_suffix: "{escape(final_values['prompt_suffix'])}"

# score judging specifics
test_set_name: {final_values['test_set_name']}
test_set_type: {final_values['test_set_type']}
judging_method: {final_values['judging_method']}"""
    
    resultfile_path = f"{log_dir}/RESULTFILE"
    
    def log_msg():
        logger.info(f"A RESULTFILE for {dataset_name} has been initialized at: {resultfile_path}.")
        
    def log_failed_msg():
        logger.error(f"Failed to initialize RESULTFILE for {dataset_name} at {resultfile_path}. The target evaluation directory likely does not exist.")
        
    # Write the RESULTFILE content to the specified file
    if os.path.isdir(log_dir):
        write_to_file(resultfile_path, resultfile_content)
        log_msg()
    else:
        log_failed_msg()
        
def escape(str):
    state = str.replace(':', '\\:').replace("\n", "\\n")
    return state