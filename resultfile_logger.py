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

def log_resultfile(dataset_name, worker: Worker, eval_dir: str):
    # Default values
    default_values = {
        'date': datetime.now().strftime('%m-%d-%Y'),
        'model': 'local-model',
        'backend': 'null',
        'quantized': 'null',
        'quantization_bits': 'null',
        'temperature': 'null',
        'top_p': 'null',
        'frequency_penalty': 'null',
        'system_prompt': '',
        'prompt_prefix': '',
        'prompt_suffix': '',
        'test_set_name': dataset_name,
        'test_set_type': 'mcq',
        'judging_method': 'null'
    }
    
    # Update default values with provided parameter values
    values = worker.get_params()
    final_values = {**default_values, **values}
    
    # Create the RESULTFILE content
    resultfile_content = f"""date: {final_values['date']}
model: {final_values['model']}
backend: {final_values['backend']}
quantized: {final_values['quantized']}
quantization_bits: {final_values['quantization_bits']}

# model settings
temperature: {final_values['temperature']}
top_p: {final_values['top_p']}
frequency_penalty: {final_values['frequency_penalty']}

# prompt settings
system_prompt: "{final_values['system_prompt']}"
prompt_prefix: "{final_values['prompt_prefix']}"
prompt_suffix: "{final_values['prompt_suffix']}"

# score judging specifics
test_set_name: {final_values['test_set_name']}
test_set_type: {final_values['test_set_type']}
judging_method: {final_values['judging_method']}"""
    
    resultfile_path = f"{eval_dir}/RESULTFILE"
    
    def log_msg():
        print(f"A RESULTFILE for {dataset_name} has been initialized at: {resultfile_path}.")
        
    def log_failed_msg():
        print(f"Failed to initialize RESULTFILE for {dataset_name} at {resultfile_path}. The target evaluation directory likely does not exist.")
        
    # Write the RESULTFILE content to the specified file
    if os.path.isdir(eval_dir):
        write_to_file(resultfile_path, resultfile_content)
        log_msg()
    else:
        log_failed_msg()