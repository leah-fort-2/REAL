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
- top_k: (worker's parameter)
- frequency_penalty: (worker's parameter)
- presence_penalty: (worker's parameter)
- repetition_penalty: (worker's parameter)

# prompt settings
- system_prompt: (worker's parameter)
- prompt_prefix: (worker's parameter)
- prompt_suffix: (worker's parameter)

# score judging specifics
- test_set_name: (set in adapter)
- test_set_type: (set in adapter)
- judging_method: (set in adapter)

# other parameters
- *remaining parameters
"""

import os
from datetime import datetime
from worker import Worker
from io_managers import get_writer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_FIELD_VALUE="null"

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
        'test_set_name': dataset_name,
        'test_set_type': 'mcq',
    }
    
    # Update default values with provided parameter values
    values = worker.get_params()
    final_values = {**default_values, **values, **params}
    final_values.pop("api_key", None)
        
    # Create the RESULTFILE content
    resultfile_content = f"""date: {final_values.pop('date', FALLBACK_FIELD_VALUE)}
model: {final_values.pop('model', 'local-model')}
backend: {final_values.pop('backend', FALLBACK_FIELD_VALUE)}
quantized: {final_values.pop('quantized', FALLBACK_FIELD_VALUE)}
quantization_bits: {final_values.pop('quantization_bits', FALLBACK_FIELD_VALUE)}

# model settings
temperature: {final_values.pop('temperature', FALLBACK_FIELD_VALUE)}
top_p: {final_values.pop('top_p', FALLBACK_FIELD_VALUE)}
top_k: {final_values.pop('top_k', FALLBACK_FIELD_VALUE)}
max_tokens: {final_values.pop('max_tokens', FALLBACK_FIELD_VALUE)}
frequency_penalty: {final_values.pop('frequency_penalty', FALLBACK_FIELD_VALUE)}
presence_penalty: {final_values.pop('presence_penalty', FALLBACK_FIELD_VALUE)}
repetition_penalty: {final_values.pop('repetition_penalty', FALLBACK_FIELD_VALUE)}

# prompt settings
system_prompt: {escape(final_values.pop('system_prompt', FALLBACK_FIELD_VALUE))}
prompt_prefix: {escape(final_values.pop('prompt_prefix', FALLBACK_FIELD_VALUE))}
prompt_suffix: {escape(final_values.pop('prompt_suffix', FALLBACK_FIELD_VALUE))}

# score judging specifics
test_set_name: {final_values.pop('test_set_name', FALLBACK_FIELD_VALUE)}
test_set_type: {final_values.pop('test_set_type', FALLBACK_FIELD_VALUE)}
judging_method: {final_values.pop('judging_method', FALLBACK_FIELD_VALUE)}
subset_max_size: {final_values.pop('subset_max_size', FALLBACK_FIELD_VALUE)}

# other parameters
{"\n".join([
    f"{key}: {escape(str(val))}" for key, val in final_values.items()
    ])}
"""
    
    resultfile_path = f"{log_dir}/RESULTFILE"
    
    def log_msg():
        logger.info(f"A RESULTFILE for {dataset_name} has been initialized at: {resultfile_path}.")
        
    def log_failed_msg():
        logger.error(f"Failed to initialize RESULTFILE for {dataset_name} at {resultfile_path}. The target evaluation directory likely does not exist.")
        
    # Write the RESULTFILE content to the specified file
    if os.path.isdir(log_dir):
        writer, ext = get_writer(resultfile_path)
        writer(resultfile_path, resultfile_content)
        log_msg()
    else:
        log_failed_msg()
        
def escape(s):
    if s != FALLBACK_FIELD_VALUE:
        s = s.replace("\n", "\\n")
        s = f"\"{s}\""
    return s