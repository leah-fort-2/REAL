# Batch Request Tool

## Usage

- Preparation: Set parameters in run.py
  - Specify query input and output path
    - If not specified, will use "queries.csv" as input and "responses.csv" as output
  - Set the following parameters. If not specified, the global setting will be used in .env file:
    - base_url
    - api_keys
    - model
    - temperature
    - top_p
    - max_tokens
    - frequency_penalty
    - presence_penalty
- Spawn worker methods and gather them
- Wait for the resultant file(s).

> For security reason, it's strongly recommended to set base_url and api_key in a .env file.
> 
> An example .env file is provided in the root directory.
> 

## Advanced Usage

### Use multiple models/parameter settings

Just specify each in run.py and launch different worker methods.

```python
async def main():
    deepseek_params = {...}
    deepseek_worker = work(deepseek_params, input_path="deepseek_queries.csv", output_path="deepseek_responses.csv")

    friday_params = {...}
    friday_worker = work(friday_params, input_path="friday_queries.csv", output_path="friday_responses.xlsx")

    lingyi_params = {...}
    lingyi_worker = work(lingyi_params, input_path="lingyi_queries.xlsx", output_path="lingyi_responses.csv")
    # ...

    asyncio.gather(deepseek_worker, friday_worker, lingyi_worker)
```

And you batch run workers just like you do with one worker.

### Directly process a request list

It's possible to use process_requests in batch_request module directly.
It takes a list of string requests and returns a list of string responses.

Example:

```python
from batch_request import process_requests

requests = [
    "What's the capital of France?",
    "What is the greatest lake in surface area?",
    "How many continents are greater in surface area than Asia?"
    ]

# Provide the parameters in a dict.
# Required keys: base_url, api_key, model
# Optional keys: temperature, top_p, max_tokens, frequency_penalty, presence_penalty 
# The parameters here are not affected by the global settings for functional purity reason.
# Unspecified optional keys will not be sent.
responses = process_requests(requests, {...})

print(responses)
# ["The capital of France is Paris.",
#  "The largest lake in surface area is the Caspian Sea.",
#  "There are zero continents greater in surface area than Asia."]
```

## Example http request

See details in api_actions module.


Header:

```
{
  'Authorization': 'Bearer [Your API Key]',
  'Content-Type': 'application/json'
}
```

Body: 

```
{
  'messages':
    [
      {
      'role': 'user', 
      'content': "What's the name of the first bridge in the world? "
      }
    ],
  'model': 'deepseek-chat', 
  'temperature': 0.0, 
  'top_p': 1.0, 
  'frequency_penalty': 0.0, 
  'presence_penalty': 0.0, 
  'max_tokens': 8192
}
```
