# LLM Batch Request Tool

## Usage

The tool is designed to conveniently launch concurrent requests to LLM APIs.

0. **Setup**:
   - Install the required dependencies:
     pip install -r requirements.txt
   - Create `.env` where global settings are saved. A `.env.example` file is provided for your reference. 

1. **Prepare Your Input**:
   - Prepare a CSV or XLSX file with your queries. A demo input file is provided as `example_queries.csv`.
     - The file has empty cells in its query column. The tool will safely use them as request input while warning the user.
   - In `run.py`, create `QuerySet` instances that loads your file, or a literal query list.
    ```py
    test_queries = QuerySet("example_queries.csv")

    # Or

    literal_queries = [
                        "What",
                        "When",
                        "Why"
                      ]
    test_queries = QuerySet(literal_queries)
    ```

2. **Configure the Worker**:
   - Create `RequestParams` profile with specified parameters if needed.
     - Omit certain parameters in instantiation to fall back to their global settings.
    ```py
    deepseek_params = RequestParams(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-chat"
    )

    # Or, use global settings entirely:

    default_params = RequestParams()
    ```
   - Spawn `Worker` instances. `Worker` instances are not callable. To actually use them, load `invoke` with a `QuerySet` instance and an optional query column name `query_key`.
    > The worker will submit request with only values in `query_key` column. If left blank, the column `query` will be used.
    >
      - Adjust the workflow to your use case
        - Use multiple workers for different test sets
        - Or use multiple profiles
        - Or write your own task functions
    > `Worker.invoke(...)` returns `ResponseSet` instance. You can easily instantiate it from a list of responses.
    > ```py
    > EditedResponses = ResponseSet({"response": "...", "score": 1})
    > ```
   - When you are done, use `store_to` method to export results (append). Only csv and xlsx are supported.
   ```py
    responses = await deepseek_worker.invoke(test_queries)
    responses.store_to("deepseek.csv")
    # Or
    EditedResponses.store_to("edited_responses.csv")
   ```

### Example Usage

```py
# Example usage with deepseek. But you can use any api provider:    

query_set=QuerySet("example_queries.csv")
# You can use a local list instead

deepseek_params = RequestParams(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat"
)

# Create a worker (QuerySet-> ResponseSet)
deepseek_worker = Worker(deepseek_params)

async def deepseek_task():
    result = await deepseek_worker.invoke(query_set, query_key="query")
    
    # Note: Will overwrite existing files
    result.store_to("deepseek.xlsx")
    
await asyncio.gather(deepseek_task())
```

> For security reasons, it's strongly recommended to set base_url and api_key in a .env file.
> 
> An example .env file is provided in the root directory.

## Advanced Usage

### Multiple workers

Specify each just like you do with one worker.

```py
async def main():
    query_set = QuerySet("example_queries.csv")

    deepseek_params = {...}
    deepseek_worker = Worker(deepseek_params)

    friday_params = {...}
    friday_worker = Worker(friday_params)

    # ...

    async def deepseek_task():
        result = await deepseek_worker.invoke(query_set, query_key="query")
        result.store_to("deepseek.xlsx")
        return result

    async def friday_task():
        result = await friday_worker.invoke(query_set, query_key="query")
        result.store_to("friday.csv")
        return result

    # ...

    asyncio.gather(deepseek_task(), friday_task())
```

### Batch request & Dataset division

You can set a `BATCH_SIZE` limit in .env file to limit concurrent request number.

You can also use `divide` method from `QuerySet` instances. This helps keep tasks in parcels when you have a large dataset.

```py
for div in query_set.divide(10):
    result = await deepseek_worker.invoke(div)
    # Note: Will append to existing file
    result.store_to("deepseek.xlsx")
```
Multiple results will append sequentially to the specified file.

### Directly access a request list

It's possible to use process_requests in batch_request module directly.
It takes a list of string requests and returns a list of string responses.

```py
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

See request details in api_actions module.


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