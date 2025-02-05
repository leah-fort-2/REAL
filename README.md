# REAL

- [About](#about)
- [Example Usage](#example-usage)
- [Workflow Orchestration in Adapters](#workflow-orchestration-in-adapters)
- [Timeout and Batch size](#timeout-and-batch-size)

## About

REAL (RESTful Evaluation Automation Layer) is a tool for evaluation LLMs via RESTful APIs. It is designed to be a layer between the LLM api and test datasets.

Features:

- **API-based evaluation**: Benchmark LLM performances via any OpenAI-compatible api, in a fashion closer to real-world scenarios.
- **Workers**: Benchmark tasks are assigned to workers. A worker takes its profile (model name, base url and api key, and model parameters) and conduct batched requests/score judging.
- **Dataset adapters**: The evaluation logic is encapsulated in adapters, dedicated to meet the requirements of differnt data structures of datasets.
- **Create adapters**: Customize the evaluation workflow by creating adapters that cater to specific needs.
- **Evaluation presets**: mmlu/ceval/cmmlu (datasets are not included)
- **Score judging**: Conduct flexible score judging via response/answer preprocessor chain. CoT? Reflection? Preprocess it.

## Example Usage

REAL starts at `run.py`. To start the first evaluation with REAL, follow the steps:

1. Create an .env file

Look at .env.example which has all the required env variables. Duplicate it, create a `.env`, and fill in your own values.

2. Prepare a dataset

Take your own datasets. Put it in `datasets` directory (nested directories are supported).

Currently, REAL supports ceval, cmmlu and mmlu. But theoretically you can use REAL for any MCQ-type dataset.

You might need to implement your own adapter for other data structures than "[question, a,  b, c, d, answer]". But trust, it's not that difficult. Keep reading for how an adapter is used in REAL architecture.

3. Create a worker @`run.py`

A worker takes a `RequestParams` instance which is its profile. It will `invoke` requests following this profile.

4. Call an adapter @`run.py`

An adapter wraps the evaluation logic and takes 1) the dataset directory 2) the worker 3) the results dir (for where to store responses) 4) score output path (for storing the metrics to a file) and 5) test mode (to test only the first subset, used for workflow debug purposes, default to False).

Below is an example of how to use an adapter.

```python
dataset_path = "path/to/mmlu/dataset"

worker_profile = RequestParams(
  model="deepseek-chat",
  temperature=0,
  max_tokens=128,
  frequency_penalty=0
  system_prompt="You are a helpful assistant."
)

industrious_worker = Worker(worker_profile)

results_dir = "path/to/results"

score_output_path

await conduct_mmlu(dataset_path, worker, results_dir, score_output_path)
```

## Workflow Orchestration in Adapters

An adapter is dedicated to these procedures:

**Pre**
- Validate dataset and results path, preventing braindead mistakes like FileNotFoundError after 10,000 questions have been evaluated.

**Evaluating**

- Scrape subset test files (csv/xlsx/jsonl) from the dataset directory using `list_files_in_directory`: `str` -> `str`. An optional file extension criterion can be specified.
- Create `QuerySet` instance with the path of each subset test file path.
- Merge dataset fields if needed. Designed for mcq datasets, `QuerySet` implemented `merge_keys` that connects field names/values with linebreakers.
- Call workers with a `QuerySet` instance and a query field name (which field to request, default to `query`). This creates a `Job`. A `Job` can be `invoke`d to fetch all llm responses and returns them in a `ResponseSet` instance.
  - Orchestrate multiple workers here.

**Post**

- Call `ResponseSet` method `store_to` with an output path. (supported format: csv, xlsx, jsonl) Storing full response records is highly advised for archiving purposes.
- Call `ResponseSet` method `judge` with 1) an answer field, 2) an eval name (for marking each subset record), and 3) response preprocessors before scoring. The `judge` method will return a literal dictionary containing scoring info.
  - Some preprocessors are ready at `dataset_adapters.response_preprocessors`. 
- Spawn a `ResponseSet` instance with the judge result dictionary to `store_to` a score output file.
- Lastly, log the metadata of evaluation(s). Method `log_resultfile` is provided for templated log files.

## Timeout and Batch size

Since REAL is essentially an augmented batch request tool, it has timeout and batch size settings.

- **Timeout**: Happens when a request has not been responded after a given time (timeout). Configure at `api_actions` module.

```python
timeout = ClientTimeout(total=60)  # 1 minute timeout
```

- **Batch size**: When a query set is submitted to an api, its concurrent request number is limited to `BATCH_SIZE` by a semaphore. Configure at .env file.

NOTE: The semaphore is JOB-SPECIFIC: each job has a semaphore, so 2 jobs invoked at a time can launch `2 * BATCH_SIZE` concurrent requests.

```bash
BATCH_SIZE=5
```

```python
job1 = worker(dataset_1, "question").invoke()
# This has a semaphore of 5
job2 = worker(dataset_2, "Question").invoke()
# Also has a semaphore of 5
```