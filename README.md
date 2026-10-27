# DataBench — baseline for question answering over tables

This repository contains a baseline system for answering natural-language questions about tabular data from [DataBench](https://huggingface.co/datasets/cardiffnlp/databench). It asks a Gemini model to generate Python code, runs that code in a restricted Docker container, and compares the returned value with the dataset's reference answer for development checks.

> **Status:** baseline implementation in progress. The current interactive program supports one question or a contiguous batch from the Hugging Face `cardiffnlp/databench` dataset (`qa` configuration, `train` split), using either `full` or `lite` local tables. Local results are development checks, not official benchmark results.

## Pipeline

1. Load question metadata from Hugging Face.
2. Load the corresponding local Parquet table: `all.parquet` in `full` mode or `sample.parquet` in `lite` mode.
3. Build a prompt using the question, column names/types, and expected answer type.
4. Ask Gemini to generate Python code that assigns the answer to `result`.
5. Parse the generated code to check Python syntax.
6. Execute the code in Docker with networking disabled, resource limits, a read-only container filesystem, dropped Linux capabilities, and a timeout.
7. Serialize the result to JSON in the container and parse it back into a Python value.
8. If syntax validation or Docker execution fails, make at most one additional code-generation attempt. Temporary Gemini rate-limit errors are retried separately in `llm_client.py`.
9. Compare successful predictions with the reference answer using `databench_eval` and append a record to `results.jsonl`.

The retry loop currently treats a Docker failure as a generic execution error in the follow-up prompt. Although the traceback is printed in the console, it is not passed back as structured error data from `code_executor.py` to `process_question()` yet.

## Repository layout

```text
.
├── main.py                         # Interactive single-question and batch runs
├── dataset_loader.py               # Loads local Parquet tables and extracts schemas
├── prompt_builder.py               # Builds type-aware code-generation prompts
├── llm_client.py                   # Gemini client, rate-limit retries, code cleanup
├── code_executor.py                # Syntax check and restricted Docker execution
├── sandbox_runner.py               # Executes code and normalizes the result to JSON
├── evaluator.py                    # Compares predictions using databench_eval
├── metrics.py                      # Computes metrics from results.jsonl
├── requirements.txt                # Pinned dependencies matching config.json
├── Dockerfile                      # Docker image for code execution
├── scripts/
│   ├── download_all_tables.py       # Downloads all.parquet and sample.parquet files
│   ├── evaluate_baseline.py        # 100-question development evaluation harness
│   ├── verify_reproducibility.py   # Independent cryptographic hash and metric verifier
│   ├── inspect_databench_files.py   # Inspects repository files on Hugging Face
│   ├── validate_all_tables.py      # Checks downloaded table files
│   ├── find_test_indices.py        # Prints first question index for each table
│   └── list_models.py              # Helper for inspecting available models
└── tests/
    ├── test_answer_types.py         # Unit tests for all 5 DataBench answer types
    ├── test_evaluate_baseline.py    # Unit tests for evaluation selection and integrity
    ├── test_retry.py                # Unit test for retry after execution failure
    ├── test_dataset_loader.py       # Manual validation of all local table loads
    ├── test_docker_error.py         # Checks saving a fabricated error record
    ├── test_docker_execution_error.py # Manual Docker failure check
    └── test_evaluator.py            # Manual evaluator comparison check
```

## Requirements

- Python 3.11 or newer recommended (tested on Python 3.14.8 on Windows and Python 3.10+ on Linux; Docker image uses Python 3.13). Note that `pandas>=3.0` requires Python >= 3.11.
- Docker Desktop on Windows, or Docker Engine on Linux.
- A Google Gemini API key (for running baseline generations; offline evaluation and verification do not require an API key).
- Local DataBench Parquet files (only for running baseline against tables).

A pinned `requirements.txt` is provided in the repository root matching the exact versions used during the stage-one evaluation.

## Setup

Run commands from the repository root.

### 1. Create a virtual environment and install dependencies

**On Linux / macOS (Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**On Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `databench_eval` cannot be installed from PyPI, install directly from its official repository: `pip install git+https://github.com/jorses/databench_eval.git`.

### 2. Configure Gemini (Optional for evaluation & tests)

Create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Keep the real key private. Do not commit `.env` or paste the key into issues or logs. The model identifier is configured in `llm_client.py` as `gemini-3.5-flash-lite`; availability and quotas can change.

### 3. Download the local tables

```powershell
python scripts/download_all_tables.py
```

The downloader obtains both files for each table. The expected layout is:

```text
data/
└── 001_Forbes/
    ├── all.parquet
    └── sample.parquet
```

The local `data/` directory is intentionally not part of the source archive because the tables are large. After downloading, validate them with:

```powershell
python scripts/validate_all_tables.py
```

### 4. Build the Docker image

```powershell
docker build -t databench-executor .
docker image inspect databench-executor
```

The image must be named `databench-executor`, as expected by `code_executor.py`. Rebuild it after changing the Dockerfile.

## Run the baseline

```powershell
python main.py
```

Choose the run mode and data mode when prompted:

- `single`: process one question index.
- `batch`: process a contiguous range of question indices.
- `full`: use `all.parquet` and the question's `answer` field.
- `lite`: use `sample.parquet` and the question's `sample_answer` field.

The index is the position in the currently loaded Hugging Face `qa` / `train` split. The program asks before executing the selected question(s). If `results.jsonl` already exists, it asks whether to overwrite it or append new records.

## Output and metrics

Each line of `results.jsonl` is one JSON object with fields such as:

- `index`, `dataset`, `mode`, `question`, `answer_type`;
- `expected_answer`, `predicted_answer`, `correct`;
- `status`, `error`, `generated_code`, `attempts`.

The `attempts` field records code-generation attempts within `process_question()`. Rate-limit retries inside the Gemini client are separate and are not reflected in that field.

Run metrics on the records in the current `results.jsonl` file with:

```powershell
python metrics.py
```

The current metrics are defined as follows:

- **Accuracy:** correct records divided by successful records.
- **Success rate:** successful records divided by all records.
- **Execution error rate:** all non-success records divided by all records.

The current implementation counts every non-success status—including `unscored` records—as an error in these aggregate metrics. Interpret the metric labels accordingly; refine status categories and denominators before using them for final scientific reporting. Also note that when records are appended, the metrics cover the entire file, not just the most recent batch.

## Tests and verification

### 1. Independent reproducibility verification

To verify the integrity and cryptographic hashes of all stage-one evaluation artifacts, predictions, and reviewed errors (no API key or tables required):

```bash
python scripts/verify_reproducibility.py
```

### 2. Unit tests

To run all automated test suites (covering retries, dataset selection integrity, and all 5 DataBench answer types: boolean, number, category, list[category], list[number]):

```bash
python -m unittest tests.test_answer_types tests.test_retry tests.test_evaluate_baseline -v
```

Syntax check:

```bash
python -m py_compile main.py code_executor.py dataset_loader.py evaluator.py llm_client.py metrics.py prompt_builder.py sandbox_runner.py scripts/evaluate_baseline.py scripts/verify_reproducibility.py
```

The files `test_dataset_loader.py`, `test_docker_error.py`, `test_docker_execution_error.py`, and `test_evaluator.py` in `tests/` are written as manual scripts for local debugging.

## Stage-one evaluation

`scripts/evaluate_baseline.py` selects 100 paired questions (20 per answer type),
saves Full/Lite results, validates prediction files, and scores them with `databench_eval`.

### Fast offline re-scoring

To verify and re-score the frozen stage-one evaluation without calling the Gemini API or re-running Docker:

```bash
python scripts/evaluate_baseline.py score --experiment state/eval_stage1
```

### Running a new evaluation

```bash
python scripts/evaluate_baseline.py prepare --experiment state/new_eval --count 100 --seed 42 --offline
python scripts/evaluate_baseline.py run --experiment state/new_eval
python scripts/evaluate_baseline.py score --experiment state/new_eval
```

Use a new experiment directory; `--offline` requires cached dataset metadata.
Add `--limit 10` to `run` and `score` for a smoke check before the full run.
Repeated runs skip saved results, including failures; scoring makes no API calls.
Code, environment or table changes require a new experiment. Run sequentially,
as code execution uses a shared `generated_code.py` file.

Completed development evaluation: 100 questions from 59 tables in `qa/train`, seed 42.

| Metric | Full | Lite |
|---|---:|---:|
| Overall accuracy (all questions) | 86% | 81% |
| Accuracy among successful records | 88.66% | 85.26% |
| Processing success rate | 97% | 95% |

All 33 incorrect or failed records were reviewed. Some mismatches involve ambiguous
questions or reference/format inconsistencies; scores were not adjusted. These are
development results, not competition scores.

Saved experiment results: [report](state/eval_stage1/scores/report.md),
[metrics](state/eval_stage1/scores/metrics.json), [error analysis](state/eval_stage1/scores/errors.csv).
The snapshot's metadata, frozen questions, results and final scores can be committed;
logs, smoke scoring outputs and other experiment directories remain ignored.

## Remaining baseline limitations

- The generated code is passed to Docker through a shared `generated_code.py` file. Concurrent runs can conflict; use per-run temporary files or unique paths before parallelizing.
- The retry prompt receives a generic Docker failure message rather than the captured traceback. Return structured execution errors so the model can correct the actual failure.
- The code-execution sandbox is restricted with Docker settings, but executing arbitrary LLM-generated Python still requires careful security review; Docker settings alone are not a formal guarantee of safety.
- Clarify metric definitions and distinguish generation, syntax-validation, execution, missing-reference, and incorrect-answer cases.
- The local development run uses `qa` / `train` reference answers. Confirm the required held-out split and official evaluation procedure before reporting benchmark scores.
- The evaluation harness exports verified prediction files for its frozen development subset. A complete competition submission and competition-data loader remain separate work.
- The course brief requires a separate appendix documenting LLM/AI-tool use, including tools/models, purposes, prompts, and relevant outputs.

## Project context

The code is the baseline implementation for a team project on natural-language question answering over tabular data. The first project stage also includes the problem statement, motivation, related work and data sources, justified evaluation methodology, reproducibility, and a clear report. Code alone does not cover all stage-one deliverables.

## References

- DataBench dataset: <https://huggingface.co/datasets/cardiffnlp/databench>
- DataBench evaluation framework: <https://github.com/jorses/databench_eval>
- Osés Grijalba et al. (2024), “Question Answering over Tabular Data with DataBench: A Large-Scale Empirical Evaluation of LLMs”: <https://aclanthology.org/2024.lrec-main.1179/>
- Herzig et al. (2020), “TaPas: Weakly Supervised Table Parsing via Pre-training”: <https://aclanthology.org/2020.acl-main.398/>
- Yu et al. (2018), “Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task”: <https://aclanthology.org/D18-1425/>
