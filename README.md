# DataBench: Question Answering over Tabular Data

A baseline system for answering natural-language questions about structured tables from [DataBench](https://huggingface.co/datasets/cardiffnlp/databench). The system uses a Gemini model to generate Python/pandas code, executes that code in a restricted Docker container, and compares the resulting answer with the reference answer.

> **Project status:** baseline implementation in progress. The repository currently supports running one question or a contiguous batch from the Hugging Face `qa` / `train` split in `full` or `lite` mode. The numbers printed by local runs are development results, not final benchmark results.

## How it works

1. Load questions from `cardiffnlp/databench` (`qa`, `train`).
2. Load the matching local Parquet table:
   - `all.parquet` for `full`;
   - `sample.parquet` for `lite`.
3. Build a prompt from the question, table schema, and expected answer type.
4. Ask Gemini to generate Python code that stores its answer in `result`.
5. Check the generated code's syntax.
6. Execute it in Docker with network access disabled and resource limits.
7. Compare the predicted answer with the reference answer using `databench_eval`.
8. Save one JSON object per question to `results.jsonl` and calculate summary metrics.

## Repository structure

```text
.
├── main.py                 # Single-question and batch execution
├── dataset_loader.py       # Loads local Parquet tables and their schema
├── prompt_builder.py       # Builds prompts for code generation
├── llm_client.py           # Gemini API client and retry handling
├── code_executor.py        # Syntax validation and restricted Docker execution
├── sandbox_runner.py       # Loads the table and executes generated code
├── evaluator.py            # Compares prediction with reference answer
├── metrics.py              # Summarizes results.jsonl
├── Dockerfile              # Runtime image for generated code
├── data/                   # Local datasets; intentionally not tracked by Git
├── scripts/                # Dataset download/inspection/validation utilities
└── tests/                  # Basic checks and test utilities
```

## Requirements

- Python 3.10+ (the Docker execution image uses Python 3.13).
- Docker Desktop or a working Docker Engine.
- A Google Gemini API key.
- The local DataBench tables required for the questions you run.

Python dependencies used by the current source include `datasets`, `pandas`, `pyarrow`, `python-dotenv`, `google-genai`, and `databench_eval`. A pinned `requirements.txt` is not included yet; create one before treating the setup as fully reproducible.

## Setup

### 1. Create and activate a virtual environment (PowerShell)

```powershell
cd D:\TUKE\DataBench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install datasets pandas pyarrow python-dotenv google-genai databench_eval
```

The `databench_eval` package name and installation instructions should be checked against the official evaluator repository if installation fails.

### 2. Configure the API key

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Do not commit `.env` or publish your API key. The repository's `.gitignore` excludes it.

The current model identifier is configured in `llm_client.py` as `gemini-3.5-flash-lite`; model availability can change. Update `MODEL_NAME` if needed.

### 3. Download local tables

The project expects files in this structure:

```text
data/
└── 001_Forbes/
    ├── all.parquet
    └── sample.parquet
```

The repository includes scripts for downloading and validating tables. Inspect their options and usage before running them. The `data/` directory is intentionally excluded from Git because the files are large.

### 4. Build the Docker execution image

From the repository root:

```powershell
docker build -t databench-executor .
docker image inspect databench-executor
```

`code_executor.py` expects an image named `databench-executor`. Rebuild it after changing the Dockerfile.

## Run

Start the interactive program:

```powershell
python main.py
```

Choose:
- **Run mode:** `single` or `batch`
- **Dataset mode:** `full` or `lite`

For a single question, enter its index. For batch mode, enter the starting index and number of questions. The program asks for confirmation before execution.

The question indices refer to the loaded `qa` / `train` split. The current implementation uses the reference `answer` for `full` and `sample_answer` for `lite`; this is useful for development evaluation, but it is not by itself the official SemEval test evaluation.

## Results and metrics

Each line in `results.jsonl` is a JSON record containing the question index, table ID, mode, question, answer type, expected and predicted answers, correctness, status, error, and generated code.

Run the metrics script separately:

```powershell
python metrics.py
```

The current metrics include:
- **Accuracy** — currently calculated over successful records only;
- **Success rate** — successful records divided by all records;
- **Execution error rate** — currently all non-success records divided by all records.

These definitions need refinement before final reporting. In particular, the project proposal defines accuracy over evaluated questions and execution error rate over execution attempts; generation, validation, and execution failures should be counted separately so the denominators are explicit.

## Tests and validation

Syntax check:

```powershell
python -m py_compile main.py code_executor.py dataset_loader.py evaluator.py llm_client.py metrics.py prompt_builder.py sandbox_runner.py
```

The scripts in `scripts/` validate downloaded tables and help locate question indices. Test utilities are in `tests/`. Some current test files are manual scripts rather than tests discoverable by `pytest`; see the project status notes below.

## Current limitations / next steps

- Add a pinned `requirements.txt` (or equivalent lock file) and document tested versions.
- Make the Docker image setup and dataset setup fully reproducible.
- Expand prompt instructions to cover all answer types (`boolean`, `category`, `number`, `list[category]`, `list[number]`) and normalize values before evaluation.
- Strengthen tests: the current `tests/test_docker_error.py` writes a fabricated error record; it does not actually execute code that fails inside Docker.
- Improve error categories and metrics. Distinguish generation errors, syntax/validation errors, Docker execution errors, missing expected answers, and incorrect predictions.
- Fix the accuracy denominator and document every metric's formula.
- Avoid using a single shared `generated_code.py` as the handoff between generation and execution; pass a per-run code file or use unique temporary files.
- Record model name, prompt/version, dataset configuration/split, run parameters, and package versions with each experiment.
- Use the official DataBench evaluation procedure and the correct held-out split for final benchmark claims; do not tune against the final test set.
- Add a README-linked project report and a separate appendix documenting LLM/AI-tool usage, prompts, and relevant outputs, as required by the course brief.

## Project context

This repository implements the **baseline** for a course project on natural-language question answering over tabular data. The first stage also requires a clear problem statement, a review of related work and data sources, justified evaluation methodology, reproducibility, and a report with citations. The code alone does not satisfy all first-stage deliverables.

## References

- DataBench dataset: https://huggingface.co/datasets/cardiffnlp/databench
- DataBench evaluation framework: https://github.com/jorses/databench_eval
- Osés Grijalba et al. (2024), *Question Answering over Tabular Data with DataBench: A Large-Scale Empirical Evaluation of LLMs*: https://aclanthology.org/2024.lrec-main.1179/
- Herzig et al. (2020), *TaPas: Weakly Supervised Table Parsing via Pre-training*: https://aclanthology.org/2020.acl-main.398/
- Yu et al. (2018), *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*: https://aclanthology.org/D18-1425/
