"""Reproducible paired DataBench development evaluation (not a submission)."""

import argparse
import ast
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TYPES = ("boolean", "number", "category", "list[category]", "list[number]")
FAILURE = "__DATABENCH_NO_PREDICTION__"
ERROR_CATEGORIES = ("api_infrastructure", "syntax", "execution", "result_format",
                    "data_selection", "operation", "unknown")
PACKAGES = ("datasets", "pandas", "pyarrow", "python-dotenv", "google-genai",
            "numpy", "databench_eval", "huggingface-hub")
BASELINE_FILES = ("main.py", "dataset_loader.py", "prompt_builder.py", "llm_client.py",
                  "code_executor.py", "sandbox_runner.py", "evaluator.py", "Dockerfile",
                  "scripts/evaluate_baseline.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: empty result line")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON; restore the interrupted line before resuming") from error
    return rows


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(arguments):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f"Command failed: {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def baseline_hashes():
    return {name: sha256(ROOT / name) for name in BASELINE_FILES}


def model_settings():
    # Read constants without importing the API client or loading any secret.
    tree = ast.parse((ROOT / "llm_client.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "MODEL_NAME", "MAX_ATTEMPTS", "INITIAL_DELAY", "MAX_DELAY"):
                    constants[target.id] = ast.literal_eval(node.value)
    return {**constants, "generation_parameters": "API defaults; not explicitly configured",
            "code_generation_attempts_per_question": 2}


def select_questions(dataset, count=100, seed=42):
    if count < 5 or count % 5:
        raise ValueError("Question count must be a positive multiple of five")
    rng = random.Random(seed)
    groups = {kind: defaultdict(list) for kind in TYPES}
    excluded = []
    for index, row in enumerate(dataset):
        if row.get("answer") is None or row.get("sample_answer") is None:
            excluded.append({"index": index, "dataset": row["dataset"], "type": row["type"],
                             "reason": "Missing full or lite reference"})
        elif row["type"] in groups:
            groups[row["type"]][row["dataset"]].append({**row, "index": index})
        else:
            raise ValueError(f"Unsupported answer type at index {index}: {row['type']}")
    per_type = count // len(TYPES)
    chosen = {}
    for kind in TYPES:
        tables = sorted(groups[kind])
        rng.shuffle(tables)
        for table in tables:
            rng.shuffle(groups[kind][table])
        available = sum(len(rows) for rows in groups[kind].values())
        if available < per_type:
            raise ValueError(f"Not enough eligible {kind} questions: need {per_type}, found {available}")
        chosen[kind] = []
        while len(chosen[kind]) < per_type:
            for table in tables:
                if groups[kind][table]:
                    chosen[kind].append(groups[kind][table].pop())
                    if len(chosen[kind]) == per_type:
                        break
    # Interleaving makes the first ten exactly two of every answer type.
    return [chosen[kind][i] for i in range(per_type) for kind in TYPES], excluded


def prepare(directory, count, seed, offline=False):
    if directory.exists():
        raise ValueError(f"Experiment directory already exists: {directory}; choose a new directory")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    dataset = load_dataset("cardiffnlp/databench", "qa", split="train")
    selected, excluded = select_questions(dataset, count, seed)
    table_hashes = {}
    for table in sorted({row["dataset"] for row in selected}):
        for filename in ("all.parquet", "sample.parquet"):
            relative = f"data/{table}/{filename}"
            if not (ROOT / relative).is_file():
                raise ValueError(f"Missing local table: {relative}")
            table_hashes[relative] = sha256(ROOT / relative)
    config = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage-one development evaluation; not an official competition submission",
        "dataset": {"repo": "cardiffnlp/databench", "config": "qa", "split": "train",
                    "fingerprint": dataset._fingerprint, "source_count": len(dataset)},
        "count": count, "seed": seed, "type_counts": dict(Counter(row["type"] for row in selected)),
        "table_count": len({row["dataset"] for row in selected}), "excluded_count": len(excluded),
        "model": model_settings(), "python": sys.version, "packages": versions(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "baseline_sha256": baseline_hashes(), "table_sha256": table_hashes,
    }
    directory.mkdir(parents=True)
    write_json(directory / "selected_qa.json", selected)
    write_json(directory / "excluded_questions.json", excluded)
    config["selection_sha256"] = sha256(directory / "selected_qa.json")
    write_json(directory / "config.json", config)
    print(f"Prepared {count} questions from {config['table_count']} tables in {directory}", flush=True)


def load_experiment(directory):
    config = read_json(directory / "config.json")
    if sha256(directory / "selected_qa.json") != config["selection_sha256"]:
        raise ValueError("Saved selection was modified; start a new experiment")
    selected = read_json(directory / "selected_qa.json")
    if len(selected) != config["count"] or len({r["index"] for r in selected}) != len(selected):
        raise ValueError("Selection count or unique indices do not match configuration")
    return config, selected


def validate_records(records, selected, mode, complete=True):
    expected = {row["index"]: row for row in selected}
    found = {}
    for record in records:
        index = record.get("index")
        if index not in expected:
            raise ValueError(f"Unexpected question index: {index}")
        if index in found:
            raise ValueError(f"Duplicate index {index} in {mode} results")
        row = expected[index]
        reference = row["sample_answer" if mode == "lite" else "answer"]
        if (record.get("mode") != mode or record.get("dataset") != row["dataset"]
                or record.get("question") != row["question"]
                or record.get("answer_type") != row["type"]
                or record.get("expected_answer") != reference):
            raise ValueError(f"Mismatched mode or question metadata at index {index}")
        if record.get("status") not in ("success", "error", "unscored"):
            raise ValueError(f"Unknown status at index {index}")
        found[index] = record
    if complete and set(found) != set(expected):
        raise ValueError(f"Missing {mode} results for indices: {sorted(set(expected) - set(found))}")
    return found


def redact(text, secrets):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def run(directory, mode="both", limit=None):
    config, selected = load_experiment(directory)
    if baseline_hashes() != config["baseline_sha256"]:
        raise ValueError("Baseline or evaluation script changed; prepare a new experiment")
    if versions() != config["packages"] or sys.version != config["python"]:
        raise ValueError("Python environment changed; prepare a new experiment")
    for relative, expected_hash in config["table_sha256"].items():
        if sha256(ROOT / relative) != expected_hash:
            raise ValueError(f"Table changed: {relative}; prepare a new experiment")
    if limit is not None and not 1 <= limit <= len(selected):
        raise ValueError("Limit must be within the saved selection")
    modes = ("full", "lite") if mode == "both" else (mode,)
    pending = {}
    for current in modes:
        found = validate_records(read_jsonl(directory / f"results_{current}.jsonl"), selected,
                                 current, complete=False)
        pending[current] = [row for row in selected[:limit] if row["index"] not in found]
    if not any(pending.values()):
        print("All requested records already exist; no API calls made")
        return
    image_id = command_output(["docker", "image", "inspect", "databench-executor", "--format", "{{.Id}}"])
    command_output(["docker", "info", "--format", "{{.ServerVersion}}"])
    environment_path = directory / "execution_environment.json"
    environment = {"docker_image_id": image_id, "docker_version": command_output(["docker", "--version"])}
    if environment_path.exists() and read_json(environment_path) != environment:
        raise ValueError("Docker environment changed; prepare a new experiment")
    write_json(environment_path, environment)
    # main uses paths relative to the repository and a single shared generated_code.py.
    # An exclusive repository-wide lock prevents overlapping runs of this harness.
    lock_path = ROOT / ".evaluation.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Another evaluation is running (.evaluation.lock); verify its process before removing a stale lock") from error
    original_cwd = Path.cwd()
    try:
        with lock:
            lock.write(str(os.getpid()))
        os.chdir(ROOT)
        import main as baseline
        secrets = (os.getenv("GEMINI_API_KEY"),)
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        for current in modes:
            for row in pending[current]:
                index = row["index"]
                print(f"[{current} index={index}] {row['type']} {row['dataset']}", flush=True)
                output = io.StringIO()
                started = time.monotonic()
                # Use the frozen question snapshot, preserving the original dataset index.
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    record = baseline.process_question({index: row}, index, current)
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                record = json.loads(redact(json.dumps(record, ensure_ascii=False), secrets))
                (logs / f"{current}_{index}.log").write_text(redact(output.getvalue(), secrets), encoding="utf-8")
                append_jsonl(directory / f"results_{current}.jsonl", record)
                print(f"  {record['status']}; correct={record['correct']}; attempts={record['attempts']}; {record['elapsed_seconds']}s", flush=True)
    finally:
        os.chdir(original_cwd)
        lock_path.unlink(missing_ok=True)


def prediction_line(record):
    if record["status"] != "success":
        return FAILURE, None
    value = record.get("predicted_answer")
    if isinstance(value, (dict, list, tuple)):
        line = repr(value)
    else:
        line = str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return FAILURE, "Non-finite numeric prediction"
    if len(line.splitlines()) > 1 or any(char in line for char in "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        return FAILURE, "Prediction contains a line separator"
    if line == FAILURE:
        return FAILURE, "Prediction collides with failure marker"
    return line, None


def summarize(records, flags, format_errors):
    total = len(records)
    successful = sum(row["status"] == "success" for row in records)
    correct = sum(bool(flag) for flag in flags)
    percentage = lambda n, d: round(100 * n / d, 4) if d else 0.0
    return {"total": total, "successful": successful, "correct": correct,
            "overall_accuracy": percentage(correct, total),
            "conditional_accuracy": percentage(correct, successful),
            "success_rate": percentage(successful, total),
            "processing_failure_rate": percentage(total - successful, total),
            "format_errors": sum(error is not None for error in format_errors),
            "unscored": sum(row["status"] == "unscored" for row in records)}


def classify(record, format_error):
    if format_error:
        return "result_format", format_error
    error = record.get("error") or ""
    if "Invalid Python syntax" in error:
        return "syntax", error
    if "Docker execution failed" in error:
        return "execution", "Baseline returned a generic execution failure; inspect the saved log, do not infer a traceback."
    if record["status"] != "success":
        if any(term in error.lower() for term in ("429", "503", "unavailable", "resource_exhausted", "clienterror", "servererror", "api", "filenotfounderror", "permission", "connection")):
            return "api_infrastructure", error
        return "unknown", error or "No diagnostic details available"
    return "unknown", "Executed successfully but disagrees with the official evaluator; review question and generated code."


def score(directory, limit=None):
    config, selected = load_experiment(directory)
    if importlib.metadata.version("databench_eval") != config["packages"]["databench_eval"]:
        raise ValueError("Evaluator version differs from the saved experiment")
    if limit is not None and not 1 <= limit <= len(selected):
        raise ValueError("Limit must be within the saved selection")
    target = selected[:limit]
    from datasets import Dataset
    from databench_eval import Evaluator
    qa = Dataset.from_list([{key: value for key, value in row.items() if key != "index"}
                            for row in target])
    # Validate all source records before creating or replacing any score artifact.
    ordered = {}
    for mode in ("full", "lite"):
        found = validate_records(read_jsonl(directory / f"results_{mode}.jsonl"), selected,
                                 mode, complete=limit is None)
        missing = [row["index"] for row in target if row["index"] not in found]
        if missing:
            raise ValueError(f"Missing {mode} results for indices: {missing}")
        ordered[mode] = [found[row["index"]] for row in target]
    output = directory / (f"scores_{limit}" if limit is not None else "scores")
    output.mkdir(exist_ok=True)
    report = {"scope": config["purpose"], "count_per_mode": len(target),
              "evaluator_version": config["packages"]["databench_eval"], "modes": {}}
    errors = []
    scored = {}
    for mode in ("full", "lite"):
        evaluator = Evaluator(qa=qa)
        reference_key = "sample_answer" if mode == "lite" else "answer"
        for row in target:
            if evaluator.default_compare(FAILURE, row[reference_key], row["type"]):
                raise ValueError(f"Failure marker matches reference at index {row['index']}")
        converted = [prediction_line(row) for row in ordered[mode]]
        lines = [item[0] for item in converted]
        format_errors = [item[1] for item in converted]
        path = output / ("predictions_lite.txt" if mode == "lite" else "predictions.txt")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reloaded = path.read_text(encoding="utf-8").splitlines()
        if len(reloaded) != len(target) or reloaded != lines:
            raise ValueError(f"Invalid prediction line count or serialization: {path}")
        # eval(path) uses the platform encoding; supplying an explicitly UTF-8 decoded
        # list preserves the exact file content on Windows as well.
        official_accuracy = evaluator.eval(reloaded, lite=mode == "lite")
        flags = [bool(value) for value in evaluator.evals]
        metrics = summarize(ordered[mode], flags, format_errors)
        metrics["official_accuracy"] = round(100 * official_accuracy, 4)
        if metrics["overall_accuracy"] != metrics["official_accuracy"]:
            raise ValueError("Official evaluator and aggregate accuracy disagree")
        metrics["by_type"] = {}
        for kind in TYPES:
            positions = [i for i, row in enumerate(target) if row["type"] == kind]
            metrics["by_type"][kind] = summarize([ordered[mode][i] for i in positions],
                                                [flags[i] for i in positions],
                                                [format_errors[i] for i in positions])
        report["modes"][mode] = metrics
        scored[mode] = {row["index"]: flag for row, flag in zip(target, flags)}
        decisions = []
        for row, record, line, flag, format_error in zip(target, ordered[mode], lines, flags, format_errors):
            decisions.append({"index": row["index"], "prediction_line": line,
                              "official_correct": flag, "format_error": format_error})
            if not flag or record["status"] != "success" or format_error:
                category, explanation = classify(record, format_error)
                errors.append({"index": row["index"], "mode": mode, "dataset": row["dataset"],
                               "answer_type": row["type"], "question": row["question"],
                               "expected": row[reference_key], "predicted": record.get("predicted_answer"),
                               "status": record["status"], "category": category, "explanation": explanation,
                               "generated_code": record.get("generated_code"), "error": record.get("error"),
                               "log": f"../logs/{mode}_{row['index']}.log"})
        write_json(output / f"decisions_{mode}.json", decisions)
    comparison = {"both_correct": [], "full_only": [], "lite_only": [], "both_incorrect": []}
    for row in target:
        index = row["index"]
        full, lite = scored["full"][index], scored["lite"][index]
        category = "both_correct" if full and lite else "full_only" if full else "lite_only" if lite else "both_incorrect"
        comparison[category].append(index)
    reviews_path = directory / "error_reviews.json"
    reviews = read_json(reviews_path) if reviews_path.exists() else []
    review_map = {}
    valid_indices = {row["index"] for row in selected}
    for review in reviews:
        key = (review["index"], review["mode"])
        if (key in review_map or review["index"] not in valid_indices
                or review["mode"] not in ("full", "lite")
                or review.get("category") not in ERROR_CATEGORIES
                or not review.get("explanation", "").strip()):
            raise ValueError(f"Invalid or duplicate manual error review: {key}")
        review_map[key] = review
    for error in errors:
        review = review_map.get((error["index"], error["mode"]))
        error["reviewed"] = review is not None
        if review:
            error.update(category=review["category"], explanation=review["explanation"])
    report["paired_comparison"] = comparison
    report["error_categories"] = dict(Counter(row["category"] for row in errors))
    report["reviewed_errors"] = sum(row["reviewed"] for row in errors)
    report["total_errors"] = len(errors)
    write_json(output / "metrics.json", report)
    write_json(output / "errors.json", errors)
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("index", "mode", "dataset", "answer_type", "question", "expected", "predicted",
                  "status", "category", "explanation", "reviewed", "error", "generated_code", "log")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for error in errors:
            writer.writerow({**error, "expected": repr(error["expected"]), "predicted": repr(error["predicted"])})
    render_report(output, report, errors, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def render_report(output, report, errors, config):
    lines = ["# DataBench development evaluation", "", config["purpose"], "",
             f"Selection: {report['count_per_mode']} paired questions; seed {config['seed']}; "
             f"dataset fingerprint `{config['dataset']['fingerprint']}`.", "",
             "Percentages below use the official comparator. Conditional accuracy is accuracy among successful records.", "",
             "| Metric | Full | Lite |", "|---|---:|---:|"]
    for key in ("total", "successful", "correct", "overall_accuracy", "conditional_accuracy", "success_rate", "processing_failure_rate", "format_errors"):
        lines.append(f"| {key} | {report['modes']['full'][key]} | {report['modes']['lite'][key]} |")
    lines += ["", "## Accuracy by answer type", "", "| Type | Full correct / total | Lite correct / total |", "|---|---:|---:|"]
    for kind in TYPES:
        full = report["modes"]["full"]["by_type"][kind]
        lite = report["modes"]["lite"]["by_type"][kind]
        lines.append(f"| {kind} | {full['correct']} / {full['total']} | {lite['correct']} / {lite['total']} |")
    lines += ["", "## Paired comparison", ""]
    for key, indices in report["paired_comparison"].items():
        lines.append(f"- {key}: {len(indices)}; indices {indices}")
    lines += ["", "## Error review", "", f"Manually reviewed: {report['reviewed_errors']} / {report['total_errors']} errors.", "",
              "Every failure is listed in errors.csv and errors.json. Categories inferred from explicit error messages only; wrong answers require manual review.",
              "Do not infer missing tracebacks. Review data-selection and operation errors using the saved code and local table.", "",
              "## Representative examples", ""]
    examples = []
    seen = set()
    for error in errors:
        key = (error["category"], error["answer_type"], error["mode"])
        if key not in seen:
            examples.append(error)
            seen.add(key)
        if len(examples) == 10:
            break
    if len(examples) < min(5, len(errors)):
        examples += [error for error in errors if error not in examples][:5 - len(examples)]
    for error in examples:
        lines += [f"### Index {error['index']} ({error['mode']}, {error['category']})", "",
                  error["question"], "", f"Expected: `{error['expected']!r}`; predicted: `{error['predicted']!r}`.",
                  "", error["explanation"], ""]
    lines += ["## Team handoff", "", "- Participant 1: reproduce failures using saved generated code and logs; any baseline fix requires a new experiment.",
              "- Participant 3: use this table, paired comparison and reviewed examples; these are development scores, not competition scores.",
              "- Participant 4: use config.json, execution_environment.json and frozen question metadata to verify reproducibility.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "score"):
        subparser = commands.add_parser(action)
        subparser.add_argument("--experiment", type=Path, required=True)
        if action == "prepare":
            subparser.add_argument("--count", type=int, default=100)
            subparser.add_argument("--seed", type=int, default=42)
            subparser.add_argument("--offline", action="store_true")
        else:
            subparser.add_argument("--limit", type=int)
        if action == "run":
            subparser.add_argument("--mode", choices=("full", "lite", "both"), default="both")
    arguments = parser.parse_args()
    directory = arguments.experiment.resolve()
    try:
        if arguments.action == "prepare":
            prepare(directory, arguments.count, arguments.seed, arguments.offline)
        elif arguments.action == "run":
            run(directory, arguments.mode, arguments.limit)
        else:
            score(directory, arguments.limit)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation error: {error}\n")


if __name__ == "__main__":
    main()
