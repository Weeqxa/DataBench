
import json
from pathlib import Path


def calculate_metrics(results_path="results.jsonl"):
    path = Path(results_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    total = len(records)
    successful = sum(
        record.get("status") == "success"
        for record in records
    )
    correct = sum(
        record.get("status") == "success"
        and record.get("correct") is True
        for record in records
    )
    errors = total - successful

    accuracy = correct / successful if successful else 0.0
    success_rate = successful / total if total else 0.0
    execution_error_rate = errors / total if total else 0.0

    return {
        "total": total,
        "successful": successful,
        "correct": correct,
        "errors": errors,
        "accuracy": round(accuracy * 100, 2),
        "success_rate": round(success_rate * 100, 2),
        "execution_error_rate": round(
            execution_error_rate * 100, 2
        ),
    }


if __name__ == "__main__":
    metrics = calculate_metrics()

    print("\n--- Evaluation metrics ---")
    print("Total questions:", metrics["total"])
    print("Successful executions:", metrics["successful"])
    print("Correct answers:", metrics["correct"])
    print("Errors:", metrics["errors"])
    print("Accuracy:", f"{metrics['accuracy']}%")
    print("Success rate:", f"{metrics['success_rate']}%")
    print(
        "Execution error rate:",
        f"{metrics['execution_error_rate']}%",
    )
