import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datasets import load_dataset
from main import process_question, save_result, RESULTS_PATH

dataset = load_dataset(
    "cardiffnlp/databench",
    "qa",
    split="train",
)

# Імітуємо результат помилки, яку повернув Docker.
record = {
    "index": 0,
    "dataset": "001_Forbes",
    "mode": "full",
    "question": dataset[0]["question"],
    "answer_type": dataset[0]["type"],
    "expected_answer": dataset[0]["answer"],
    "predicted_answer": None,
    "correct": None,
    "status": "error",
    "error": "Execution failed",
    "generated_code": "result = df['non_existing_column'].sum()",
}

# Зберігаємо початковий файл результатів, якщо він існує.
original_results = (
    RESULTS_PATH.read_text(encoding="utf-8")
    if RESULTS_PATH.exists()
    else None
)

try:
    save_result(record)

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        saved_records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("Last record status:", saved_records[-1]["status"])
    print("Last record error:", saved_records[-1]["error"])

finally:
    if original_results is None:
        RESULTS_PATH.unlink(missing_ok=True)
    else:
        RESULTS_PATH.write_text(
            original_results,
            encoding="utf-8",
        )

    print("Original results.jsonl restored.")
