
import json
import time

from pathlib import Path


from datasets import load_dataset

from metrics import calculate_metrics

from dataset_loader import load_table, get_table_schema
from prompt_builder import build_prompt
from llm_client import generate_code, clean_generated_code
from code_executor import validate_code, run_code_in_docker
from evaluator import evaluate_answer


RESULTS_PATH = Path("results.jsonl")
CODE_PATH = Path("generated_code.py")




def process_question(dataset, index, mode):
    """Обробляє запитання з максимум однією повторною генерацією коду."""

    if mode not in ("full", "lite"):
        return {
            "index": index,
            "mode": mode,
            "status": "error",
            "error": "Invalid dataset mode. Use 'full' or 'lite'.",
        }

    question_data = dataset[index]

    table_id = question_data["dataset"]
    question = question_data["question"]
    answer_type = question_data["type"]

    expected_answer = (
        question_data["sample_answer"]
        if mode == "lite"
        else question_data["answer"]
    )

    record = {
        "index": index,
        "dataset": table_id,
        "mode": mode,
        "question": question,
        "answer_type": answer_type,
        "expected_answer": expected_answer,
        "predicted_answer": None,
        "correct": None,
        "status": "error",
        "error": None,
        "generated_code": None,
        "attempts": 0,
    }

    if expected_answer is None:
        record["status"] = "unscored"
        record["error"] = "Expected answer is missing"
        return record

    try:
        df = load_table(table_id, lite=(mode == "lite"))
        schema = get_table_schema(df)

        base_prompt = build_prompt(
            question,
            schema,
            answer_type=answer_type,
        )

        previous_error = None

        # Максимум дві спроби: початкова та одна повторна.
        for attempt in range(1, 3):
            record["attempts"] = attempt

            if attempt == 1:
                prompt = base_prompt
            else:
                print("Retrying code generation after execution error...")

                prompt = f"""
{base_prompt}

The previous generated code failed.

Previous code:
{record["generated_code"]}

Error:
{previous_error}

Generate corrected Python code.
Fix the specific error shown above.
Follow all original rules and return only executable Python code.
"""

            # Генеруємо код.
            raw_code = generate_code(prompt)
            generated_code = clean_generated_code(raw_code)
            record["generated_code"] = generated_code

            # Перевіряємо синтаксис.
            if not validate_code(generated_code):
                previous_error = "Invalid Python syntax"
                record["error"] = previous_error

                if attempt < 2:
                    continue

                return record

            # Зберігаємо код перед виконанням.
            CODE_PATH.write_text(
                generated_code,
                encoding="utf-8",
            )

            # Виконуємо код у Docker.
            answer = run_code_in_docker(
                table_id,
                lite=(mode == "lite"),
            )

            if answer is None:
                previous_error = "Docker execution failed. Check the traceback above."
                record["error"] = previous_error

                if attempt < 2:
                    continue

                return record

            # Виконання успішне — оцінюємо відповідь.
            record["predicted_answer"] = answer
            record["correct"] = bool(
                evaluate_answer(
                    predicted=answer,
                    expected=expected_answer,
                    answer_type=answer_type,
                )
            )
            record["status"] = "success"
            record["error"] = None

            return record

    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    return record


def save_result(record):
    """Дописує один результат у JSONL-файл."""
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_results_file():
    """Запитує, чи перезаписувати попередні результати."""
    if not RESULTS_PATH.exists():
        return True

    answer = input(
        "Results file already exists. "
        "Overwrite it? (yes/no): "
    ).strip().lower()

    if answer == "yes":
        RESULTS_PATH.write_text("", encoding="utf-8")
        print("Previous results cleared.")
        return True

    if answer == "no":
        print("Existing results will be preserved; new records will be appended.")
        return True

    print("Please enter 'yes' or 'no'.")
    return False


def main():
    # 1. Завантажуємо набір запитань
    print("Loading DataBench questions...")

    dataset = load_dataset(
        "cardiffnlp/databench",
        "qa",
        split="train",
    )

    print(f"Questions loaded: {len(dataset)}")

    # 2. Вибираємо режим запуску та даних
    run_mode = input("Run mode (single/batch): ").strip().lower()
    data_mode = input("Dataset mode (full/lite): ").strip().lower()

    if run_mode not in ("single", "batch"):
        print("Error: run mode must be 'single' or 'batch'.")
        return

    if data_mode not in ("full", "lite"):
        print("Error: dataset mode must be 'full' or 'lite'.")
        return

    # 3. Готуємо файл результатів
    if not prepare_results_file():
        return

    # 4. Одиночний режим
    if run_mode == "single":
        try:
            index = int(
                input(f"Question index (0-{len(dataset) - 1}): ")
            )
        except ValueError:
            print("Error: enter a valid integer.")
            return

        if not 0 <= index < len(dataset):
            print("Error: invalid question index.")
            return

        confirmation = input(
            "Generate code and execute it in Docker? (yes/no): "
        ).strip().lower()

        if confirmation != "yes":
            print("Execution cancelled.")
            return

        record = process_question(dataset, index, data_mode)
        save_result(record)

        print("\n--- Result ---")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print("Results saved to:", RESULTS_PATH)
        return

    # 5. Пакетний режим
    try:
        start = int(input("Start index: "))
        count = int(input("Number of questions: "))
    except ValueError:
        print("Error: start index and count must be integers.")
        return

    if start < 0 or count < 1 or start + count > len(dataset):
        print("Error: invalid index range or count.")
        return

    confirmation = input(
        f"Process {count} questions from index {start} "
        f"in {data_mode} mode? (yes/no): "
    ).strip().lower()

    if confirmation != "yes":
        print("Batch execution cancelled.")
        return

    summary = {
        "total": 0,
        "success": 0,
        "correct": 0,
        "errors": 0,
    }

    # 6. Послідовно обробляємо запитання
    for index in range(start, start + count):
        print(f"\n[{index}] Processing question...")

        record = process_question(dataset, index, data_mode)
        save_result(record)

        summary["total"] += 1

        if record["status"] == "success":
            summary["success"] += 1

            if record["correct"]:
                summary["correct"] += 1
        else:
            summary["errors"] += 1

        print("Question:", record["question"])
        print("Status:", record["status"])
        print("Predicted:", record["predicted_answer"])
        print("Correct:", record["correct"])

        if record["error"]:
            print("Error:", record["error"])

    # 7. Підсумок
    print("\n--- Batch summary ---")
    print("Processed:", summary["total"])
    print("Successful executions:", summary["success"])
    print("Correct answers:", summary["correct"])
    print("Errors:", summary["errors"])
    print("Results saved to:", RESULTS_PATH)

    metrics = calculate_metrics(RESULTS_PATH)

    print("\n--- Evaluation metrics ---")
    print("Total questions:", metrics["total"])
    print("Successful executions:", metrics["successful"])
    print("Correct answers:", metrics["correct"])
    print("Errors:", metrics["errors"])
    print("Accuracy:", f'{metrics["accuracy"]}%')
    print("Success rate:", f'{metrics["success_rate"]}%')
    print(
        "Execution error rate:",
        f'{metrics["execution_error_rate"]}%'
    )



if __name__ == "__main__":
    main()
