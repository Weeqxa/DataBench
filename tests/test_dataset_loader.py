
from datasets import load_dataset

from dataset_loader import load_table, get_table_schema


def main():
    dataset = load_dataset(
        "cardiffnlp/databench",
        "qa",
        split="train",
    )

    table_ids = sorted(set(dataset["dataset"]))
    failures = []
    checked = 0

    for index, table_id in enumerate(table_ids, start=1):
        for lite in (False, True):
            mode = "lite" if lite else "full"

            try:
                df = load_table(table_id, lite=lite)
                schema = get_table_schema(df)

                if df.empty:
                    raise ValueError("Loaded table is empty")

                if not schema:
                    raise ValueError("Table schema is empty")

                checked += 1
                print(
                    f"[{index}/80] {table_id} ({mode}): "
                    f"{len(df)} rows, {len(schema)} columns — OK"
                )

            except Exception as error:
                failures.append((table_id, mode, str(error)))
                print(
                    f"[{index}/80] {table_id} ({mode}): "
                    f"FAILED — {error}"
                )

    print("\n--- Loader test summary ---")
    print("Successful loads:", checked)
    print("Failures:", len(failures))

    if failures:
        for table_id, mode, error in failures:
            print(f"- {table_id} ({mode}): {error}")
    else:
        print("All tables load correctly through dataset_loader.py")


if __name__ == "__main__":
    main()
