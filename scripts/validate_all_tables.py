
from pathlib import Path

import pandas as pd

from datasets import load_dataset


DATA_DIR = Path("data")


def main():
    dataset = load_dataset(
        "cardiffnlp/databench",
        "qa",
        split="train",
    )

    table_ids = sorted(set(dataset["dataset"]))

    checked = 0
    failed = []

    for index, table_id in enumerate(table_ids, start=1):
        print(f"[{index}/{len(table_ids)}] {table_id}")

        for filename in ("all.parquet", "sample.parquet"):
            path = DATA_DIR / table_id / filename

            try:
                if not path.is_file():
                    raise FileNotFoundError(str(path))

                df = pd.read_parquet(path)

                if df.empty:
                    raise ValueError("Table is empty")

                print(
                    f"  OK: {filename} — "
                    f"{len(df)} rows, {len(df.columns)} columns"
                )
                checked += 1

            except Exception as error:
                print(f"  FAILED: {filename}: {error}")
                failed.append({
                    "table": table_id,
                    "file": filename,
                    "error": str(error),
                })

    print("\n--- Validation summary ---")
    print("Tables:", len(table_ids))
    print("Files successfully read:", checked)
    print("Failed files:", len(failed))

    if failed:
        print("\nFailures:")
        for item in failed:
            print(
                f"- {item['table']}/{item['file']}: "
                f"{item['error']}"
            )
    else:
        print("All Full and Lite files are readable.")


if __name__ == "__main__":
    main()
