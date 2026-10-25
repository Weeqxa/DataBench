
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


REPO_ID = "cardiffnlp/databench"
DATASET_CONFIG = "qa"
DATA_DIR = Path("data")


def main():
    print("Loading DataBench question metadata...")

    dataset = load_dataset(
        REPO_ID,
        DATASET_CONFIG,
        split="train",
    )

    table_ids = sorted(set(dataset["dataset"]))
    print(f"Tables required: {len(table_ids)}")

    downloaded = 0
    skipped = 0
    failed = []

    for index, table_id in enumerate(table_ids, start=1):
        table_dir = DATA_DIR / table_id
        table_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{index}/{len(table_ids)}] {table_id}")

        for filename in ("all.parquet", "sample.parquet"):
            destination = table_dir / filename

            if destination.is_file():
                print(f"  Exists, skipping: {filename}")
                skipped += 1
                continue

            repo_path = f"data/{table_id}/{filename}"

            try:
                cached_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=repo_path,
                )

                # Копіюємо файл у локальну структуру проєкту.
                import shutil
                shutil.copy2(cached_path, destination)

                print(f"  Downloaded: {filename}")
                downloaded += 1

            except Exception as error:
                print(f"  FAILED: {filename}: {error}")
                failed.append((table_id, filename, str(error)))

    print("\n--- Download summary ---")
    print("Tables required:", len(table_ids))
    print("Files downloaded:", downloaded)
    print("Existing files skipped:", skipped)
    print("Failed files:", len(failed))

    if failed:
        print("\nFiles that failed:")
        for table_id, filename, error in failed:
            print(f"- {table_id}/{filename}: {error}")

    # Перевіряємо, що кожна таблиця має обидва файли.
    missing = []

    for table_id in table_ids:
        for filename in ("all.parquet", "sample.parquet"):
            if not (DATA_DIR / table_id / filename).is_file():
                missing.append(f"{table_id}/{filename}")

    print("Files still missing:", len(missing))

    if missing:
        for path in missing:
            print("-", path)
    else:
        print("All Full and Lite tables are available locally.")


if __name__ == "__main__":
    main()
