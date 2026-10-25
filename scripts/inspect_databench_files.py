
from huggingface_hub import HfApi

REPO_ID = "cardiffnlp/databench"

api = HfApi()

print("Inspecting repository files...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("Total repository files:", len(files))

# Показуємо приклади шляхів, що стосуються таблиць.
parquet_files = [
    path for path in files
    if path.endswith(".parquet")
]

print("Parquet files:", len(parquet_files))

for path in parquet_files[:40]:
    print(path)
