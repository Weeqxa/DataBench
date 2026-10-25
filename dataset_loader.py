
import pandas as pd
from pathlib import Path


def load_table(table_id, lite=False):
    """
    Завантажує таблицю DataBench.

    lite=False — повна таблиця (all.parquet).
    lite=True  — вибірка DataBench Lite (sample.parquet).
    """
    filename = "sample.parquet" if lite else "all.parquet"
    table_path = Path("data") / table_id / filename

    if not table_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {table_path}"
        )

    return pd.read_parquet(table_path)


def get_table_schema(df):
    schema = {}

    for column in df.columns:
        schema[column] = str(df[column].dtype)

    return schema
