
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


def normalize_result(value):
    """Перетворює результат на JSON-сумісні типи."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {
            str(key): normalize_result(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [normalize_result(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


df = pd.read_parquet("/app/data/table.parquet")

code = Path("/app/generated_code.py").read_text(
    encoding="utf-8"
)

namespace = {"df": df}
exec(code, namespace)

if "result" not in namespace:
    raise ValueError(
        "Generated code did not define the required variable 'result'."
    )

result = normalize_result(namespace["result"])

print(
    "Sandbox result:",
    json.dumps(result, ensure_ascii=False, allow_nan=False),
)
