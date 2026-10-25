
from pathlib import Path
import pandas as pd


# 1. Завантажуємо вибрану таблицю
df = pd.read_parquet("/app/data/table.parquet")

# 2. Читаємо згенерований Python-код
code = Path("/app/generated_code.py").read_text(
    encoding="utf-8"
)

# 3. Створюємо середовище виконання
namespace = {
    "df": df
}

# 4. Виконуємо згенерований код
exec(code, namespace)

# 5. Перевіряємо наявність результату
if "result" not in namespace:
    raise ValueError(
        "Generated code did not define the required variable 'result'."
    )

result = namespace["result"]

# 6. Виводимо результат для Docker Executor
print("Sandbox result:", result)
