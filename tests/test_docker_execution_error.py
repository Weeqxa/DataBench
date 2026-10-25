
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from code_executor import run_code_in_docker

CODE_PATH = PROJECT_ROOT / "generated_code.py"

original_code = (
    CODE_PATH.read_text(encoding="utf-8")
    if CODE_PATH.exists()
    else None
)

try:
    CODE_PATH.write_text(
        "result = df['non_existing_column'].sum()\n",
        encoding="utf-8",
    )

    answer = run_code_in_docker("001_Forbes", lite=False)

    print("\nReturned answer:", repr(answer))

    if answer is None:
        print("PASS: Docker execution error was detected.")
    else:
        print("FAIL: Invalid code unexpectedly returned a result.")

finally:
    if original_code is None:
        CODE_PATH.unlink(missing_ok=True)
    else:
        CODE_PATH.write_text(original_code, encoding="utf-8")

    print("Original generated_code.py restored.")
