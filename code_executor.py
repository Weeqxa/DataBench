
import ast
import re
import subprocess
import uuid
from pathlib import Path


def validate_code(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as error:
        print("Syntax error:", error)
        return False


def run_code_in_docker(table_id, lite=False):
    # Перевіряємо ID таблиці
    if not re.fullmatch(r"\d{3}_[A-Za-z0-9_]+", table_id):
        raise ValueError("Invalid table ID")

    project_dir = Path(__file__).resolve().parent
    table_dir = project_dir / "data" / table_id

    # Вибираємо файл для відповідного режиму
    filename = "sample.parquet" if lite else "all.parquet"
    table_file = table_dir / filename

    if not table_file.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {table_file}"
        )

    code_file = project_dir / "generated_code.py"
    runner_file = project_dir / "sandbox_runner.py"

    if not code_file.is_file():
        raise FileNotFoundError(
            f"Generated code not found: {code_file}"
        )

    if not runner_file.is_file():
        raise FileNotFoundError(
            f"Sandbox runner not found: {runner_file}"
        )

    container_name = f"databench-{uuid.uuid4().hex[:12]}"

    command = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "--network", "none",
        "--memory", "512m",
        "--cpus", "1",
        "--pids-limit", "64",
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--mount",
        f"type=bind,source={table_file},target=/app/data/table.parquet,readonly",
        "--mount",
        f"type=bind,source={runner_file},target=/app/sandbox_runner.py,readonly",
        "--mount",
        f"type=bind,source={code_file},target=/app/generated_code.py,readonly",
        "databench-executor",
        "python", "-B", "/app/sandbox_runner.py"
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20
        )

        if completed.returncode != 0:
            print("Docker error:", completed.stderr)
            return None

        output = completed.stdout.strip()
        prefix = "Sandbox result: "

        if not output.startswith(prefix):
            print("Error: Unexpected Docker output")
            return None

        answer = output[len(prefix):].strip()
        print("Docker answer:", answer)

        return answer

    except subprocess.TimeoutExpired:
        print("Error: Execution timed out!")
        return None

    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
        except (subprocess.TimeoutExpired, OSError):
            pass
