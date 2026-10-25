
import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30000,
        retry_options=types.HttpRetryOptions(
            attempts=1
        ),
    ),
)

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_ATTEMPTS = 5
INITIAL_DELAY = 5
MAX_DELAY = 60


def generate_code(prompt):
    """Генерує код із повторними спробами після тимчасових помилок."""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response")

            return response.text

        except Exception as error:
            error_text = str(error).upper()

            is_rate_limit = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "RATE LIMIT" in error_text
            )

            if not is_rate_limit or attempt == MAX_ATTEMPTS:
                raise

            delay = min(
                INITIAL_DELAY * (2 ** (attempt - 1)),
                MAX_DELAY,
            )

            delay += random.uniform(0, 1)

            print(
                f"Gemini rate limit reached. "
                f"Retry {attempt}/{MAX_ATTEMPTS - 1} "
                f"in {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError("Gemini code generation failed")


def clean_generated_code(code):
    code = code.strip()

    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        code = "\n".join(lines)

    return code.strip()
