
def build_prompt(question, schema, answer_type="unknown"):
    columns = "\n".join(
        f"- {name}: {dtype}"
        for name, dtype in schema.items()
    )

    type_instructions = {
        "boolean": """
- Return exactly one boolean value: True or False.
- Do not return a sentence or explanation.
""",
        "number": """
- Return one numeric value.
- Do not round unless the question explicitly requests rounding
  or the expected answer format clearly requires it.
- Do not return a sentence or explanation.
""",
        "string": """
- Return one plain Python string.
- Do not return a sentence or explanation.
""",
        "category": """
- Return one category value as a plain Python string or number.
- Do not return pandas Categorical objects.
- Do not construct values using type(series.dtype)(value).
- Do not use pandas or NumPy dtype objects as constructors.
- Preserve the category's original value and meaning.
- Do not return a sentence or explanation.
""",
        "list[category]": """
- Return a Python list of category values.
- Use plain strings or numbers, not pandas Categorical objects.
- Preserve duplicates unless the question explicitly requests unique values.
- Respect the requested order and tie-breaking rules.
- Preserve missing values in a form consistent with the question.
""",
        "list[number]": """
- Return a Python list of numeric values.
- Preserve duplicate values unless the question explicitly requests unique values.
- Do not use unique() unless the question explicitly asks for distinct values.
- Preserve the requested number of records, even when values repeat.
- Convert NumPy numeric scalars to native Python numbers when needed.
""",
    }

    normalized_type = str(answer_type).strip().lower()

    output_instruction = type_instructions.get(
        normalized_type,
        "- Return only the answer value, without explanations."
    )

    prompt = f"""
You are a Python data analysis assistant.

A pandas DataFrame named df is already loaded.

Available columns and their data types:
{columns}

Question:
{question}

Expected answer type: {answer_type}

Generate Python code that answers the question using the DataFrame.

Rules:
- Use only the DataFrame df and the columns listed above.
- Use only existing column names.
- Store the final answer in a variable named result.
- The result must match the expected answer type.
- Interpret the exact wording of the question carefully.
- Do not use unique() or drop_duplicates() unless the question explicitly
  asks for unique or distinct values.
- When the question asks for the top or bottom N records, preserve
  duplicate values if different records have the same value.
- When alphabetical tie-breaking is requested, apply it among candidates
  tied at the relevant rank or count.
- Return a plain Python scalar, string, number, boolean, or list as appropriate.
- Convert NumPy scalar values to native Python values when needed.
- Do not return pandas Categorical objects.
- Do not construct values using type(series.dtype)(value).
- Do not use pandas or NumPy dtype objects as constructors.
- Use int(value), float(value), str(value), or bool(value) only when
  the conversion is appropriate for the expected answer.
- Do not round numeric results unless the question or expected answer
  format explicitly requires rounding.
- Do not invent columns or values.
{output_instruction}
- Return only executable Python code, without explanations or markdown.
"""

    return prompt.strip()
