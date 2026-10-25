
def build_prompt(question, schema, answer_type="unknown"):
    columns = "\n".join(
        f"- {name}: {dtype}"
        for name, dtype in schema.items()
    )

    type_instructions = {
        "boolean": """
- The result must be a single boolean value: True or False.
- Do not return a sentence or explanation.
""",
        "number": """
- The result must be a single numeric value.
- Do not return a sentence or explanation.
""",
        "string": """
- The result must be a single string value.
- Do not return a sentence or explanation.
""",
    }

    output_instruction = type_instructions.get(
        str(answer_type).lower(),
        "- Return only the answer value, without explanations."
    )

    prompt = f"""
You are a Python data analysis assistant.

A pandas DataFrame named df is already loaded.

Available columns:
{columns}

Question:
{question}

Expected answer type: {answer_type}

Generate Python code that answers the question.

Rules:
- Use only the DataFrame df.
- Use only existing columns.
- Store the final answer in a variable named result.
- The value of result must match the expected answer type.
{output_instruction}
- Return only Python code, without explanations or markdown.
"""

    return prompt.strip()
