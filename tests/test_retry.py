
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


class RetryTest(unittest.TestCase):

    @patch("main.run_code_in_docker", side_effect=[None, 1])
    @patch("main.generate_code", side_effect=[
        "result = df['missing_column'].sum()",
        "result = 1",
    ])
    @patch("main.clean_generated_code", side_effect=lambda code: code)
    @patch("main.validate_code", return_value=True)
    @patch("main.load_table", return_value=object())
    @patch("main.get_table_schema", return_value={"value": "int64"})
    @patch("main.build_prompt", return_value="Test prompt")
    @patch("main.evaluate_answer", return_value=True)
    def test_retries_after_execution_failure(
        self,
        mock_evaluate,
        mock_prompt,
        mock_schema,
        mock_load,
        mock_validate,
        mock_clean,
        mock_generate,
        mock_docker,
    ):
        questions = [{
            "dataset": "001_Forbes",
            "question": "Test question",
            "type": "number",
            "answer": 1,
            "sample_answer": 1,
        }]

        with patch("main.CODE_PATH", Path("test_generated_code.py")):
            with patch.object(Path, "write_text", return_value=None) as mock_write:
                record = main.process_question(
                    questions,
                    0,
                    "full",
                )

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["predicted_answer"], 1)
        self.assertTrue(record["correct"])
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(mock_generate.call_count, 2)
        self.assertEqual(mock_docker.call_count, 2)

        retry_prompt = mock_generate.call_args_list[1].args[0]
        self.assertIn("previous generated code failed", retry_prompt.lower())
        self.assertIn("Docker execution failed", retry_prompt)

        print("\nPASS: retry mechanism works correctly.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
