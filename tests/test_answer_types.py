"""Unit tests for DataBench answer types normalization, serialization, and evaluation.

Validates all 5 DataBench answer types:
  1. boolean
  2. number
  3. category
  4. list[category]
  5. list[number]
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import contextlib
import io
from unittest.mock import patch

with contextlib.redirect_stdout(io.StringIO()):
    with patch("pandas.read_parquet", return_value=None), \
         patch("pathlib.Path.read_text", return_value="result = None"):
        from sandbox_runner import normalize_result

from scripts.evaluate_baseline import prediction_line, FAILURE

# Optional evaluator import: Evaluator may be from databench_eval or mocked
try:
    from evaluator import evaluate_answer
    from databench_eval import Evaluator
    HAS_DATABENCH_EVAL = True
except ImportError:
    HAS_DATABENCH_EVAL = False


class BooleanTypeTest(unittest.TestCase):
    """Tests for the 'boolean' answer type."""

    def test_normalization(self):
        self.assertIs(normalize_result(True), True)
        self.assertIs(normalize_result(False), False)

    def test_serialization(self):
        line_true, err_true = prediction_line({"status": "success", "predicted_answer": True})
        self.assertEqual(line_true, "True")
        self.assertIsNone(err_true)

        line_false, err_false = prediction_line({"status": "success", "predicted_answer": False})
        self.assertEqual(line_false, "False")
        self.assertIsNone(err_false)

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_matches(self):
        evaluator = Evaluator.__new__(Evaluator)
        for pred, exp in [(True, "True"), (False, "False"), ("true", "True"), ("yes", "True"), ("no", "False")]:
            with self.subTest(pred=pred, exp=exp):
                self.assertTrue(evaluator.default_compare(pred, exp, "boolean"))

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_mismatches(self):
        evaluator = Evaluator.__new__(Evaluator)
        self.assertFalse(evaluator.default_compare(True, "False", "boolean"))
        self.assertFalse(evaluator.default_compare(False, "True", "boolean"))


class NumberTypeTest(unittest.TestCase):
    """Tests for the 'number' answer type."""

    def test_normalization(self):
        self.assertEqual(normalize_result(42), 42)
        self.assertEqual(normalize_result(3.14), 3.14)
        self.assertEqual(normalize_result(0), 0)
        self.assertEqual(normalize_result(-10), -10)
        self.assertIsNone(normalize_result(float("nan")))
        self.assertIsNone(normalize_result(float("inf")))

    def test_serialization(self):
        line, err = prediction_line({"status": "success", "predicted_answer": 19})
        self.assertEqual(line, "19")
        self.assertIsNone(err)

        line_zero, err_zero = prediction_line({"status": "success", "predicted_answer": 0})
        self.assertEqual(line_zero, "0")
        self.assertIsNone(err_zero)

        line_nan, err_nan = prediction_line({"status": "success", "predicted_answer": float("nan")})
        self.assertEqual(line_nan, FAILURE)
        self.assertIsNotNone(err_nan)

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_matches(self):
        evaluator = Evaluator.__new__(Evaluator)
        test_cases = [
            ("19", "19.0"),
            (19, "19"),
            (19.0, "19"),
            (0, "0.0"),
            (-5.2, "-5.2"),
            ("1000", "1000"),
        ]
        for pred, exp in test_cases:
            with self.subTest(pred=pred, exp=exp):
                self.assertTrue(evaluator.default_compare(pred, exp, "number"))

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_mismatches(self):
        evaluator = Evaluator.__new__(Evaluator)
        self.assertFalse(evaluator.default_compare(19, "20", "number"))
        self.assertFalse(evaluator.default_compare(0, "1", "number"))


class CategoryTypeTest(unittest.TestCase):
    """Tests for the 'category' answer type."""

    def test_normalization(self):
        self.assertEqual(normalize_result("Forbes"), "Forbes")
        self.assertEqual(normalize_result(""), "")

    def test_serialization(self):
        line, err = prediction_line({"status": "success", "predicted_answer": "United States"})
        self.assertEqual(line, "United States")
        self.assertIsNone(err)

        # Multi-line prediction should trigger FAILURE
        line_multi, err_multi = prediction_line({"status": "success", "predicted_answer": "Line 1\nLine 2"})
        self.assertEqual(line_multi, FAILURE)
        self.assertIn("line separator", err_multi)

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_matches(self):
        evaluator = Evaluator.__new__(Evaluator)
        test_cases = [
            ("United States", "United States"),
            ("Brazil", "Brazil"),
            ("Forbes", "Forbes"),
        ]
        for pred, exp in test_cases:
            with self.subTest(pred=pred, exp=exp):
                self.assertTrue(evaluator.default_compare(pred, exp, "category"))

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_mismatches(self):
        evaluator = Evaluator.__new__(Evaluator)
        # Categories are case-sensitive and must match exactly
        self.assertFalse(evaluator.default_compare("Male", "male", "category"))
        self.assertFalse(evaluator.default_compare("France", "Germany", "category"))


class ListCategoryTypeTest(unittest.TestCase):
    """Tests for the 'list[category]' answer type."""

    def test_normalization(self):
        res = normalize_result(["Apple", "Banana", "Cherry"])
        self.assertEqual(res, ["Apple", "Banana", "Cherry"])

        # Tuples converted to list
        res_tuple = normalize_result(("Apple", "Banana"))
        self.assertEqual(res_tuple, ["Apple", "Banana"])

    def test_serialization(self):
        line, err = prediction_line({"status": "success", "predicted_answer": ["A", "B"]})
        self.assertEqual(line, "['A', 'B']")
        self.assertIsNone(err)

        line_empty, err_empty = prediction_line({"status": "success", "predicted_answer": []})
        self.assertEqual(line_empty, "[]")
        self.assertIsNone(err_empty)

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_order_invariance(self):
        evaluator = Evaluator.__new__(Evaluator)
        # Order should not matter in list[category]
        self.assertTrue(evaluator.default_compare(["A", "B"], "['B', 'A']", "list[category]"))
        self.assertTrue(evaluator.default_compare("['B', 'A']", "['A', 'B']", "list[category]"))

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_mismatches(self):
        evaluator = Evaluator.__new__(Evaluator)
        self.assertFalse(evaluator.default_compare(["A", "B"], "['A', 'C']", "list[category]"))
        self.assertFalse(evaluator.default_compare([], "['A']", "list[category]"))


class ListNumberTypeTest(unittest.TestCase):
    """Tests for the 'list[number]' answer type."""

    def test_normalization(self):
        res = normalize_result([1, 2, 3.5])
        self.assertEqual(res, [1, 2, 3.5])

    def test_serialization(self):
        line, err = prediction_line({"status": "success", "predicted_answer": [10, 20]})
        self.assertEqual(line, "[10, 20]")
        self.assertIsNone(err)

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_matches(self):
        evaluator = Evaluator.__new__(Evaluator)
        self.assertTrue(evaluator.default_compare([1.0, 2.0], "[2, 1]", "list[number]"))
        self.assertTrue(evaluator.default_compare("[10, 20]", "[20.0, 10.0]", "list[number]"))

    @unittest.skipUnless(HAS_DATABENCH_EVAL, "databench_eval not installed")
    def test_comparison_mismatches(self):
        evaluator = Evaluator.__new__(Evaluator)
        self.assertFalse(evaluator.default_compare([1, 2], "[1, 3]", "list[number]"))


class ComplexStructuresTest(unittest.TestCase):
    """Tests for numpy/pandas complex structures conversion in sandbox_runner."""

    def test_numpy_primitives(self):
        try:
            import numpy as np
            self.assertIsInstance(normalize_result(np.int64(5)), int)
            self.assertIsInstance(normalize_result(np.float64(5.5)), float)
            self.assertIsInstance(normalize_result(np.bool_(True)), bool)
        except ImportError:
            self.skipTest("numpy not installed")

    def test_pandas_series_conversion(self):
        try:
            import pandas as pd
            s = pd.Series([10, 20, 30])
            self.assertEqual(normalize_result(s), [10, 20, 30])
        except ImportError:
            self.skipTest("pandas not installed")

    def test_nested_dictionary(self):
        nested = {"a": [1, 2], "b": {"c": 3}}
        self.assertEqual(normalize_result(nested), {"a": [1, 2], "b": {"c": 3}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
