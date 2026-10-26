"""Evaluation integrity tests: no API key, Gemini requests or Docker required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_baseline as evaluation


def question(index, kind="number", answer="1", table="001_Test"):
    return {"index": index, "dataset": table, "question": f"Question {index}",
            "type": kind, "answer": answer, "sample_answer": answer}


def record(row, mode="full", success=True, prediction=1):
    return {"index": row["index"], "dataset": row["dataset"], "question": row["question"],
            "answer_type": row["type"], "mode": mode,
            "expected_answer": row["sample_answer" if mode == "lite" else "answer"],
            "status": "success" if success else "error", "predicted_answer": prediction if success else None,
            "correct": success, "error": None if success else "Docker execution failed", "attempts": 1,
            "generated_code": f"result = {prediction!r}"}


class SelectionTest(unittest.TestCase):
    def test_balanced_repeatable_diverse_selection_and_smoke_prefix(self):
        rows = []
        for kind in evaluation.TYPES:
            for table in range(25):
                for _ in range(2):
                    rows.append(question(len(rows), kind, table=f"{table:03d}_Table"))
        rows.append(question(len(rows), answer=None))
        selected, excluded = evaluation.select_questions(rows)
        again, _ = evaluation.select_questions(rows)
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({row["index"] for row in selected}), 100)
        self.assertEqual(evaluation.Counter(row["type"] for row in selected[:10]),
                         {kind: 2 for kind in evaluation.TYPES})
        for kind in evaluation.TYPES:
            self.assertEqual(len({row["dataset"] for row in selected if row["type"] == kind}), 20)
        self.assertEqual(len(excluded), 1)

    def test_shortage_and_invalid_count_fail(self):
        with self.assertRaisesRegex(ValueError, "Not enough eligible"):
            evaluation.select_questions([question(0)])
        with self.assertRaisesRegex(ValueError, "multiple of five"):
            evaluation.select_questions([], 9)


class IntegrityTest(unittest.TestCase):
    def test_duplicates_missing_wrong_mode_and_metadata_fail(self):
        selected = [question(0), question(1)]
        first = record(selected[0])
        for records in ([first, first], [first], [first, record(selected[1], "lite")],
                        [first, {**record(selected[1]), "expected_answer": "wrong"}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                evaluation.validate_records(records, selected, "full")

    def test_serialization_preserves_false_zero_empty_lists_and_duplicates(self):
        for value, expected in ((False, "False"), (0, "0"), ([], "[]"), ([1, 1], "[1, 1]"),
                                (["é", "é"], "['é', 'é']"), ("", ""), (None, "None")):
            with self.subTest(value=value):
                line, error = evaluation.prediction_line({"status": "success", "predicted_answer": value})
                self.assertEqual(line, expected)
                self.assertIsNone(error)
        line, error = evaluation.prediction_line({"status": "success", "predicted_answer": "a\nb"})
        self.assertEqual(line, evaluation.FAILURE)
        self.assertIsNotNone(error)
        self.assertEqual(evaluation.prediction_line({"status": "error"})[0], evaluation.FAILURE)

    def test_metrics_penalize_processing_failures(self):
        records = [record(question(i), success=i == 0) for i in range(10)]
        result = evaluation.summarize(records, [True] + [False] * 9, [None] * 10)
        self.assertEqual(result["overall_accuracy"], 10)
        self.assertEqual(result["conditional_accuracy"], 100)
        self.assertEqual(result["processing_failure_rate"], 90)


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.selected = [question(i, evaluation.TYPES[i % 5],
                                  "True" if i % 5 == 0 else "[1, 1]" if i % 5 in (3, 4) else "1")
                         for i in range(10)]
        evaluation.write_json(self.directory / "selected_qa.json", self.selected)
        self.config = {"count": 10, "seed": 42, "purpose": "Synthetic test only",
                       "selection_sha256": evaluation.sha256(self.directory / "selected_qa.json"),
                       "packages": evaluation.versions(), "dataset": {"fingerprint": "test"}}
        evaluation.write_json(self.directory / "config.json", self.config)
        for mode in ("full", "lite"):
            for row in reversed(self.selected):
                evaluation.append_jsonl(self.directory / f"results_{mode}.jsonl",
                                        record(row, mode, success=row["index"] == 0, prediction=True))

    def test_official_evaluator_file_roundtrip_and_repeat_score(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = evaluation.score(self.directory)
            before = {path.name: path.read_bytes() for path in (self.directory / "scores").iterdir()}
            second = evaluation.score(self.directory)
        self.assertEqual(first, second)
        after = {path.name: path.read_bytes() for path in (self.directory / "scores").iterdir()}
        self.assertEqual(before, after)
        for mode in ("full", "lite"):
            self.assertEqual(first["modes"][mode]["official_accuracy"], 10)
            self.assertEqual(first["modes"][mode]["conditional_accuracy"], 100)
        lines = (self.directory / "scores" / "predictions.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["True"] + [evaluation.FAILURE] * 9)

    def test_missing_results_fail_before_export(self):
        (self.directory / "results_lite.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            evaluation.score(self.directory)
        self.assertFalse((self.directory / "scores").exists())

    def test_failure_marker_collision_is_rejected(self):
        self.selected[0]["answer"] = evaluation.FAILURE
        self.selected[0]["type"] = "category"
        evaluation.write_json(self.directory / "selected_qa.json", self.selected)
        self.config["selection_sha256"] = evaluation.sha256(self.directory / "selected_qa.json")
        evaluation.write_json(self.directory / "config.json", self.config)
        for mode in ("full", "lite"):
            path = self.directory / f"results_{mode}.jsonl"
            path.unlink()
            for row in self.selected:
                evaluation.append_jsonl(path, record(row, mode, success=False))
        with self.assertRaisesRegex(ValueError, "marker matches"):
            evaluation.score(self.directory)

    def test_resume_completed_records_never_initializes_api_or_docker(self):
        self.config.update(baseline_sha256=evaluation.baseline_hashes(), python=evaluation.sys.version,
                           table_sha256={})
        evaluation.write_json(self.directory / "config.json", self.config)
        with patch.object(evaluation, "command_output") as command, contextlib.redirect_stdout(io.StringIO()):
            evaluation.run(self.directory)
        command.assert_not_called()

    def test_changed_selection_or_baseline_cannot_be_resumed(self):
        self.config.update(baseline_sha256={"wrong": "hash"})
        evaluation.write_json(self.directory / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluation.run(self.directory)
        (self.directory / "selected_qa.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified"):
            evaluation.load_experiment(self.directory)

    def test_manual_review_is_preserved_and_validated(self):
        evaluation.write_json(self.directory / "error_reviews.json", [
            {"index": 1, "mode": "full", "category": "execution", "explanation": "Reviewed log: generic failure only."}])
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluation.score(self.directory)
        self.assertEqual(result["reviewed_errors"], 1)
        errors = evaluation.read_json(self.directory / "scores" / "errors.json")
        self.assertTrue(next(row for row in errors if row["index"] == 1 and row["mode"] == "full")["reviewed"])


if __name__ == "__main__":
    unittest.main()
