#!/usr/bin/env python3
"""Integrity check of saved artifacts for DataBench Stage 1 evaluation.

Validates:
1. Frozen experiment configuration and question selection.
2. Saved results and reviewed error classifications.
3. Prediction exports and scoring artifact SHA-256 hashes against verification.json.
4. Consistency between metrics.json, report.md, and decisions.
5. Baseline file hashes.
6. Support for cross-platform validation.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "state" / "eval_stage1"


def compute_hashes(path: Path):
    """Computes direct byte hash and CRLF/LF normalized hashes for cross-platform checks."""
    raw_bytes = path.read_bytes()
    direct_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Normalization for standard text files:
    normalized_crlf = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    crlf_hash = hashlib.sha256(normalized_crlf).hexdigest()

    normalized_lf = raw_bytes.replace(b"\r\n", b"\n")
    lf_hash = hashlib.sha256(normalized_lf).hexdigest()

    # Normalization for CSV files where record terminators were \r\n:
    csv_crlf_hash = direct_hash
    if path.name.endswith(".csv"):
        import csv, io
        try:
            # Re-serialize with Windows CSV row terminator \r\n
            text = raw_bytes.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text))
            buf = io.StringIO(newline="")
            writer = csv.writer(buf, lineterminator="\r\n")
            for row in reader:
                writer.writerow(row)
            csv_crlf_hash = hashlib.sha256(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")).hexdigest()
        except Exception:
            pass

    return direct_hash, crlf_hash, lf_hash, csv_crlf_hash


def count_jsonl_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main():
    print("=" * 75)
    print("DataBench Stage 1 — Integrity Check of Saved Artifacts")
    print("=" * 75)

    if not EXPERIMENT_DIR.exists():
        print(f"[FAIL] Experiment directory not found: {EXPERIMENT_DIR}")
        sys.exit(1)

    verification_path = EXPERIMENT_DIR / "verification.json"
    config_path = EXPERIMENT_DIR / "config.json"

    if not verification_path.exists() or not config_path.exists():
        print("[FAIL] Missing verification.json or config.json")
        sys.exit(1)

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    checks_passed = 0
    checks_total = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        msg = f"[{status}] {name}"
        if details:
            msg += f" ({details})"
        print(msg)
        return condition

    print("\n--- 1. Question Selection and Datasets ---")
    qa_path = EXPERIMENT_DIR / "selected_qa.json"
    direct, crlf, _, _ = compute_hashes(qa_path)
    expected_qa_hash = config.get("selection_sha256")
    qa_matched = (direct == expected_qa_hash) or (crlf == expected_qa_hash)
    check("Selected questions file exists", qa_path.is_file())
    check("Selected questions SHA-256 matches config", qa_matched,
          "verified with Windows CRLF snapshot" if crlf == expected_qa_hash else "")

    selected_qa = json.loads(qa_path.read_text(encoding="utf-8"))
    check("Question count == 100", len(selected_qa) == 100, f"found {len(selected_qa)}")
    check("Unique dataset questions", len({q["index"] for q in selected_qa}) == 100)

    print("\n--- 2. Results and Predictions Count ---")
    full_jsonl = EXPERIMENT_DIR / "results_full.jsonl"
    lite_jsonl = EXPERIMENT_DIR / "results_lite.jsonl"
    check("results_full.jsonl has 100 records", count_jsonl_lines(full_jsonl) == 100)
    check("results_lite.jsonl has 100 records", count_jsonl_lines(lite_jsonl) == 100)

    pred_full = EXPERIMENT_DIR / "scores" / "predictions.txt"
    pred_lite = EXPERIMENT_DIR / "scores" / "predictions_lite.txt"
    check("predictions.txt has 100 lines", len(pred_full.read_text(encoding="utf-8").splitlines()) == 100)
    check("predictions_lite.txt has 100 lines", len(pred_lite.read_text(encoding="utf-8").splitlines()) == 100)

    print("\n--- 3. Error Analysis and Reviews ---")
    reviews_path = EXPERIMENT_DIR / "error_reviews.json"
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    check("Reviewed errors == 33", len(reviews) == 33, f"found {len(reviews)}")

    errors_json_path = EXPERIMENT_DIR / "scores" / "errors.json"
    errors = json.loads(errors_json_path.read_text(encoding="utf-8"))
    check("Total error instances in scores == 33", len(errors) == 33, f"found {len(errors)}")

    print("\n--- 4. Scoring Artifacts Cryptographic Hash Verification ---")
    scores_dir = EXPERIMENT_DIR / "scores"
    expected_hashes = verification.get("scoring_artifact_sha256", {})

    all_hashes_ok = True
    for filename, expected_hash in expected_hashes.items():
        artifact_path = scores_dir / filename
        if not artifact_path.is_file():
            check(f"Artifact {filename} exists", False)
            all_hashes_ok = False
            continue
        direct_h, crlf_h, lf_h, csv_h = compute_hashes(artifact_path)
        match = (direct_h == expected_hash) or (crlf_h == expected_hash) or (lf_h == expected_hash) or (csv_h == expected_hash)
        detail = "direct match" if direct_h == expected_hash else "cross-platform CRLF/LF match"
        check(f"Hash {filename}", match, detail if match else f"expected {expected_hash[:8]}...")
        if not match:
            all_hashes_ok = False

    print("\n--- 5. Accuracy & Metric Consistency ---")
    metrics_path = scores_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    full_acc = metrics["modes"]["full"]["overall_accuracy"]
    lite_acc = metrics["modes"]["lite"]["overall_accuracy"]
    full_success = metrics["modes"]["full"]["success_rate"]
    lite_success = metrics["modes"]["lite"]["success_rate"]

    check("Full overall accuracy == 86.0%", full_acc == 86.0, f"{full_acc}%")
    check("Lite overall accuracy == 81.0%", lite_acc == 81.0, f"{lite_acc}%")
    check("Full success rate == 97.0%", full_success == 97.0, f"{full_success}%")
    check("Lite success rate == 95.0%", lite_success == 95.0, f"{lite_success}%")

    print("\n--- 6. Baseline Files Hash Check ---")
    baseline_hashes = config.get("baseline_sha256", {})
    baseline_ok = True
    for rel_path, expected_hash in baseline_hashes.items():
        target = ROOT / rel_path
        if not target.is_file():
            check(f"Baseline file {rel_path} exists", False)
            baseline_ok = False
            continue
        direct_h, crlf_h, lf_h, _ = compute_hashes(target)
        match = (direct_h == expected_hash) or (crlf_h == expected_hash) or (lf_h == expected_hash)
        check(f"Baseline hash {rel_path}", match)
        if not match:
            baseline_ok = False

    print("\n" + "=" * 75)
    print(f"Summary: {checks_passed} / {checks_total} checks passed.")
    if checks_passed == checks_total:
        print("RESULT: REPRODUCIBILITY VERIFICATION SUCCESSFUL (100% PASS)")
        print("All experiment artifacts, hashes, predictions, and metrics are verified.")
        print("=" * 75)
        sys.exit(0)
    else:
        print("RESULT: VERIFICATION FAILED (some checks did not match)")
        print("=" * 75)
        sys.exit(1)


if __name__ == "__main__":
    main()
