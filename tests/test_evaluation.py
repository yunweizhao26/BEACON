"""Rescore saved evidence and compare all in-scope numeric rows at atol=1e-9."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE

def unit():
    from evaluation.metrics import metrics, binary_metrics, paired, top_weights, all_tf_ap, match_pairs
    from evaluation.probabilities import selective_report
    labels = np.array([1, 0, 1, 0])
    scores = np.ones(4)
    value = binary_metrics(labels, scores)
    assert value["average_precision"] == .5 and value["auprc_trapezoid"] == .75
    assert np.array_equal(top_weights(scores, 2), np.full(4, .5))
    edges = np.array([[0, 1], [0, 2], [3, 1], [3, 2]])
    assert all_tf_ap(edges, np.array([1, 0, 0, 0]), scores) == .25
    indices = np.array([[0, 1], [1, 0]])
    assert paired(np.array([.1, -.1]), indices=indices)["mean"] == 0
    a = selective_report(labels, scores * .3, scores)
    b = selective_report(labels[::-1], scores * .3, scores)
    assert a == b
    from preparation.labels import response_labels
    schema = dict(tested=np.ones((1, 1), bool), evaluation_mask=np.ones((1, 1), bool),
                  tf_genes=np.array(["regulator"]), target_genes=np.array(["target"]), constructs=np.array(["construct"]))
    support, calibration = response_labels(schema, np.array([[.03, .04, .05]]), np.zeros((1, 3)),
        np.ones((200, 3)), np.zeros((200, 3)), np.ones((200, 3), bool),
        work=["construct"], family_genes=["target", "second", "third"], genes=["target", "second", "third"])
    assert calibration["passed"] and not support["response"].any()
    assert np.isclose(support["adjusted_pvalues"][0, 0], .05)
    print("Synthetic metric, zero-positive regulator, fractional tie, and bootstrap checks passed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=RELEASE / "results" / "evaluation_test")
    parser.add_argument("--unit", action="store_true")
    args = parser.parse_args()
    if args.unit:
        unit()
        return
    bundle = Bundle(args.config)
    command = [sys.executable, "-B", "-m", "evaluation.report", "--out", str(args.out), "--responses"]
    if args.config:
        command += ["--config", str(args.config.resolve())]
    subprocess.run(command, cwd=RELEASE, check=True)
    expected = bundle.frame(bundle.manifest["tables"]["key_numbers"])
    actual = pd.read_csv(args.out / "key_numbers.csv")
    keys = ["section", "item", "context", "metric"]
    for frame in (expected, actual):
        frame[keys] = frame[keys].fillna("")
    if list(expected.columns) != [*keys, "value"] or len(expected) != 1528:
        raise ValueError("Expected 1,528 published key numbers with one value column")
    expected = expected.set_index(keys)
    actual = actual.set_index(keys)
    if not expected.index.is_unique or not actual.index.is_unique:
        raise ValueError("Duplicate paper-number keys")
    if bundle.manifest["key_number_column"] != "value":
        raise ValueError("Expected the value column")
    checks = []
    for key, row in expected.iterrows():
        target = row["value"]
        value = actual.loc[key, "value"] if key in actual.index else np.nan
        passed = bool(key in actual.index and (pd.isna(target) and pd.isna(value) or pd.notna(target) and pd.notna(value) and abs(target-value) <= 1e-9))
        checks.append(dict(zip(keys, key), expected=None if pd.isna(target) else float(target), actual=None if pd.isna(value) else float(value),
                           passed=passed, present=key in actual.index))
    status = "passed" if all(r["passed"] for r in checks) else "failed"
    receipt = dict(status=status, tolerance=1e-9, relative_tolerance=0, rows=checks,
                   bundle_manifest_sha256=bundle.manifest_sha256)
    (args.out / "test_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if status != "passed":
        raise SystemExit(f"Evaluation failed for {sum(not r['passed'] for r in checks)} rows; see test_manifest.json")

if __name__ == "__main__":
    main()
