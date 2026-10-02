"""Sixteen L40S prediction cases, plus a separate inducing metric check."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import hashlib
import os
import platform
from importlib.metadata import version
import numpy as np
from beacon.data import Bundle, RELEASE

def compare_arrays(expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {"exact": False, "reason": "shape_or_dtype", "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape), "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype)}
    exact = expected.tobytes() == actual.tobytes()
    if not np.issubdtype(expected.dtype, np.number):
        return {"exact": exact, "differing": int(np.count_nonzero(expected != actual))}
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
    return {"exact": exact, "differing": int(np.count_nonzero(expected != actual)),
            "maximum_absolute_difference": float(np.max(difference, initial=0)),
            "maximum_relative_difference": float(np.max(difference / denominator, initial=0))}

def scientific_log(log):
    fields = {"fallback", "validation_pairs", "validation_positives", "labeled_pairs", "labeled_positives",
              "e_star", "training_pairs", "best_epoch", "stop_epoch", "early_stopping", "validation_checks",
              "step1_selection", "best_validation_ap", "max_epochs", "step1_training_pairs", "epochs",
              "stage", "mode", "selection_embeddings"}
    if isinstance(log, list):
        return [scientific_log(value) for value in log]
    if isinstance(log, dict):
        kept = {key: scientific_log(value) for key, value in log.items() if key in fields or key in ("encoder", "fits", "epoch", "validation_ap")}
        return kept
    return log


def unit():
    from beacon.model import Training, ENCODER_EPOCHS, GP_MAX_EPOCHS, GP_FALLBACK_EPOCHS
    from beacon.features import feature_control
    from beacon.pairs import make_split
    from experiments.expression import transform, control_expression
    from experiments.fixed_pools import fit
    import inspect
    import torch
    assert (ENCODER_EPOCHS, GP_MAX_EPOCHS, GP_FALLBACK_EPOCHS) == (100, 200, 25)
    x = np.full((40, 40), -1, dtype=np.int8)
    x.flat[:99] = 1
    x.flat[99:900] = 0
    selected = Training().internal_split(x)
    assert not selected["info"]["fallback"] and selected["info"]["validation_positives"] == 10
    x.flat[90:99] = 0
    assert Training().internal_split(x)["info"]["fallback"]
    features = np.arange(80, dtype=np.float64).reshape(20, 4)
    assert transform(features, "random") is features
    assert np.array_equal(control_expression(features, "random")[0],
                          np.random.default_rng(2718).choice(features.ravel(), size=features.shape, replace=True))
    assert np.array_equal(feature_control(features, random_features=True), np.random.default_rng(2718).normal(size=features.shape).astype(np.float32))
    assert np.array_equal(feature_control(features, permuted_features=True), features[np.random.default_rng(2718).permutation(20)])
    for name in ("snn_weight", "random_features", "permuted_features", "without_encoder", "components", "pca", "inducing_points", "diagnostics", "decoder", "logistic", "nnpu"):
        assert inspect.signature(fit).parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
    training = Training(snn_weight=1)
    encoder = training.encoder_class(64, 16)
    assert [layer.out_features for layer in encoder.encoder if isinstance(layer, torch.nn.Linear)] == [256, 256, 16]
    assert isinstance(encoder.encoder[-1], torch.nn.LayerNorm)
    truth = np.zeros((80, 80), dtype=np.int8)
    truth[np.repeat(np.arange(5), 8), np.tile(np.arange(10, 18), 5)] = 1
    a, b = make_split(truth, np.arange(5), 42, .05, 5), make_split(truth, np.arange(5), 42, .8, 5)
    assert np.array_equal(a["test"], b["test"]) and np.array_equal(a["valid"], b["valid"])
    for control in ("gene_permuted", "cell_shuffled", "random"):
        assert transform(np.asfortranarray(features), control).flags.f_contiguous
    assert not compare_arrays(np.array([0.]), np.array([-0.]))["exact"]
    print("Synthetic branch and bit-comparison checks passed")

def metric_equal(expected, actual, path=""):
    errors = []
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, dict):
            errors.extend(metric_equal(other, value, path + "/" + key))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if other is None or not np.isclose(other, value, rtol=0, atol=1e-9, equal_nan=True):
                errors.append({"metric": path + "/" + key, "expected": other, "actual": value})
        elif value != other:
            errors.append({"metric": path + "/" + key, "expected": other, "actual": value})
    return errors

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, default=RELEASE / "results" / "equivalence")
    p.add_argument("--unit", action="store_true")
    args = p.parse_args()
    if args.unit:
        unit()
        return
    import torch
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name():
        raise RuntimeError("Prediction identity requires NVIDIA L40S")
    bundle = Bundle(args.config)
    bundle.verify()
    environment = dict(python=platform.python_version(), torch_cuda=torch.version.cuda,
        packages={name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gpytorch", "linear-operator")},
        threads={name: os.getenv(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    release_hash = hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [e for e in bundle.manifest["experiments"] if e["identity"]]
    if len(cases) != 16:
        raise ValueError("Expected exactly 16 authorized identity cases")
    records = []
    for experiment in cases:
        record = {"case": experiment["setting"], "suite": experiment["suite"], "condition": experiment["condition"]}
        current = args.out / "current"
        command = [sys.executable, "-B", "-m", "experiments.run", "--case", str(record["case"]), "--out", str(current)]
        if args.config:
            command += ["--config", str(args.config.resolve())]
        try:
            subprocess.run(command, cwd=RELEASE, check=True)
            checks = {}
            primary = "beacon" if experiment["suite"] in ("external", "rpe1") else "predictions"
            if not (current / (primary + ".npz")).is_file():
                raise ValueError("The fit did not produce its primary predictions")
            if experiment["suite"] == "trrust":
                reference = bundle.json(experiment["reference"]["record"])["records"]
                with np.load(current / "predictions.npz", allow_pickle=False) as actual:
                    keys = list(zip(actual["source"], actual["target"], actual["analysis_role"]))
                    by_key = {(r["source"], r["target"], r["analysis_role"]): r for r in reference}
                    if len(keys) != len(set(keys)) or set(keys) != set(by_key):
                        raise ValueError("TRRUST record axes are not unique and identical")
                    checks["probability"] = compare_arrays(np.array([by_key[k]["probability"] for k in keys]), actual["probability"])
            else:
                for path in sorted(current.glob("*.npz")):
                    role = path.stem
                    if role not in experiment["reference"]:
                        raise ValueError(f"Missing reference role: {role}")
                    expected = bundle.arrays(experiment["reference"][role])
                    with np.load(path, allow_pickle=False) as actual:
                        if set(expected) != set(actual.files):
                            raise ValueError(f"Array key mismatch: {role}")
                        for key in expected:
                            checks[role + "/" + key] = compare_arrays(expected[key], actual[key])
            log = json.loads((current / "manifest.json").read_text())["training_log"]
            if "training_log" in experiment["reference"]:
                expected_log = bundle.json(experiment["reference"]["training_log"])
                checks["training_log"] = {"exact": scientific_log(expected_log) == scientific_log(log)}
            record.update(status="exact" if all(c["exact"] for c in checks.values()) else "failed", checks=checks)
        except Exception as error:
            record.update(status="failed", error=str(error))
        records.append(record)
        (args.out / "manifest.json").write_text(json.dumps({"status": "running", "cases": records}, indent=2) + "\n")
        if current.exists():
            shutil.rmtree(current)
    # Raw inducing predictions were never saved; this is explicitly metric-only.
    from experiments.sensitivity import inducing
    metric_case, = [e for e in bundle.manifest["experiments"] if e["metric_only"]]
    try:
        actual = inducing(bundle, metric_case)
        expected = bundle.json(metric_case["reference"]["record"])["fits"]
        errors = metric_equal(expected, actual)
        metric_check = {"status": "metric_only" if not errors else "failed", "errors": errors}
    except Exception as error:
        metric_check = {"status": "failed", "error": str(error)}
    status = "exact_minimal_gate" if all(r["status"] == "exact" for r in records) and metric_check["status"] == "metric_only" else "failed"
    (args.out / "manifest.json").write_text(json.dumps({"status": status, "gpu": torch.cuda.get_device_name(), "environment": environment,
        "release_manifest_sha256": release_hash, "bundle_manifest_sha256": bundle.manifest_sha256, "cases": records,
        "inducing": metric_check, "scope": "16 settings only; no full-corpus identity claim"}, indent=2) + "\n")
    if status == "failed":
        raise SystemExit("Identity failed; see maximum differences and errors in manifest.json")

if __name__ == "__main__":
    main()
