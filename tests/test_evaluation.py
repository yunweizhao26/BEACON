from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
KEY_NUMBERS = 1428

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
    fractional_ties()
    random_ap()
    single_table_queries()
    trrust_protocol()
    temporal_seed_table()
    seed_ensembles()
    signflip()
    print("Synthetic metric, zero-positive regulator, fractional tie, chance-level, query, TRRUST, temporal-table, seed-ensemble, sign-flip and bootstrap checks passed")

def seed_ensembles():
    from evaluation import seeds
    a, b, c = np.array([.1, .2], np.float32), np.array([.3, .2], np.float32), np.array([.5, .2], np.float32)
    mean = seeds.average([a, b, c])
    assert mean.dtype == np.float64 and np.array_equal(mean, (np.float64(a) + np.float64(b) + np.float64(c)) / 3)
    same = seeds.average([a, a.copy(), a.copy()])
    assert same.dtype == np.float32 and np.array_equal(same, a)
    edges = np.array([[0, 1], [0, 2]])
    combined = seeds.combine([dict(edges=edges, labels=np.array([1, 0]), beacon=v, degree_logistic=np.array([.4, .6])) for v in (a, b, c)])
    assert np.array_equal(combined["edges"], edges) and np.array_equal(combined["degree_logistic"], [.4, .6]) and np.allclose(combined["beacon"], [.3, .2])
    for broken in ([dict(edges=edges, beacon=a), dict(edges=edges[::-1], beacon=b)], [dict(edges=edges, beacon=a), dict(beacon=b, edges=edges)]):
        try:
            seeds.combine(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Misaligned seed fits accepted")
    class Fake:
        manifest = {"experiments": [dict(suite="fixed_pools", condition=dict(dataset=1, seed=s), setting=i) for i, s in enumerate((100, 42, 14))],
                    "comparators": [dict(suite="fixed_pools", method="gnnlink", dataset=1, seed=s) for s in (42, 14)]}
    assert [e["condition"]["seed"] for e in seeds.experiments(Fake, "fixed_pools", dataset=1)] == [42, 14, 100]
    for call in (lambda: seeds.comparators(Fake, "fixed_pools", "gnnlink", dataset=1), lambda: seeds.experiments(Fake, "fixed_pools", dataset=2)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete seed set accepted")

def signflip():
    from evaluation.responses import exact_signflip
    result = exact_signflip(np.array([1., 2., 3.]))
    assert result["patterns"] == 8 and np.isclose(result["p_greater"], 1 / 8) and np.isclose(result["p_two_sided"], 2 / 8)
    assert exact_signflip(np.zeros(4))["p_two_sided"] == 1

def fractional_ties():
    from evaluation.metrics import top_weights
    from evaluation.responses import joint_counts
    from evaluation.temporal import pooled_top
    values = np.array([5., 4., 4., 4., 1.])
    support = np.array([True, True, False, False, True])
    bound = np.array([False, True, True, False, True])
    counts = joint_counts(top_weights(values, 2), support, bound)
    assert np.isclose(counts["response"], 1 + 1 / 3) and np.isclose(counts["bound"], 2 / 3) and np.isclose(counts["both"], 1 / 3)
    assert np.isclose(sum(counts[k] for k in ("both", "neither", "response_only", "binding_only")), 2)
    order = np.array([0, 3, 2, 1, 4])
    permuted = joint_counts(top_weights(values[order], 2), support[order], bound[order])
    assert all(np.isclose(permuted[k], counts[k]) for k in counts)
    assert np.isclose(pooled_top(support, values, 2), pooled_top(support[order], values[order], 2)) and np.isclose(pooled_top(support, values, 2), 4 / 3)

def random_ap():
    from itertools import combinations
    from sklearn.metrics import average_precision_score
    from evaluation.metrics import expected_random_ap
    n = 7
    for m in range(n + 1):
        exact = [average_precision_score(np.isin(np.arange(n), chosen), -np.arange(n)) if m else 0. for chosen in combinations(range(n), m)]
        assert abs(np.mean(exact) - expected_random_ap(n, m)) < 1e-12, m
    rng = np.random.default_rng(0)
    y = np.zeros(60, bool)
    y[:4] = True
    simulated = np.array([average_precision_score(y, rng.random(len(y))) for _ in range(5000)])
    assert abs(simulated.mean() - expected_random_ap(60, 4)) < 4 * simulated.std() / np.sqrt(len(simulated))
    assert expected_random_ap(1, 1) == 1 and expected_random_ap(10, 0) == 0
    assert np.allclose(expected_random_ap([7, 7], [2, 7]), [expected_random_ap(7, 2), 1.])
    assert expected_random_ap(60, 4) > 4 / 60

def single_table_queries():
    import inspect
    import pandas as pd
    from evaluation import queries
    assert list(inspect.signature(queries.add).parameters) == ["section", "item", "context", "metric", "value"]
    queries.TABLES = {"summary/resources.csv": pd.DataFrame(dict(group=["a", "a", "b"], total_seconds=[1., 3., 5.], encoder_seconds=[1., 1., 1.],
                                                                gp_seconds=[2., 2., 2.], peak_rss_mb=[4., 6., 8.], peak_cuda_allocated_mb=[0., 0., 0.]))}
    queries.rows = []
    queries.resources()
    values = {(r["item"], r["metric"]): r["value"] for r in queries.rows}
    assert values[("a", "total_seconds")] == 2. and values[("b", "peak_rss_mb")] == 8. and len(queries.rows) == 10
    queries.TABLES, queries.rows = {}, []

def trrust_protocol():
    import pandas as pd
    from preparation.trrust import build, build_any_source
    genes = [f"G{i}" for i in range(12)]
    trrust = pd.DataFrame(dict(source=["G0"] * 5, target=["G1", "G2", "G3", "G4", "G5"], effect="Activation", pmid="1"))
    prior = pd.DataFrame(dict(source_genesymbol=["G6", "G6", "G7", "G8", "G9"], target_genesymbol=["G10", "G10", "G11", "G8", "G1"]))
    record, positive = build(genes, trrust, prior, ["G0", "G6", "G7", "G8", "G9"])
    unlabeled = np.asarray(record["train_unlabeled"])
    evaluated = {(r["source_index"], r["target_index"]) for r in record["evaluation"]}
    masked = evaluated | {(t, s) for s, t in evaluated}
    known = set(map(tuple, positive.tolist()))
    assert known == {(6, 10), (7, 11), (9, 1)} and len(positive) == 3 and len(record["train_positive_edge_ids"]) == 3
    assert len(unlabeled) == 15 and len(set(map(tuple, unlabeled.tolist()))) == 15
    assert set(unlabeled[:, 0].tolist()) <= {6, 7, 8, 9} and not (unlabeled[:, 0] == unlabeled[:, 1]).any()
    assert not set(map(tuple, unlabeled.tolist())) & (known | masked)
    published, published_positive = build_any_source(genes, trrust, prior)
    assert published["evaluation"] == record["evaluation"] and len(published_positive) == 5 and len(published["train_unlabeled"]) == 25

def temporal_seed_table():
    import pandas as pd
    from figures.temporal import CONTEXTS, CUTOFFS, seed_blocks
    rows = []
    for c in CUTOFFS:
        for x in CONTEXTS:
            for seed, shift in ((42, 0.), (14, .002), (100, .004)):
                rows.append(dict(cutoff=c, context=x, method="BEACON", seed=seed, pooled_ap=.03 + shift, mean_ap=.02, pooled_top1000=100.))
                rows.append(dict(cutoff=c, context=x, method="scRegNet", seed=seed, pooled_ap=.01 + shift, mean_ap=.03, pooled_top1000=150.))
            rows.append(dict(cutoff=c, context=x, method="scRegNet permuted embeddings", seed=42, pooled_ap=.005, mean_ap=.004, pooled_top1000=1234.))
    lines = seed_blocks(pd.DataFrame(rows))
    assert lines[1].startswith("BEACON, seed 42") and lines[1].count("0.0300") == 12 and lines[3].count("0.0340") == 12
    assert lines[4].startswith("scRegNet, seed 42") and "\\textbf" not in "".join(lines)
    assert lines[-1].startswith("Top 1,000") and lines[-1].count("1,234") == 12
    try:
        seed_blocks(pd.DataFrame(rows[1:]))
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete seed set accepted")

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
    if list(expected.columns) != [*keys, "value"] or len(expected) != KEY_NUMBERS:
        raise ValueError(f"Expected {KEY_NUMBERS:,} published key numbers with one value column")
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
