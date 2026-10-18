from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from beacon.data import Bundle


def unit():
    from evaluation.trrust import consensus_spearman, exclude_overlap, regulator_tests
    targets = [f"T{i}" for i in range(40)]
    pairs = pd.DataFrame([(tf, t, "supported" if i < 6 else "low_effect") for tf in ("A", "B", "C") for i, t in enumerate(targets)],
                         columns=["source", "target", "analysis_role"])
    perfect = np.tile(np.arange(40, 0, -1, dtype=float), 3)
    result = regulator_tests(pairs, [perfect], permutations=99)
    assert result["macro_auroc"] == 1 and result["permutations_exceeding"] == 0 and result["blocked_permutation_p"] == 1 / 100
    assert regulator_tests(pairs, [perfect, -perfect], permutations=9)["macro_auroc"] == .5
    assert np.isclose(consensus_spearman(pairs, perfect), 1)
    shifted = perfect.copy()
    shifted[40:80] = np.arange(40, dtype=float)
    assert np.isclose(consensus_spearman(pairs.iloc[40:120].reset_index(drop=True), shifted[40:120]), -1)
    original = pd.DataFrame(dict(source_genesymbol=["A", "A", "C"], target_genesymbol=["T0", "T1", "T7"]))
    kept = exclude_overlap({"BEACON": pairs}, original)["BEACON"]
    assert set(kept.source) == {"B", "C"} and len(kept) == 80 and kept.analysis_role.eq("supported").sum() == 12
    print("TRRUST regulator-test, consensus and overlap-removal checks passed")


def section(bundle):
    from evaluation.seeds import TRRUST_SEEDS
    from evaluation.trrust import COMPARATORS, endpoints, training_graph
    trrust = bundle.manifest["trrust"]
    genes, evaluation = trrust["genes"], trrust["evaluation"]
    train = training_graph(bundle)
    positive, unlabeled = np.argwhere(train == 1), np.argwhere(train == 0)
    assert len(positive) == len(trrust["train_positive_edge_ids"]) == len(set(trrust["train_positive_edge_ids"])), "known edges are not unique"
    assert len(unlabeled) == len(trrust["train_unlabeled"]) == 5 * len(positive), "five unlabeled pairs per known edge"
    assert not np.diag(train != -1).any(), "self-pair in training"
    evaluated = {(r["source_index"], r["target_index"]) for r in evaluation}
    assert all(genes[s] == r["source"] and genes[t] == r["target"] for (s, t), r in zip(((r["source_index"], r["target_index"]) for r in evaluation), evaluation))
    masked = evaluated | {(t, s) for s, t in evaluated}
    assert not masked & set(map(tuple, np.argwhere(train != -1).tolist())), "an evaluation pair or its reverse is in training"
    regulators = trrust["regulators"]
    assert regulators == sorted(set(regulators)) and set(regulators) <= {r["source"] for r in evaluation}
    fits = sorted((e["condition"]["task"], e["condition"]["seed"]) for e in bundle.manifest["experiments"] if e["suite"] == "trrust")
    assert fits == [(i, t["seed"]) for i, t in enumerate(trrust["tasks"])] and tuple(t["seed"] for t in trrust["tasks"]) == TRRUST_SEEDS
    for name, key in COMPARATORS:
        indices = trrust["paired_grnboost2"] if key == "paired_grnboost2" else trrust["comparators"][key]
        assert [bundle.json(i)["method_seed"] for i in indices] == list(TRRUST_SEEDS), name
    result = endpoints(bundle)
    cohort = result["cohorts"].set_index("evaluation")
    assert cohort.loc["all", "regulators"] == len(regulators) and cohort.loc["all", "candidates"] == sum(r["source"] in set(regulators) for r in evaluation)
    assert result["ranking"].groupby("evaluation").method.nunique().eq(9).all()
    print(f"TRRUST section checks passed: {len(regulators)} regulators, {len(positive)} known edges, {len(unlabeled)} unlabeled pairs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--unit", action="store_true")
    args = p.parse_args()
    if args.unit:
        unit()
    else:
        section(Bundle(args.config))


if __name__ == "__main__":
    main()
