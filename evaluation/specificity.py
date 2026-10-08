"""Regulator specificity of target rankings.

A ranking is regulator-specific when it orders a regulator's candidate targets differently from other regulators.
In the fixed pools, the specific gain divides each method's within-regulator AP by the AP of a ranking that gives
every target its mean score over the test pool, which ignores the regulator. Held-out edges whose reverse was
revealed are excluded. In the external assays, each regulator's target ranking is compared with the consensus
ranking across regulators, and the regulator-specific residual is scored against the response labels.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from beacon.data import Bundle
from evaluation.matched import CONTEXTS, inputs

FIXED = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
COVERAGES = (.05, .1, .2, .4, .8)
SPLITS = (42, 14, 100)


def fixed_pool(bundle, dataset, seed, coverage):
    """Test pairs, labels, scores, reverse-edge flags and revealed edges per regulator for one fixed-pool fit."""
    manifest = bundle.manifest
    found, = [e for e in manifest["experiments"] if e["suite"] == "fixed_pools" and e["condition"]["dataset"] == dataset
              and e["condition"]["split_seed"] == seed and e["condition"]["coverage"] == coverage and e["condition"]["ratio"] == 5
              and e["condition"]["corruption"] == 0 and e["condition"]["control"] == "beacon" and not e["condition"]["snn_weight"]]
    split = bundle.arrays(found["reference"]["split"])
    predictions = bundle.arrays(found["reference"]["predictions"])
    edges, labels = predictions["edges"], predictions["labels"].astype(bool)
    scores = {"BEACON": predictions["beacon"], "Topology control": predictions["degree_logistic"]}
    comparator, = [c for c in manifest["comparators"] if c["suite"] == "fixed_pools" and c.get("dataset") == dataset
                   and c["split_seed"] == seed and c["coverage"] == coverage and c["method"] == "gnnlink"]
    gnnlink = bundle.arrays(comparator["artifacts"]["predictions"])
    order = {tuple(edge): i for i, edge in enumerate(gnnlink["edges"])}
    scores["GNNLink"] = gnnlink["gnnlink"][np.array([order[tuple(edge)] for edge in edges])]
    revealed = set(map(tuple, split["supplied_positive_edges"]))
    reverse = np.array([(target, source) in revealed for source, target in edges])
    return edges, labels, scores, reverse, len(revealed) / len(split["tf_indices"])


def specific_gain(edges, labels, scores):
    frame = pd.DataFrame(dict(source=edges[:, 0], target=edges[:, 1], label=labels, score=scores))
    frame["target_mean"] = frame.groupby("target").score.transform("mean")
    within, target_only = [], []
    for _, part in frame.groupby("source"):
        if part.label.all() or not part.label.any():
            continue
        within.append(average_precision_score(part.label, part.score))
        target_only.append(average_precision_score(part.label, part.target_mean))
    return float(np.mean(within)), float(np.mean(target_only))


def fixed_pools(bundle):
    rows = []
    for dataset, context in FIXED.items():
        for coverage in COVERAGES:
            for seed in SPLITS:
                edges, labels, scores, reverse, per_regulator = fixed_pool(bundle, dataset, seed, coverage)
                keep = ~(labels & reverse)
                for method, values in scores.items():
                    within, target_only = specific_gain(edges[keep], labels[keep], values[keep])
                    rows.append(dict(context=context, coverage=coverage, split=seed, method=method, revealed_per_regulator=per_regulator,
                                     within_ap=within, target_only_ap=target_only, gain=within / target_only))
    return pd.DataFrame(rows)


def external(bundle, seed=42):
    rows = []
    for context in CONTEXTS:
        train, scores, discovery, _, labels = inputs(bundle, context)
        selected = labels["selected"]
        mask = discovery[selected] & labels["tested"]
        tfs = train["tf_indices"]
        per_regulator = float((train["train"][tfs] == 1).sum() / len(tfs))
        shown = {"BEACON": scores["BEACON"], "GNNLink": scores["GNNLink"], "Topology control": scores["Degree logistic"],
                 "Random": np.random.default_rng(seed).random(scores["BEACON"].shape)}
        for method, matrix in shown.items():
            ranks = np.vstack([rankdata(matrix[i]) for i in selected]) / matrix.shape[1]
            consensus = ranks.mean(axis=0)
            agreement = [spearmanr(ranks[k][mask[k]], consensus[mask[k]])[0] for k in range(len(selected))]
            residual = ranks - consensus
            ap, chance = [], []
            for k in range(len(selected)):
                y = labels["support"][k, mask[k]]
                if y.any():
                    ap.append(average_precision_score(y, residual[k][mask[k]]))
                    chance.append(y.mean())
            rows.append(dict(context=context, method=method, regulators=len(selected), revealed_per_regulator=per_regulator,
                             consensus_spearman=float(np.median(agreement)), residual_ap_over_chance=float(np.mean(ap) / np.mean(chance))))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.config)
    fixed_pools(bundle).to_csv(args.out / "fixed_pools.csv", index=False)
    external(bundle).to_csv(args.out / "external.csv", index=False)


if __name__ == "__main__":
    main()
