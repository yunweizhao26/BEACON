"""Regulator specificity from saved, aligned predictions.

Fixed pools use average-tie percentile ranks within each regulator's full test
pool. The target consensus averages ranks over OTHER regulators containing that
target, before any exclusion. Both APs then exclude every pair whose reverse
was revealed and include only regulators with both label classes. Gain is the
ratio of mean APs; unequal target pools need not give exactly one for identical
target order. A singleton target retains its own rank and is counted explicitly.

External assays retain their published mean-rank residual estimator. Prior
strata use each evaluated regulator's own prior count; consensus is computed
over all evaluated regulators before stratification.
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


def fixed_experiment(bundle, dataset, seed, coverage):
    """The reported BEACON fit of one fixed-pool setting."""
    found, = [e for e in bundle.manifest["experiments"] if e["suite"] == "fixed_pools" and e["condition"]["dataset"] == dataset
              and e["condition"]["split_seed"] == seed and e["condition"]["coverage"] == coverage and e["condition"]["ratio"] == 5
              and e["condition"]["corruption"] == 0 and e["condition"]["control"] == "beacon" and not e["condition"]["snn_weight"]]
    return found


def fixed_pool(bundle, dataset, seed, coverage):
    """Test pairs, labels, scores, reverse-edge flags and revealed edges per regulator for one fixed-pool fit."""
    found = fixed_experiment(bundle, dataset, seed, coverage)
    split = bundle.arrays(found["reference"]["split"])
    predictions = bundle.arrays(found["reference"]["predictions"])
    edges, labels = predictions["edges"], predictions["labels"].astype(bool)
    scores = {"BEACON": predictions["beacon"], "Topology control": predictions["degree_logistic"]}
    scores.update(comparator_scores(bundle, dataset, seed, coverage, edges))
    revealed = set(map(tuple, split["supplied_positive_edges"]))
    reverse = np.array([(target, source) in revealed for source, target in edges])
    return edges, labels, scores, reverse, len(revealed) / len(split["tf_indices"])


COMPARATORS = {"gnnlink": "GNNLink", "gclink": "GCLink", "scregulate": "scRegulate", "reggain": "RegGAIN", "genelink": "GENELink",
               "scregnet": "scRegNet", "grnboost2": "GRNBoost2", "genie3": "GENIE3", "inferelator": "Inferelator"}
EXPRESSION_ONLY = ("grnboost2", "genie3")


def comparator_scores(bundle, dataset, seed, coverage, edges):
    """Saved comparator scores on the test pairs of one fit. Expression-only networks ignore the prior and the split,
    so their saved full-network scores serve every split at the coverage where they were run; absent pairs score zero."""
    scores = {}
    for entry in bundle.manifest["comparators"]:
        method = entry.get("method")
        if entry.get("suite") != "fixed_pools" or entry.get("dataset") != dataset or method not in COMPARATORS or entry.get("coverage") != coverage:
            continue
        if method not in EXPRESSION_ONLY and entry.get("split_seed") != seed:
            continue
        data = bundle.arrays(entry["artifacts"]["predictions"])
        if method in EXPRESSION_ONLY:
            order = {tuple(edge): i for i, edge in enumerate(data["eligible_edges"])}
            scores[COMPARATORS[method]] = np.array([data["eligible_scores"][order[tuple(e)]] if tuple(e) in order else 0. for e in edges])
            continue
        pairs, values = (data["test_edges"], data["test_reggain"]) if method == "reggain" else (data["edges"], data[method])
        order = {tuple(edge): i for i, edge in enumerate(pairs)}
        scores[COMPARATORS[method]] = values[np.array([order[tuple(edge)] for edge in edges])]
    return scores


def consensus_frame(edges, labels, scores):
    """Label-independent, full-test-pool percentile consensus, leaving self out."""
    edges, labels, scores = np.asarray(edges), np.asarray(labels, bool), np.asarray(scores)
    if edges.shape != (len(labels), 2) or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected aligned edges, labels and finite scores")
    frame = pd.DataFrame(dict(source=edges[:, 0], target=edges[:, 1], label=labels, score=scores))
    if frame.duplicated(["source", "target"]).any():
        raise ValueError("Duplicate regulator-target pair")
    frame["percentile"] = frame.groupby("source").score.rank(method="average", pct=True)
    totals = frame.groupby("target").percentile.transform("sum")
    counts = frame.groupby("target").percentile.transform("size")
    frame["fallback"] = counts.eq(1)
    frame["consensus"] = np.where(counts > 1, (totals - frame.percentile) / np.maximum(counts - 1, 1), totals / counts)
    return frame


def evaluate_frame(frame, reverse=None):
    """Both APs use the same symmetric-retained pairs and two-class regulators."""
    labels = frame.label.to_numpy(dtype=bool)
    reverse = np.zeros(len(frame), bool) if reverse is None else np.asarray(reverse, bool)
    if reverse.shape != labels.shape:
        raise ValueError("Reverse flags do not match the full test pool")
    keep = ~reverse
    rows = []
    for source, part in frame.loc[keep].groupby("source"):
        if part.label.all() or not part.label.any():
            continue
        rows.append(dict(source=int(source), pairs=len(part), positives=int(part.label.sum()),
                         fallback_pairs=int(part.fallback.sum()),
                         within_ap=average_precision_score(part.label, part.score),
                         consensus_ap=average_precision_score(part.label, part.consensus)))
    detail = pd.DataFrame(rows, columns=["source", "pairs", "positives", "fallback_pairs", "within_ap", "consensus_ap"])
    within, consensus = detail[["within_ap", "consensus_ap"]].mean()
    result = dict(within_ap=float(within), target_only_ap=float(consensus), gain=float(within / consensus),
                  eligible_regulators=len(detail), full_pairs=len(frame), evaluated_pairs=int(keep.sum()),
                  eligible_pairs=int(detail.pairs.sum()), excluded_pairs=int(reverse.sum()),
                  excluded_positives=int((reverse & labels).sum()), excluded_unlabeled=int((reverse & ~labels).sum()),
                  fallback_full_pairs=int(frame.fallback.sum()),
                  fallback_evaluated_pairs=int(frame.loc[keep, "fallback"].sum()),
                  fallback_eligible_pairs=int(detail.fallback_pairs.sum()))
    return result, detail


def specific_gain(edges, labels, scores, reverse=None):
    """Within and consensus AP, accepting the FULL test pool before exclusion."""
    result, _ = evaluate_frame(consensus_frame(edges, labels, scores), reverse)
    return result["within_ap"], result["target_only_ap"]


def fixed_pools(bundle, *, details=False, null=False):
    rows, regulator_rows, validation = [], [], []
    for dataset, context in FIXED.items():
        for coverage in COVERAGES:
            for seed in SPLITS:
                edges, labels, scores, reverse, per_regulator = fixed_pool(bundle, dataset, seed, coverage)
                key = dict(context=context, coverage=coverage, split=seed)
                for method, values in scores.items():
                    result, detail = evaluate_frame(consensus_frame(edges, labels, values), reverse)
                    rows.append(dict(**key, method=method, revealed_per_regulator=per_regulator, **result))
                    if details:
                        regulator_rows.extend(dict(**key, method=method, **r) for r in detail.to_dict("records"))
                if null:
                    sources, ri = np.unique(edges[:, 0], return_inverse=True)
                    targets, ti = np.unique(edges[:, 1], return_inverse=True)
                    for replicate in range(3):
                        # Frozen entropy for the paper's identical-order simulations.
                        rng = np.random.default_rng(np.random.SeedSequence([20261015, dataset, int(coverage * 100), seed, replicate]))
                        target_score = rng.uniform(-1, 1, len(targets))
                        offsets = rng.normal(0, 100, len(sources))
                        scales = rng.lognormal(0, 1, len(sources))
                        result, _ = evaluate_frame(consensus_frame(edges, labels, offsets[ri] + scales[ri] * target_score[ti]), reverse)
                        validation.append(dict(**key, replicate=replicate, **result))
    fixed = pd.DataFrame(rows)
    if details or null:
        return fixed, pd.DataFrame(regulator_rows), pd.DataFrame(validation)
    return fixed


BINS = ["0", "1-2", "3-5", "6-10", "11+"]


def prior_bin(count):
    return BINS[0 if count == 0 else 1 if count <= 2 else 2 if count <= 5 else 3 if count <= 10 else 4]


def external(bundle, seed=42, *, details=False):
    rows, regulator_rows = [], []
    for context in CONTEXTS:
        train, scores, discovery, _, labels = inputs(bundle, context)
        selected = labels["selected"]
        mask = discovery[selected] & labels["tested"]
        tfs = train["tf_indices"]
        counts = (train["train"][tfs] == 1).sum(axis=1)
        per_regulator = float(counts.sum() / len(tfs))
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
                value = float(average_precision_score(y, residual[k][mask[k]])) if y.any() else np.nan
                prevalence = float(y.mean()) if y.any() else np.nan
                if y.any():
                    ap.append(value)
                    chance.append(prevalence)
                if details:
                    i = selected[k]
                    regulator_rows.append(dict(context=context, method=method, source=int(tfs[i]),
                        tf=str(train["genes"][tfs[i]]), prior_edges=int(counts[i]), prior_bin=prior_bin(counts[i]),
                        evaluated_pairs=int(mask[k].sum()), positives=int(y.sum()), consensus_spearman=float(agreement[k]),
                        residual_ap=value, chance=prevalence, ap_eligible=bool(y.any())))
            rows.append(dict(context=context, method=method, regulators=len(selected), revealed_per_regulator=per_regulator,
                             consensus_spearman=float(np.median(agreement)), residual_ap_over_chance=float(np.mean(ap) / np.mean(chance))))
    frame = pd.DataFrame(rows)
    if not details:
        return frame
    detail = pd.DataFrame(regulator_rows)
    strata = []
    for (context, method), group in detail.groupby(["context", "method"], sort=False):
        for name in BINS:
            part = group[group.prior_bin.eq(name)]
            eligible = part[part.ap_eligible]
            ap, chance = eligible.residual_ap.mean(), eligible.chance.mean()
            strata.append(dict(context=context, method=method, prior_bin=name, regulators=len(part),
                               finite_spearman_regulators=int(np.isfinite(part.consensus_spearman).sum()),
                               ap_eligible_regulators=len(eligible),
                               consensus_spearman=float(np.median(part.consensus_spearman)) if len(part) else np.nan,
                               residual_ap=ap, chance=chance, residual_ap_over_chance=ap / chance))
    return frame, detail, pd.DataFrame(strata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.config)
    fixed, regulators, null = fixed_pools(bundle, details=True, null=True)
    fixed.to_csv(args.out / "fixed_pools.csv", index=False)
    regulators.to_csv(args.out / "fixed_per_regulator.csv", index=False)
    null.to_csv(args.out / "identical_order.csv", index=False)
    frame, detail, strata = external(bundle, details=True)
    frame.to_csv(args.out / "external.csv", index=False)
    detail.to_csv(args.out / "external_per_regulator.csv", index=False)
    strata.to_csv(args.out / "external_strata.csv", index=False)


if __name__ == "__main__":
    main()
