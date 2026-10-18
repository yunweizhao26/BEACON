import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from beacon.data import Bundle
from evaluation.matched import CONTEXTS, inputs
from evaluation.metrics import expected_random_ap
from evaluation import seeds

FIXED = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
COVERAGES = (.05, .1, .2, .4, .8)
SPLITS = (42, 14, 100)


def fixed_experiment(bundle, dataset, split_seed, coverage, control="beacon"):
    found, = [e for e in bundle.manifest["experiments"] if e["suite"] == "fixed_pools" and e["condition"]["dataset"] == dataset
              and e["condition"]["split_seed"] == split_seed and e["condition"]["coverage"] == coverage
              and e["condition"]["control"] == control and e["condition"]["seed"] == 42]
    return found


def fixed_predictions(bundle, dataset, split_seed, coverage, control="beacon", role="predictions"):
    fits = seeds.experiments(bundle, "fixed_pools", dataset=dataset, split_seed=split_seed, coverage=coverage, control=control)
    return seeds.fit_arrays(bundle, fits, role)


def fixed_pool(bundle, dataset, split_seed, coverage):
    split = bundle.arrays(fixed_experiment(bundle, dataset, split_seed, coverage)["reference"]["split"])
    predictions = fixed_predictions(bundle, dataset, split_seed, coverage)
    edges, labels = predictions["edges"], predictions["labels"].astype(bool)
    scores = {"BEACON": predictions["beacon"], "Topology control": predictions["degree_logistic"]}
    scores.update(comparator_scores(bundle, dataset, split_seed, coverage, edges))
    revealed = set(map(tuple, split["supplied_positive_edges"]))
    reverse = np.array([(target, source) in revealed for source, target in edges])
    return edges, labels, scores, reverse, len(revealed) / len(split["tf_indices"])


COMPARATORS = {"gnnlink": "GNNLink", "gclink": "GCLink", "scregulate": "scRegulate", "reggain": "RegGAIN", "genelink": "GENELink",
               "scregnet": "scRegNet", "grnboost2": "GRNBoost2", "genie3": "GENIE3", "inferelator": "Inferelator"}
EXPRESSION_ONLY = ("grnboost2", "genie3")


def aligned(pairs, values, edges, missing=None):
    order = {tuple(edge): i for i, edge in enumerate(pairs)}
    if missing is None:
        return np.asarray(values)[np.array([order[tuple(edge)] for edge in edges])]
    return np.array([values[order[tuple(e)]] if tuple(e) in order else missing for e in edges], dtype=float)


def comparator_scores(bundle, dataset, split_seed, coverage, edges, methods=COMPARATORS):
    scores = {}
    entries = {(c["method"], c.get("split_seed")) for c in bundle.manifest["comparators"] if c.get("suite") == "fixed_pools"
               and c.get("dataset") == dataset and c.get("coverage") == coverage and c.get("method") in methods}
    for method, split in sorted(entries, key=lambda item: list(COMPARATORS).index(item[0])):
        if method in EXPRESSION_ONLY:
            data = seeds.comparator_arrays(bundle, "fixed_pools", method, dataset=dataset, coverage=coverage)
            scores[methods[method]] = aligned(data["eligible_edges"], data["eligible_scores"], edges, missing=0.)
        elif split == split_seed:
            data = seeds.comparator_arrays(bundle, "fixed_pools", method, dataset=dataset, coverage=coverage, split_seed=split_seed)
            scores[methods[method]] = aligned(data["edges"], data[method], edges)
    return scores


def consensus_frame(edges, labels, scores):
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


RANDOM_DRAWS = 100


def external_statistics(matrix, selected, mask, support):

    if len(selected) < 2:
        raise ValueError("A leave-one-out consensus needs at least two regulators")
    ranks = np.vstack([rankdata(matrix[i]) for i in selected])
    total = ranks.sum(axis=0)
    others = len(selected) - 1
    agreement, residual_ap = np.empty(len(selected)), np.full(len(selected), np.nan)
    for k in range(len(selected)):
        consensus = total - ranks[k]
        agreement[k] = spearmanr(ranks[k][mask[k]], consensus[mask[k]])[0]
        y = support[k, mask[k]]
        if y.any():
            residual_ap[k] = average_precision_score(y, (others * ranks[k] - consensus)[mask[k]])
    return agreement, residual_ap


def external(bundle, seed=42, *, details=False, draws=RANDOM_DRAWS):

    rows, regulator_rows, strata = [], [], []
    for context in CONTEXTS:
        train, scores, discovery, _, labels = inputs(bundle, context)
        selected = labels["selected"]
        mask = discovery[selected] & labels["tested"]
        support = labels["support"]
        tfs = train["tf_indices"]
        counts = (train["train"][tfs] == 1).sum(axis=1)
        per_regulator = float(counts.sum() / len(tfs))
        positives = np.array([support[k, mask[k]].sum() for k in range(len(selected))])
        eligible = positives > 0
        chance = np.full(len(selected), np.nan)
        chance[eligible] = expected_random_ap(mask.sum(axis=1)[eligible], positives[eligible])
        bins = np.array([prior_bin(counts[i]) for i in selected])
        rng = np.random.default_rng(seed)
        shown = {"BEACON": [scores["BEACON"]], "GNNLink": [scores["GNNLink"]], "Topology control": [scores["Degree logistic"]],
                 "Random": (rng.random(scores["BEACON"].shape) for _ in range(draws))}
        for method, matrices in shown.items():
            results = [external_statistics(matrix, selected, mask, support) for matrix in matrices]
            agreement, residual = (np.vstack(values) for values in zip(*results))
            rows.append(dict(context=context, method=method, regulators=len(selected), revealed_per_regulator=per_regulator, draws=len(results),
                             consensus_spearman=float(np.mean(np.median(agreement, axis=1))),
                             residual_ap_over_chance=float(np.mean(np.nanmean(residual[:, eligible], axis=1) / chance[eligible].mean()))))
            if not details:
                continue
            for k, i in enumerate(selected):
                regulator_rows.append(dict(context=context, method=method, source=int(tfs[i]),
                    tf=str(train["genes"][tfs[i]]), prior_edges=int(counts[i]), prior_bin=bins[k],
                    evaluated_pairs=int(mask[k].sum()), positives=int(positives[k]), consensus_spearman=float(agreement[:, k].mean()),
                    residual_ap=float(residual[:, k].mean()), chance=float(chance[k]), ap_eligible=bool(eligible[k])))
            for name in BINS:
                part, scored = bins == name, (bins == name) & eligible
                ap, level = (float(np.mean(residual[:, scored].mean(axis=1))), float(chance[scored].mean())) if scored.any() else (np.nan, np.nan)
                strata.append(dict(context=context, method=method, prior_bin=name, regulators=int(part.sum()),
                                   finite_spearman_regulators=int(np.isfinite(agreement[0, part]).sum()),
                                   ap_eligible_regulators=int(scored.sum()),
                                   consensus_spearman=float(np.mean(np.median(agreement[:, part], axis=1))) if part.any() else np.nan,
                                   residual_ap=ap, chance=level,
                                   residual_ap_over_chance=float(np.mean(residual[:, scored].mean(axis=1) / level)) if scored.any() else np.nan))
    frame = pd.DataFrame(rows)
    if not details:
        return frame
    return frame, pd.DataFrame(regulator_rows), pd.DataFrame(strata)


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
