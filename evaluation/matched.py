import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from beacon.data import Bundle
from evaluation.metrics import binary_metrics, paired, top_weights
from evaluation.responses import CONTEXTS, joint_counts, label_bundle, score_matrices

BINS = 50
KS = (10, 25, 50, 100, 200)
METHODS = ("BEACON", "GNNLink", "RegGAIN", "Degree logistic", "Prior target in-degree", "Target expression")


def target_expression(bundle, context, genes):
    data = bundle.arrays(bundle.manifest["prepared"][context]["expression"])
    if "genes" in data and not np.array_equal(data["genes"], genes):
        raise ValueError("Expression axis differs from score matrix")
    values = data["expression"].mean(axis=1, dtype=np.float64)
    if values.shape != genes.shape:
        raise ValueError("Expression axis differs from score matrix")
    return values


def inputs(bundle, context):
    train, scores, discovery = score_matrices(bundle, context)
    genes, tfs = train["genes"], train["tf_indices"]
    expression = target_expression(bundle, context, genes)
    scores = {("Degree logistic" if m == "degree_logistic" else m): v for m, v in scores.items()}
    scores = {m: scores[m] for m in METHODS if m in scores}
    scores["Target expression"] = np.broadcast_to(expression, discovery.shape)
    return train, scores, discovery, expression, label_bundle(bundle, context, genes, genes[tfs])


def enrichment(bundle, context):
    train, scores, discovery, expression, labels = inputs(bundle, context)
    mask = discovery[labels["selected"]] & labels["tested"]
    rows, control = [], []
    for i, tf in enumerate(labels["tf_genes"]):
        eligible = mask[i]
        support = labels["support"][i, eligible].astype(float)
        level = expression[eligible]
        rank = np.argsort(np.argsort(level, kind="stable"), kind="stable")
        bins = rank * BINS // len(level)
        rate = np.bincount(bins, support, BINS) / np.maximum(np.bincount(bins, minlength=BINS), 1)
        for method, matrix in scores.items():
            values = matrix[labels["selected"][i], eligible]
            for k in KS:
                weights = top_weights(values, k)
                rows.append(dict(condition=context, tf=tf, method=method, k=k,
                                 observed=float(weights @ support / k), expected=float(weights @ rate[bins] / k)))
        control.append(dict(condition=context, assay="response", tf=tf, method="Target expression",
                            **binary_metrics(support.astype(int), level)))
        for method in ("BEACON", "GNNLink"):
            control.append(dict(condition=context, assay="expression_correlation", tf=tf, method=method,
                                spearman=float(spearmanr(scores[method][labels["selected"][i], eligible], level)[0])))
    if context == "k562":
        control += binding_control(bundle, train, discovery, expression, labels)
    return pd.DataFrame(rows), pd.DataFrame(control)


def binding_control(bundle, train, discovery, expression, response):
    data = bundle.arrays(bundle.manifest["prepared"]["binding"])
    genes, tfs = train["genes"], train["tf_indices"]
    lookup = {str(g): i for i, g in enumerate(response["tf_genes"])}
    rows = []
    for i in np.flatnonzero(data["assayed"]):
        tf = genes[tfs[i]]
        allowed = discovery[i] & data["annotated"]
        outcomes = [("binding", allowed, data["primary"][i])]
        if str(tf) in lookup:
            j = lookup[str(tf)]
            outcomes.append(("binding_and_response", allowed & response["tested"][j], data["primary"][i] & response["support"][j]))
        for assay, eligible, support in outcomes:
            rows.append(dict(condition="k562", assay=assay, tf=tf, method="Target expression",
                             **binary_metrics(support[eligible], expression[eligible])))
    i, = [i for i in np.flatnonzero(data["assayed"]) if train["symbols"][tfs[i]] == "DNMT1"]
    j = lookup[str(genes[tfs[i]])]
    idx = np.flatnonzero(discovery[i] & data["annotated"] & response["tested"][j])
    counts = joint_counts(top_weights(expression[idx], 100), response["support"][j][idx], data["primary"][i][idx])
    rows.append(dict(condition="k562", assay="dnmt1_top100", tf=str(genes[tfs[i]]), method="Target expression",
                     **{key: counts[key] for key in ("response", "bound", "both")}))
    return rows


def summarize(frame, seed=42):
    summary, comparisons = [], []
    for (condition, k), part in frame.groupby(["condition", "k"], sort=False):
        observed = part.pivot(index="tf", columns="method", values="observed")
        expected = part.pivot(index="tf", columns="method", values="expected")
        indices = np.random.default_rng(seed).integers(len(observed), size=(10000, len(observed)))
        excess = observed - expected
        for method in observed.columns:
            o, e = observed[method].to_numpy(), expected[method].to_numpy()
            low, high = np.quantile(o[indices].mean(axis=1) / e[indices].mean(axis=1), [.025, .975])
            summary.append(dict(condition=condition, k=k, method=method, tfs=len(o), observed=o.mean(), expected=e.mean(),
                                ratio=o.mean() / e.mean(), ratio_ci_low=low, ratio_ci_high=high))
            if method != "BEACON":
                result = paired(excess.BEACON - excess[method], indices=indices)
                comparisons.append(dict(condition=condition, k=k, comparator=method, beacon_minus_comparator_excess=result["mean"],
                                        ci_low=result["ci_low"], ci_high=result["ci_high"], beacon_wins=result["wins"],
                                        comparator_wins=result["losses"], ties=result["ties"], tfs=len(o)))
    return pd.DataFrame(summary), pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.config)
    frames, controls = zip(*(enrichment(bundle, context) for context in CONTEXTS))
    frame, control = pd.concat(frames, ignore_index=True), pd.concat(controls, ignore_index=True)
    summary, comparisons = summarize(frame)
    frame.to_csv(args.out / "per_tf.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    comparisons.to_csv(args.out / "paired.csv", index=False)
    control.to_csv(args.out / "target_expression.csv", index=False)


if __name__ == "__main__":
    main()
