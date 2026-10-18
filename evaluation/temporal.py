import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from beacon.data import Bundle
from evaluation.metrics import binary_metrics, paired, top_weights
from evaluation import seeds
from preparation.temporal import CONTEXTS, CUTOFFS, references, saved

KS = (10, 25, 100)
LARGE = 50
SAVED = (("GNNLink", "gnnlink"), ("GCLink", "gclink"), ("scRegulate", "scregulate"), ("RegGAIN", "reggain"), ("GENELink", "genelink"),
         ("scRegNet", "scregnet"), ("GRNBoost2", "grnboost2"), ("GENIE3", "genie3"), ("Inferelator", "inferelator"), ("Degree baseline", "degree_logistic"))
VARIANTS = ("all later edges", f"supported by a paper citing <= {LARGE} edges", "two or more later papers")


def scores(bundle, entry, prior, tfs, expression, shape):
    section = bundle.manifest["temporal"]
    stored = dict(entry["scores"], **section["expression_only"][entry["context"]])
    result = {"BEACON": seeds.stored(bundle, stored["beacon"])}
    result.update({name: seeds.stored(bundle, stored[key]) for name, key in SAVED})
    z = expression - expression.mean(1, keepdims=True)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    result["Expression correlation"] = np.abs(z[tfs] @ z.T)
    result["Target popularity"] = np.broadcast_to(prior.sum(0).astype(float), shape)
    result["Regulator popularity"] = np.broadcast_to(prior[tfs].sum(1)[:, None].astype(float), shape)
    result["Target expression"] = np.broadcast_to(expression.mean(1), shape)
    return result


def pooled_top(y, v, k=1000):
    return float(top_weights(v, k) @ np.asarray(y, dtype=float))


SEEDED = (("BEACON", "beacon"), ("scRegNet", "scregnet"))
PERMUTED = "scRegNet permuted embeddings"


def seed_metrics(bundle):
    rows = []
    for entry in bundle.manifest["temporal"]["settings"]:
        labels = bundle.arrays(entry["labels"])
        positive = labels["later"]
        eligible = bundle.array(entry["discovery"]) & ~labels["undated"]
        keep = [i for i in range(len(positive)) if positive[i, eligible[i]].any()]
        fits = [(name, int(seed), record) for name, key in SEEDED for seed, record in entry["scores"][key].items()]
        fits.append((PERMUTED, 42, entry["scores"]["scregnet_permuted"]))
        for method, seed, record in fits:
            matrix = bundle.array(record)
            y, v = positive[eligible], matrix[eligible]
            ap = [binary_metrics(positive[i, eligible[i]].astype(int), matrix[i, eligible[i]])["average_precision"] for i in keep]
            rows.append(dict(cutoff=int(entry["cutoff"]), context=entry["context"], method=method, seed=seed,
                             pooled_ap=binary_metrics(y.astype(int), v)["average_precision"], mean_ap=float(np.mean(ap)),
                             pooled_top1000=pooled_top(y, v), unique_scores=int(len(np.unique(v)))))
    return pd.DataFrame(rows)


def restrictions(table, year, symbols, tfs, later, cutoff):
    edges = table.groupby(["source_genesymbol", "target_genesymbol"]).pmids.sum().map(lambda p: sorted(set(p)))
    size = pd.Series([m for p in edges for m in p]).value_counts()
    counts = pd.Series(symbols).value_counts()
    index = {s: i for i, s in enumerate(symbols) if counts[s] == 1}
    row = {g: r for r, g in enumerate(tfs)}
    small, replicated = np.zeros_like(later), np.zeros_like(later)
    for (source, target), pmids in edges.items():
        after = [m for m in pmids if year[m] > cutoff]
        if not after or source not in index or target not in index or index[source] not in row:
            continue
        i, j = row[index[source]], index[target]
        small[i, j] = any(size[m] <= LARGE for m in after)
        replicated[i, j] = len(after) >= 2
    return small, replicated


def evaluate(bundle, cutoff, context, table, year):
    entry = saved(bundle, context, cutoff)
    train = bundle.arrays(entry["training"])
    tfs, symbols, prior = train["tf_indices"], train["symbols"], train["train"] == 1
    labels = bundle.arrays(entry["labels"])
    later = labels["later"]
    mask = bundle.array(entry["discovery"]) & ~labels["undated"]
    expression = np.asarray(bundle.arrays(bundle.manifest["prepared"][context]["expression"])["expression"], dtype=np.float64)
    indegree, outdegree = prior.sum(0), prior[tfs].sum(1)
    methods = scores(bundle, entry, prior, tfs, expression, mask.shape)
    small, replicated = restrictions(table, year, symbols, tfs, later, cutoff)
    variants = dict(zip(VARIANTS, ((later, mask), (later & small, mask & ~(later & ~small)), (later & replicated, mask & ~(later & ~replicated)))))
    rows, comparisons = [], []
    for variant, (positive, eligible) in variants.items():
        per_tf = {}
        for method, matrix in methods.items():
            y, v = positive[eligible], matrix[eligible]
            pooled = binary_metrics(y.astype(int), v)
            keep = [i for i in range(len(tfs)) if positive[i, eligible[i]].any()]
            ap = np.array([binary_metrics(positive[i, eligible[i]].astype(int), matrix[i, eligible[i]])["average_precision"] for i in keep])
            hits = {k: np.array([top_weights(matrix[i, eligible[i]], k) @ positive[i, eligible[i]] / k for i in keep]) for k in KS}
            per_tf[method] = dict(ap=ap, **{f"top{k}": hits[k] for k in KS})
            rows.append(dict(cutoff=cutoff, context=context, variant=variant, method=method, positives=int(y.sum()), regulators=len(keep),
                             prevalence=float(y.mean()), pooled_ap=pooled["average_precision"], pooled_auprc=pooled["auprc_trapezoid"],
                             unique_scores=int(len(np.unique(v))), auroc=pooled["auroc"], mean_ap=float(ap.mean()),
                             **{f"top{k}": float(hits[k].mean()) for k in KS}, pooled_top1000=pooled_top(y, v)))
        indices = np.random.default_rng(42).integers(len(per_tf["BEACON"]["ap"]), size=(10000, len(per_tf["BEACON"]["ap"])))
        for method in per_tf:
            if method == "BEACON":
                continue
            for quantity in ("ap", *[f"top{k}" for k in KS]):
                result = paired(pd.Series(per_tf["BEACON"][quantity] - per_tf[method][quantity]), indices=indices)
                comparisons.append(dict(cutoff=cutoff, context=context, variant=variant, comparator=method, quantity=quantity,
                                        beacon_minus=result["mean"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                                        wins=result["wins"], losses=result["losses"]))
    later_in, later_out = (later & mask).sum(0), (later & mask).sum(1)
    top = outdegree >= np.quantile(outdegree, .9)
    concentration = dict(cutoff=cutoff, context=context, later_edges=int((later & mask).sum()), edges_per_tf=float(outdegree.mean()),
                         regulator_spearman=spearmanr(outdegree, later_out)[0], target_spearman=spearmanr(indegree, later_in)[0],
                         top10pct_regulator_share=float(later_out[top].sum() / later_out.sum()),
                         large_paper_only_share=float((later & mask & ~small).sum() / (later & mask).sum()))
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame([concentration])


def evaluate_setting(task):
    config, cutoff, context = task
    bundle = Bundle(config)
    table, year = references(bundle)
    return evaluate(bundle, cutoff, context, table, year)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cutoff", type=int, choices=CUTOFFS, action="append", help="Default: every cutoff")
    parser.add_argument("--context", choices=CONTEXTS, action="append", help="Default: every context")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Settings evaluated in parallel processes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.config)
    tasks = [(args.config, c, x) for c in args.cutoff or CUTOFFS for x in args.context or CONTEXTS]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            frames = list(pool.map(evaluate_setting, tasks))
    else:
        frames = [evaluate_setting(task) for task in tasks]
    for name, parts in zip(("metrics", "paired", "concentration"), zip(*frames)):
        pd.concat(parts, ignore_index=True).to_csv(args.out / f"{name}.csv", index=False)
    seed_metrics(bundle).to_csv(args.out / "seeds.csv", index=False)


if __name__ == "__main__":
    main()
