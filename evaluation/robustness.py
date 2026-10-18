import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from beacon.data import Bundle
from evaluation import seeds
from evaluation.metrics import pool_metrics, popularity
from evaluation.specificity import consensus_frame, evaluate_frame

PARTS = ("reverse_blocked", "ten_splits", "references")
SEEDED = {"beacon": "BEACON", "gnnlink": "GNNLink", "genelink": "GENELink", "gclink": "GCLink", "scregnet": "scRegNet"}
DEGREE, TARGET, REGULATOR = "Degree baseline", "Target popularity", "Regulator popularity"
METHODS = ("BEACON", "GNNLink", "GENELink", "GCLink", "scRegNet", DEGREE, TARGET, REGULATOR)
METRICS = {"pooled_auprc": "Pooled AUPRC", "pooled_ap": "Pooled AP", "all_tf_ap": "All-TF mean AP",
           "within_ap": "Within-regulator AP", "ratio": "Specificity ratio"}
LEAD = ("pooled_auprc", "pooled_ap", "all_tf_ap", "within_ap")
TESTED = ("pooled_ap", "all_tf_ap", "within_ap")
TESTED_METHODS = ("GNNLink", DEGREE)
KEYS = ("part", "protocol", "dataset", "reference", "panel", "split_seed", "coverage")
REFERENCES = {"STRING": "STRING", "Non-Specific": "non-specific ChIP-seq", "Specific": "cell-type-specific ChIP-seq",
              "Lofgof": "LOF/GOF"}


def label(reference, panel):
    cell, genes = panel.rsplit(" ", 1)
    return f"{cell}, {REFERENCES.get(reference, reference)}, TFs+{genes}"


def known_matrix(known_edges, edges):
    known_edges, edges = np.asarray(known_edges, dtype=int).reshape(-1, 2), np.asarray(edges, dtype=int)
    size = int(max(edges.max(), known_edges.max() if len(known_edges) else 0)) + 1
    matrix = np.full((size, size), -1, dtype=np.int8)
    matrix[tuple(known_edges.T)] = 1
    return matrix


def baselines(known_edges, edges):
    values = popularity(known_matrix(known_edges, edges), edges)
    return {TARGET: values["Prior target in-degree"].astype(np.float64), REGULATOR: values["Prior regulator out-degree"].astype(np.float64)}


def reverse_known(known_edges, edges):
    edges = np.asarray(edges, dtype=int)
    return known_matrix(known_edges, edges)[edges[:, 1], edges[:, 0]] == 1


def fit_metrics(edges, labels, scores, reverse):
    edges, labels, scores = np.asarray(edges), np.asarray(labels).astype(int), np.asarray(scores)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores aligned with the test pool")
    pooled = pool_metrics(labels, scores, edges)
    result, detail = evaluate_frame(consensus_frame(edges, labels.astype(bool), scores), reverse)
    row = dict(pooled_auprc=pooled["auprc_trapezoid"], pooled_ap=pooled["average_precision"], auroc=pooled["auroc"],
               all_tf_ap=pooled["all_tf_ap"], within_ap=result["within_ap"], consensus_ap=result["target_only_ap"],
               ratio=result["gain"], test_pairs=len(labels), test_positives=int(labels.sum()),
               all_tf_regulators=int(len(np.unique(edges[:, 0]))), within_regulators=result["eligible_regulators"],
               within_pairs=result["eligible_pairs"], reverse_excluded_pairs=result["excluded_pairs"],
               reverse_excluded_positives=result["excluded_positives"])
    return row, detail


def evaluate_setting(key, edges, labels, known_edges, members, degree, details=False):

    reverse = reverse_known(known_edges, edges)
    scores = {name: (seeds.average(values), "seed mean") for name, values in members.items()}
    scores[DEGREE] = (np.asarray(degree), "deterministic")
    scores.update({name: (values, "deterministic") for name, values in baselines(known_edges, edges).items()})
    rows, regulators = [], []
    for name in [m for m in METHODS if m in scores]:
        values, scoring = scores[name]
        row, detail = fit_metrics(edges, labels, values, reverse)
        rows.append(dict(key, method=name, scoring=scoring, **row))
        if details:
            regulators.extend(dict(key, method=name, **r) for r in detail.to_dict("records"))
    return (rows, regulators) if details else rows


def sign_flip_p(delta):
    delta = np.asarray(delta, float)
    signs = np.array(list(itertools.product((1, -1), repeat=len(delta))))
    return float(np.mean(np.abs(signs @ delta) >= abs(delta.sum()) - 1e-12))


def t_interval(delta):
    delta = np.asarray(delta, float)
    if len(delta) < 2:
        raise ValueError("A t interval needs at least two splits")
    mean, sd = delta.mean(), delta.std(ddof=1)
    half = stats.t.ppf(.975, len(delta) - 1) * sd / np.sqrt(len(delta))
    return float(mean), float(mean - half), float(mean + half)


def split_verdict(delta):
    delta = np.asarray(delta, float)
    return "holds" if (delta > 0).all() else "reverses" if (delta < 0).all() else "ties"


def interval_verdict(low, high):
    return "holds" if low > 0 else "reverses" if high < 0 else "ties"


def paired(per_fit, intervals=("ten_splits",)):

    out = []
    for (part, dataset, coverage), group in per_fit.groupby(["part", "dataset", "coverage"], sort=False):
        table = group.pivot_table(index="split_seed", columns="method", values=list(METRICS), aggfunc="first")
        present = set(table.columns.get_level_values(1))
        for method in [m for m in METHODS[1:] if m in present]:
            for metric in METRICS:
                pair = table[metric][["BEACON", method]].dropna()
                delta = (pair["BEACON"] - pair[method]).to_numpy()
                row = dict(part=part, dataset=dataset, coverage=coverage, method=method, metric=metric, splits=len(delta),
                           beacon_mean=pair["BEACON"].mean(), method_mean=pair[method].mean(), difference_mean=delta.mean(),
                           beacon_higher=int((delta > 0).sum()), beacon_lower=int((delta < 0).sum()))
                if part in intervals:
                    _, low, high = t_interval(delta)
                    row.update(t_low=low, t_high=high, beacon_sd=pair["BEACON"].std(ddof=1), method_sd=pair[method].std(ddof=1),
                               sign_flip_p=sign_flip_p(delta) if method in TESTED_METHODS and metric in TESTED else np.nan,
                               verdict=interval_verdict(low, high))
                else:
                    row["verdict"] = split_verdict(delta)
                out.append(row)
    columns = ["part", "dataset", "coverage", "method", "metric", "splits", "beacon_mean", "method_mean", "difference_mean",
               "beacon_higher", "beacon_lower", "verdict", "t_low", "t_high", "beacon_sd", "method_sd", "sign_flip_p"]
    return pd.DataFrame(out).reindex(columns=columns)


def summary(per_fit):
    frame = per_fit.groupby(["part", "dataset", "coverage", "method"], sort=False)[list(METRICS)].agg(["mean", "std", "min", "max"])
    frame.columns = [f"{metric}_{statistic}" for metric, statistic in frame.columns]
    counts = per_fit.groupby(["part", "dataset", "coverage", "method"], sort=False).size().rename("splits")
    return frame.join(counts).reset_index()


def section(bundle):
    value = bundle.manifest.get("robustness")
    if not value:
        raise ValueError("The bundle has no robustness section")
    if tuple(value.get("seeds", ())) != seeds.SEEDS:
        raise ValueError(f"Robustness scores must use optimization seeds {seeds.SEEDS}")
    return value


def aligned(pairs, values, edges, pair_labels, labels):
    pairs, values = np.asarray(pairs), np.asarray(values)
    if not np.array_equal(pairs, edges):
        order = {tuple(edge): i for i, edge in enumerate(pairs.tolist())}
        if len(order) != len(pairs) or len(pairs) != len(edges):
            raise ValueError("Score pairs differ from the test pool")
        index = np.array([order[tuple(edge)] for edge in np.asarray(edges).tolist()])
        pairs, values, pair_labels = pairs[index], values[index], np.asarray(pair_labels)[index]
    if not np.array_equal(np.asarray(pair_labels).astype(int), np.asarray(labels).astype(int)):
        raise ValueError("Score labels differ from the test pool")
    return values


def metadata(bundle, dataset):
    entry = next(d for d in bundle.manifest["datasets"] if d["dataset"] == dataset)
    return entry["reference"], entry["panel"]


def own_inputs(bundle, fit):
    split = bundle.arrays(fit["split"])
    edges, labels = split["test_edges"], split["test_labels"]
    members, degree = {}, None
    for method, records in fit["scores"].items():
        if set(records) != {str(seed) for seed in seeds.SEEDS}:
            raise ValueError(f"Expected seeds {seeds.SEEDS} for {method}: {sorted(records)}")
        arrays = [bundle.arrays(records[str(seed)]) for seed in seeds.SEEDS]
        members[SEEDED[method]] = [aligned(a["edges"], a[method], edges, a["labels"], labels) for a in arrays]
        if method == "beacon":
            degree = seeds.average([aligned(a["edges"], a["degree_logistic"], edges, a["labels"], labels) for a in arrays])
    if degree is None:
        raise ValueError("A robustness setting needs BEACON fits")
    return edges, labels, split["supplied_positive_edges"], members, degree


def shared_entries(bundle, item):
    condition = dict(dataset=item["dataset"], coverage=item["coverage"], split_seed=item["split_seed"])
    suite = item.get("suite", "fixed_pools")
    fits = seeds.experiments(bundle, suite, control="beacon", **condition)
    comparators = {method: seeds.comparators(bundle, suite, method, **condition)
                   for method in item.get("methods", ("beacon", "gnnlink")) if method != "beacon"}
    return fits, comparators


def shared_inputs(bundle, item):
    fits, comparators = shared_entries(bundle, item)
    split = bundle.arrays(fits[0]["reference"]["split"])
    edges = np.argwhere(split["test"] != -1)
    labels = split["test"][tuple(edges.T)]
    known = np.argwhere(split["train"] == 1)
    predictions = [bundle.arrays(fit["reference"]["predictions"]) for fit in fits]
    members = {"BEACON": [aligned(p["edges"], p["beacon"], edges, p["labels"], labels) for p in predictions]}
    degree = seeds.average([aligned(p["edges"], p["degree_logistic"], edges, p["labels"], labels) for p in predictions])
    for method, entries in comparators.items():
        arrays = [bundle.arrays(entry["artifacts"]["predictions"]) for entry in entries]
        members[SEEDED[method]] = [aligned(a["edges"], a[method], edges, a["labels"], labels) for a in arrays]
    return edges, labels, known, members, degree


def records(bundle):
    value = section(bundle)
    found = {value["settings"]}
    for fit in value["fits"]:
        found.add(fit["split"])
        found.update(record for by_seed in fit["scores"].values() for record in by_seed.values())
    for item in value.get("shared", ()):
        fits, comparators = shared_entries(bundle, item)
        found.update([fits[0]["reference"]["split"], *(fit["reference"]["predictions"] for fit in fits)])
        found.update(entry["artifacts"]["predictions"] for entries in comparators.values() for entry in entries)
    return found


def settings(bundle):
    value = section(bundle)
    found = [(fit, own_inputs) for fit in value["fits"]] + [(item, shared_inputs) for item in value.get("shared", ())]
    keys = {(entry["part"], entry["dataset"], entry["coverage"], entry["split_seed"]) for entry, _ in found}
    if len(keys) != len(found):
        raise ValueError("Duplicate robustness setting")
    order = {part: i for i, part in enumerate(PARTS)}
    datasets = list(dict.fromkeys(entry["dataset"] for entry, _ in found))
    return sorted(found, key=lambda x: (order[x[0]["part"]], datasets.index(x[0]["dataset"]), x[0]["coverage"], x[0]["split_seed"]))


def per_fit_metrics(bundle, details=False):
    rows, regulators = [], []
    for entry, loader in settings(bundle):
        reference, panel = metadata(bundle, entry["dataset"])
        key = dict(part=entry["part"], protocol=entry["protocol"], dataset=entry["dataset"], reference=reference, panel=panel,
                   split_seed=entry["split_seed"], coverage=entry["coverage"])
        edges, labels, known, members, degree = loader(bundle, entry)
        result = evaluate_setting(key, edges, labels, known, members, degree, details=details)
        if details:
            rows.extend(result[0])
            regulators.extend(result[1])
        else:
            rows.extend(result)
    frame = pd.DataFrame(rows)
    return (frame, pd.DataFrame(regulators)) if details else frame


def counts(bundle):
    return bundle.frame(section(bundle)["settings"])


GROUPS = ("reverse was a known edge in the standard split", "other test pairs")


def reverse_groups(bundle, datasets=(1501, 1605), coverage=.8):
    from sklearn.metrics import average_precision_score
    removed = counts(bundle).set_index(["part", "dataset", "coverage", "split_seed"]).removed_known_edges
    rows = []
    fits = [f for f in section(bundle)["fits"] if f["part"] == "reverse_blocked" and f["dataset"] in datasets and f["coverage"] == coverage]
    fits.sort(key=lambda f: (datasets.index(f["dataset"]), seeds.SEEDS.index(f["split_seed"])))
    if len(fits) != len(datasets) * len(seeds.SEEDS):
        raise ValueError("Reverse-blocked decomposition needs three splits per context")
    for fit in fits:
        edges, labels, known, members, _ = own_inputs(bundle, fit)
        standard = seeds.experiments(bundle, "fixed_pools", control="beacon", dataset=fit["dataset"], coverage=coverage,
                                     split_seed=fit["split_seed"])
        train = bundle.arrays(standard[0]["reference"]["split"])["train"] == 1
        blocked = np.zeros_like(train)
        blocked[tuple(np.asarray(known).T)] = True
        if (blocked & ~train).any() or int((train & ~blocked).sum()) != int(removed.loc[("reverse_blocked", fit["dataset"], coverage, fit["split_seed"])]):
            raise ValueError("Reverse-blocked known edges are not the standard known edges minus the removed reverses")
        reverse = train[edges[:, 1], edges[:, 0]]
        scores = {method: seeds.average(members[method]) for method in ("BEACON", "GNNLink")}
        reference, panel = metadata(bundle, fit["dataset"])
        for group, mask in zip(GROUPS, (reverse, ~reverse)):
            rows.append(dict(dataset=fit["dataset"], reference=reference, panel=panel, coverage=coverage, split_seed=fit["split_seed"],
                             group=group, pairs=int(mask.sum()), held_out_edges=int(labels[mask].sum()),
                             **{f"{method}_ap": float(average_precision_score(labels[mask], values[mask])) for method, values in scores.items()}))
    return pd.DataFrame(rows)


def run(bundle, details=False):
    result = per_fit_metrics(bundle, details=details)
    per_fit = result[0] if details else result
    out = dict(per_fit=per_fit, paired=paired(per_fit), summary=summary(per_fit), settings=counts(bundle), reverse_groups=reverse_groups(bundle))
    if details:
        out["per_regulator"] = result[1]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = run(Bundle(args.config), details=True)
    result["per_fit"].to_csv(args.out / "per_fit_metrics.csv", index=False)
    result["per_regulator"].to_csv(args.out / "per_regulator_within_ap.csv", index=False)
    result["paired"].to_csv(args.out / "paired.csv", index=False)
    result["paired"][result["paired"].part.eq("ten_splits")].to_csv(args.out / "ten_splits_paired.csv", index=False)
    result["summary"].to_csv(args.out / "summary.csv", index=False)
    result["settings"].to_csv(args.out / "settings.csv", index=False)
    result["reverse_groups"].to_csv(args.out / "reverse_groups.csv", index=False)


if __name__ == "__main__":
    main()
