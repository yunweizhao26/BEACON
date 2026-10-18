from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
from evaluation.metrics import metrics, pool_metrics, popularity, match_pairs
from evaluation.probabilities import calibrate, probability_report, selective_report
from evaluation import queries, seeds
from evaluation.specificity import COMPARATORS, aligned, comparator_scores

CONTEXT = queries.CONTEXT
AXES = ("edges", "labels", "latent_variance")
SUITES = {"fixed_pools": "completion", "sampled_pairs": "sampled_pair", "expression": "expression_controls", "sergio": "simulator",
          "hardware": "hardware"}


def score_keys(data):
    return [k for k, value in data.items() if k not in AXES and value.shape == data["labels"].shape]


def scored(meta, method, labels, values, edges):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite saved scores")
    row = dict(meta, method=method, **pool_metrics(labels, values, edges))
    row.update(metrics(labels, values, edges[:, 0]))
    return row


def readouts(bundle, fits):
    prediction = seeds.fit_arrays(bundle, fits, "predictions")
    edges, labels = prediction["edges"], prediction["labels"]
    scores = {k: prediction[k] for k in score_keys(prediction)}
    reference = fits[0]["reference"]
    if "decoder_predictions" in reference:
        decoder = seeds.fit_arrays(bundle, fits, "decoder_predictions")
        if not np.array_equal(decoder["edges"], edges) or not np.array_equal(decoder["labels"], labels):
            raise ValueError("Decoder axes differ")
        scores["decoder"] = decoder["decoder"]
    if "expression_predictions" in reference:
        outer = seeds.fit_arrays(bundle, fits, "expression_predictions")
        if not np.array_equal(outer["edges"], edges) or not np.array_equal(outer["labels"], labels):
            raise ValueError("Expression-control readout axes differ")
        scores["decoder"] = outer["beacon_decoder"]
    split = bundle.arrays(reference["split"])
    if not np.array_equal(split["test"][tuple(edges.T)], labels):
        raise ValueError("Prediction labels and frozen split differ")
    return prediction, edges, labels, scores, split["train"]


def probability_rows(bundle, fits, prediction, meta):
    valid = seeds.fit_arrays(bundle, fits, "validation_predictions")
    labels = prediction["labels"]
    names = [k for k in valid if k not in AXES and k in prediction]
    calibrated, _ = calibrate(valid["labels"], {k: valid[k] for k in names}, labels, {k: prediction[k] for k in names},
                              prediction["latent_variance"])
    probability, selective = [], []
    for method in [*calibrated, "validation_prevalence_constant"]:
        forecasts = {"raw": prediction[method], "platt": calibrated[method]} if method in calibrated else {
            "constant": np.full(len(labels), valid["labels"].mean())}
        for form, values in forecasts.items():
            probability.append(dict(meta, method=method, form=form, **{k: v for k, v in probability_report(labels, values).items()
                                                                       if k != "quantile_reliability_bins"}))
        if method in calibrated:
            cp = calibrated[method]
            criteria = {"probability_entropy_order": cp * (1 - cp)}
            if method == "beacon":
                criteria["latent_variance"] = prediction["latent_variance"]
            for criterion, uncertainty in criteria.items():
                for r in selective_report(labels, cp, uncertainty):
                    selective.append(dict(meta, method=method, criterion=criterion, retained_coverage=r["coverage"],
                                          positive_retention=r["positives"] / labels.sum()))
    return probability, selective


def comparator_frame(bundle, dataset, split_seed, coverage, edges, labels):
    return comparator_scores(bundle, dataset, split_seed, coverage, edges, methods={m: m for m in COMPARATORS})


def resources_row(bundle, experiment):
    if "summary" not in experiment["reference"]:
        return None
    summary = bundle.json(experiment["reference"]["summary"])
    if "peak_rss_mb" not in summary:
        return None
    suite, seed = experiment["suite"], experiment["condition"].get("seed")
    group = {"fixed_pools": "completion" if seed == 42 else "stability", "sergio": "simulator", "sampled_pairs": "sampled_pair"}.get(suite, suite)
    times = summary.get("timings", {})
    return dict(group=group, total_seconds=times.get("total_seconds"), encoder_seconds=times.get("representation_training_seconds"),
                gp_seconds=times.get("gp_training_seconds"), peak_rss_mb=summary["peak_rss_mb"],
                peak_cuda_allocated_mb=summary.get("peak_cuda_allocated_mb"))


def replay(bundle):
    pools, optimization, sergio, expression, probability, selective, resources, concordance, comparator_seeds = ([] for _ in range(9))
    primary_scores = {}
    experiments = bundle.manifest["experiments"]
    for experiment in experiments:
        if "summary" in experiment["reference"] and experiment["suite"] in ("fixed_pools", "sergio", "external", "sampled_pairs"):
            row = resources_row(bundle, experiment)
            if row:
                resources.append(row)
    settings = sorted({(e["condition"]["dataset"], e["condition"]["coverage"], e["condition"]["split_seed"], e["condition"]["control"])
                       for e in experiments if e["suite"] == "fixed_pools"}, key=str)
    for dataset, coverage, split_seed, control in settings:
        fits = seeds.experiments(bundle, "fixed_pools", dataset=dataset, coverage=coverage, split_seed=split_seed, control=control)
        prediction, edges, labels, scores, train = readouts(bundle, fits)
        meta = dict(suite="completion", dataset_id=dataset, context=CONTEXT[dataset], split_seed=split_seed, coverage=coverage,
                    control=control, variant=control, fit="ensemble", opt_seed=np.nan)
        scores.update(popularity(train, edges))
        if control == "beacon":
            scores.update(comparator_frame(bundle, dataset, split_seed, coverage, edges, labels))
        for method, values in scores.items():
            pools.append(scored(meta, method, labels, values, edges))
        if control == "beacon" and "calibrated_predictions" in fits[0]["reference"]:
            rows, chosen = probability_rows(bundle, fits, prediction, meta)
            probability += rows
            selective += chosen
        if coverage == .8:
            stash = primary_scores.setdefault((dataset, split_seed), {"edges": edges, "labels": labels, "scores": {}, "train": train})
            for method, values in scores.items():
                if control == "beacon" or method == control:
                    stash["scores"][method] = values
            if control == "beacon":
                for fit in fits:
                    single = bundle.arrays(fit["reference"]["predictions"])
                    optimization.append(scored(dict(meta, fit="seed", seed=fit["condition"]["seed"], opt_seed=fit["condition"]["seed"]),
                                               "beacon", labels, single["beacon"], edges))
                for method in ("gnnlink", "genelink", "gclink"):
                    for entry in seeds.comparators(bundle, "fixed_pools", method, dataset=dataset, coverage=coverage, split_seed=split_seed):
                        data = bundle.arrays(entry["artifacts"]["predictions"])
                        values = aligned(data["edges"], data[method], edges)
                        comparator_seeds.append(scored(dict(meta, fit="seed", seed=entry["seed"], opt_seed=entry["seed"]), method, labels, values, edges))
    groups = {}
    for experiment in experiments:
        suite, c = experiment["suite"], experiment["condition"]
        if suite in ("sampled_pairs", "sergio"):
            groups.setdefault((suite, json.dumps({k: v for k, v in c.items() if k != "seed"}, sort_keys=True)), []).append(experiment)
        elif suite in ("expression", "hardware"):
            groups[(suite, experiment["setting"])] = [experiment]
    for (suite, _), members in groups.items():
        if suite in ("sampled_pairs", "sergio"):
            c = {k: v for k, v in members[0]["condition"].items() if k != "seed"}
            fits = seeds.experiments(bundle, suite, **c)
            fit_meta = dict(fit="ensemble", opt_seed=np.nan)
        else:
            fits, c = members, members[0]["condition"]
            fit_meta = dict(fit="seed", opt_seed=c["seed"])
        reference = fits[0]["reference"]
        meta = dict(suite=SUITES[suite], dataset_id=c["dataset"], context=CONTEXT.get(c["dataset"], ""), split_seed=c.get("split_seed"),
                    coverage=c.get("coverage"), control=c["control"], variant=c["control"], **fit_meta)
        if suite == "sampled_pairs":
            prediction = seeds.fit_arrays(bundle, fits, "predictions")
            edges, labels = prediction["edges"], prediction["labels"]
            scores = {k: prediction[k] for k in score_keys(prediction)}
            dataset = next(d for d in bundle.manifest["datasets"] if d["dataset"] == c["dataset"])
            train = bundle.array(dataset["sampled"]["train"])
        else:
            prediction, edges, labels, scores, train = readouts(bundle, fits)
        scores.update(popularity(train, edges))
        for method, values in scores.items():
            row = scored(meta, {"beacon": "beacon_gp", "graph_only": "beacon_gp", "decoder": "beacon_decoder"}.get(method, method)
                         if suite == "expression" else method, labels, values, edges)
            pools.append(row)
            if suite == "sergio" and method == "beacon":
                metadata = bundle.json(reference["metadata"])
                sergio.append(dict(row, dataset=metadata["dataset"], ap_over_prevalence=row["average_precision"] / row["prevalence"],
                                   hidden_fraction_of_unlabeled=metadata["hidden_reference_positives_sampled_as_unlabeled"] / metadata["train_unlabeled"]))
            if suite == "expression" and method in ("beacon", "graph_only", "decoder"):
                expression.append(dict(row, readout=row["method"]))
        if suite == "sergio" and "calibrated_predictions" in reference:
            rows, chosen = probability_rows(bundle, fits, prediction, dict(meta, suite="simulator"))
            probability += rows
            selective += chosen
    for entry in bundle.manifest["comparators"]:
        if entry["suite"] != "expression":
            continue
        data = bundle.arrays(entry["artifacts"]["predictions"])
        meta = dict(suite="expression_controls", dataset_id=entry["dataset"], context=CONTEXT[entry["dataset"]], split_seed=entry["split_seed"],
                    coverage=entry["coverage"], control=entry["control"], variant=entry["control"], fit="seed", opt_seed=entry["seed"])
        pools.append(scored(meta, entry["method"], data["labels"], data[entry["method"]], data["edges"]))
    for (dataset, seed), record in sorted(primary_scores.items()):
        item = next(d for d in bundle.manifest["datasets"] if d["dataset"] == dataset)
        values = bundle.frame(item["expression"], index_col=0).to_numpy()
        raw = np.column_stack([np.log1p((record["train"] == 1).sum(axis=0)), np.log1p(values.mean(axis=1)), (values > 0).mean(axis=1)])
        scale = raw.std(axis=0)
        scale[scale == 0] = 1
        features = (raw - raw.mean(axis=0)) / scale
        pairs = match_pairs(record["edges"], record["labels"], features)
        pos, neg = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        for method, scores in record["scores"].items():
            values = (scores[pos] > scores[neg]).astype(float) + .5 * (scores[pos] == scores[neg])
            concordance.append(dict(dataset=dataset, split_seed=seed, method=method, pair_concordance=float(values.mean())))
    inducing = []
    for e in experiments:
        if e["suite"] != "inducing":
            continue
        for count, values in bundle.json(e["reference"]["record"])["fits"].items():
            inducing.append(dict(inducing_points=int(count), training_seconds=values["training_seconds"], peak_cuda_allocated_mb=values["peak_cuda_allocated_mb"],
                **{f"{pool}_{metric}": values[pool][metric] for pool in ("validation", "test") for metric in ("average_precision", "auprc_trapezoid")}))
    pools = pd.DataFrame(pools)
    completion = pools[pools.suite.eq("completion")].reset_index(drop=True)
    tables = {"metrics.csv": pools, "summary/completion_runs.csv": completion, "summary/optimization_seeds.csv": pd.DataFrame(optimization),
              "summary/sergio_runs.csv": pd.DataFrame(sergio), "summary/resources.csv": pd.DataFrame(resources),
              "completion_summary/hard_challenge_metrics.csv": pd.DataFrame(concordance),
              "probability_summary/probability_metrics.csv": pd.DataFrame(probability),
              "probability_summary/selective_metrics.csv": pd.DataFrame(selective),
              "summary/inducing_points.csv": pd.DataFrame(inducing), "expression": pd.DataFrame(expression),
              "summary/comparator_seeds.csv": pd.DataFrame(comparator_seeds),
              "comparators": completion[completion.method.isin(list(COMPARATORS))].reset_index(drop=True)}
    return tables


def comparator_queries(tables):
    completion = tables["summary/completion_runs.csv"]
    primary = completion.query('coverage == .8 and variant == "beacon"')
    comp = tables["comparators"].query('coverage == .8')
    add = queries.add
    for metric in ("auprc_trapezoid", "all_tf_ap", "auroc"):
        cm = comp.groupby(["method", "context"])[metric].mean()
        beacon = primary.query('method == "beacon"').groupby("context")[metric].mean()
        decoder = primary.query('method == "decoder"').groupby("context")[metric].mean()
        degree = primary.query('method == "degree_logistic"').groupby("context")[metric].mean()
        for (method, context), value in cm.items():
            add("BEACON minus comparator (80%)", f"BEACON - {method}", context, metric, beacon[context] - value)
        for context in queries.ORDER:
            add("BEACON minus comparator (80%)", "BEACON - topology control", context, metric, beacon[context] - degree[context])
            add("BEACON minus comparator (80%)", "BEACON GP - pair decoder", context, metric, beacon[context] - decoder[context])
        if metric == "auprc_trapezoid":
            best = cm[cm.index.get_level_values(0).isin(["gnnlink", "gclink", "scregulate", "reggain"])].groupby(level=1).max()
            for context in queries.ORDER:
                add("BEACON minus comparator (80%)", "BEACON - best of GNNLink/GCLink/scRegulate/RegGAIN", context, metric, beacon[context] - best[context])
    wide = comp.pivot_table(index=["context", "split_seed"], columns="method", values="auprc_trapezoid")
    beacon = primary.query('method == "beacon"').set_index(["context", "split_seed"]).auprc_trapezoid
    for method in ("gnnlink", "gclink", "scregulate", "reggain"):
        b, w = beacon.align(wide[method], join="inner")
        if len(b) != 12 or b.isna().any() or w.isna().any():
            raise ValueError(f"Expected 12 matched context-split pairs for {method}: beacon {list(beacon.index)}, comparator {list(wide[method].dropna().index)}")
        add("BEACON minus comparator (80%)", f"split wins over {method} (of 12)", "all", "auprc_trapezoid", int((b > w).sum()))


def sampled_queries(bundle, tables):
    beacon = tables["metrics.csv"].query('suite == "sampled_pair" and method == "beacon"').set_index("dataset_id")
    source = bundle.frame(bundle.manifest["tables"]["sampled_degree"])
    degree = source.query('pool == "saved_sampled_test" and method == "training_degree_logistic"').set_index("dataset_id")
    beacon = beacon.loc[degree.index]
    if len(beacon) != 44 or not np.array_equal(beacon.pairs, degree.pairs) or not np.array_equal(beacon.positives, degree.positives):
        raise ValueError("Incomplete or unmatched sampled-pair reference")
    contexts = degree.dataset.str.rsplit("_", n=1).str[0].str.split("_", n=1).str[1]
    rng = np.random.default_rng(20_260_905)
    for metric in ("auroc", "average_precision", "auprc_trapezoid", "macro_auroc", "macro_average_precision"):
        delta = degree[metric] - beacon[metric]
        frame = pd.DataFrame({"delta": delta, "context": contexts}).dropna()
        blocks = frame.groupby("context").delta.agg(["sum", "count"])
        draws = rng.integers(0, len(blocks), size=(10000, len(blocks)))
        boot = blocks["sum"].to_numpy()[draws].sum(axis=1) / blocks["count"].to_numpy()[draws].sum(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        values = dict(beacon_mean=beacon[metric].mean(), degree_mean=degree[metric].mean(), paired_difference=delta.mean(),
                      degree_wins=int((delta > 0).sum()), beacon_wins=int((delta < 0).sum()))
        values.update({"ci_low (degree - BEACON)": low, "ci_high (degree - BEACON)": high})
        for item, value in values.items():
            queries.add("Sampled-pair benchmark (44 settings)", item, "all", metric, value)


def sampled_comparison(bundle, tables):
    beacon = tables["metrics.csv"].query('suite == "sampled_pair" and method == "beacon"').set_index("dataset_id")
    degree = bundle.frame(bundle.manifest["tables"]["sampled_degree"]).query(
        'pool == "saved_sampled_test" and method == "training_degree_logistic"').set_index("dataset_id")
    rows = []
    for dataset_id, item in degree.iterrows():
        name = item.dataset
        row = dict(dataset_id=int(dataset_id), dataset=name, cell_context=name.rsplit("_", 1)[0].split("_", 1)[1], family=name.split("_", 1)[0])
        for metric in ("auroc", "average_precision", "auprc_trapezoid", "macro_auroc", "macro_average_precision"):
            row[f"beacon_{metric}"] = beacon.loc[dataset_id, metric]
            row[f"degree_{metric}"] = item[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def reverse_group_queries(bundle):
    from evaluation.robustness import reverse_groups
    for row in reverse_groups(bundle).itertuples(index=False):
        context = f"{CONTEXT[row.dataset]}, 80% prior coverage, split {row.split_seed}"
        queries.add("Reverse-blocked decomposition", f"{row.group}: pairs", context, "count", row.pairs)
        queries.add("Reverse-blocked decomposition", f"{row.group}: held-out edges", context, "count", row.held_out_edges)
        for method in ("BEACON", "GNNLink"):
            queries.add("Reverse-blocked decomposition", f"{row.group}: {method}", context, "average_precision", getattr(row, f"{method}_ap"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=RELEASE / "results" / "evaluation")
    parser.add_argument("--responses", action="store_true")
    args = parser.parse_args()
    bundle = Bundle(args.config)
    bundle.verify()
    args.out.mkdir(parents=True, exist_ok=False)
    tables = replay(bundle)
    queries.TABLES = tables
    queries.rows = []
    for name in ("primary", "coverage", "concordance", "calibration", "sergio", "stability", "expression_controls", "resources"):
        getattr(queries, name)()
    comparator_queries(tables)
    sampled_queries(bundle, tables)
    reverse_group_queries(bundle)
    tables["sampled_pair_summary/paired_comparison.csv"] = sampled_comparison(bundle, tables)
    pd.DataFrame(queries.rows).to_csv(args.out / "key_numbers.csv", index=False)
    for key, frame in tables.items():
        path = args.out / (key if key.endswith(".csv") else key + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    (args.out / "manifest.json").write_text(json.dumps({"status": "rescored", "key_rows": len(queries.rows),
        "metric_only": ["inducing reference metrics", "resource receipts", "sampled topology comparator metrics"],
        "seed_ensembles": list(seeds.SEEDS), "prediction_refits": False}, indent=2) + "\n")
    if args.responses:
        from evaluation.responses import evaluate
        evaluate(bundle, args.out / "responses")
        from evaluation.trrust import evaluate as trrust
        trrust(bundle, args.out / "trrust")


if __name__ == "__main__":
    main()
