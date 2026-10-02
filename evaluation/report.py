"""Rescore saved predictions and evaluate the manuscript's numeric queries."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
from evaluation.metrics import metrics, pool_metrics, popularity, match_pairs
from evaluation.probabilities import probability_report, selective_report
from evaluation import queries

CONTEXT = queries.CONTEXT

def score_keys(data):
    return [k for k, value in data.items() if k not in ("edges", "labels", "latent_variance") and value.shape == data["labels"].shape]

def aligned_scores(data, edges, labels, method):
    if "eligible_edges" in data:
        n = int(np.max(data["eligible_edges"])) + 1
        matrix = np.full((n, n), np.nan)
        matrix[tuple(data["eligible_edges"].T)] = data["eligible_scores"]
        return {method: matrix[tuple(edges.T)]}
    prefix = "test_" if "test_edges" in data else ""
    if not np.array_equal(data[prefix + "edges"], edges) or not np.array_equal(data[prefix + "labels"], labels):
        raise ValueError(f"Comparator axes differ: {method}")
    keys = [k for k in data if k not in (prefix + "edges", prefix + "labels", "latent_variance") and data[k].shape == labels.shape]
    return {k.removeprefix(prefix): data[k] for k in keys}

def replay(bundle):
    pools, completion, optimization, sergio, expression, probability, selective, resources, concordance = ([] for _ in range(9))
    primary_scores = {}
    for experiment in bundle.manifest["experiments"]:
        suite, c, reference = experiment["suite"], experiment["condition"], experiment["reference"]
        meta = dict(c, dataset_id=c["dataset"], context=CONTEXT.get(c["dataset"], ""), variant=c["control"] + ("_snn" if c["snn_weight"] else ""))
        if "summary" in reference and suite in ("fixed_pools", "optimization", "sergio", "external", "sampled_pairs"):
            summary = bundle.json(reference["summary"])
            if "peak_rss_mb" in summary:
                times = summary.get("timings", {})
                resources.append(dict(group={"fixed_pools":"completion", "optimization":"stability", "sergio":"simulator", "sampled_pairs":"sampled_pair"}.get(suite, suite),
                    total_seconds=times.get("total_seconds"), encoder_seconds=times.get("representation_training_seconds"),
                    gp_seconds=times.get("gp_training_seconds"), peak_rss_mb=summary["peak_rss_mb"], peak_cuda_allocated_mb=summary.get("peak_cuda_allocated_mb")))
        if "predictions" not in reference or suite in ("external", "rpe1"):
            continue
        prediction = bundle.arrays(reference["predictions"])
        edges, labels = prediction["edges"], prediction["labels"]
        scores = {k: prediction[k] for k in score_keys(prediction)}
        if "decoder_predictions" in reference:
            decoder = bundle.arrays(reference["decoder_predictions"])
            if not np.array_equal(decoder["edges"], edges) or not np.array_equal(decoder["labels"], labels):
                raise ValueError("Decoder axes differ")
            scores["decoder"] = decoder["decoder"]
        if suite == "expression":
            outer = bundle.arrays(reference["expression_predictions"])
            if not np.array_equal(outer["edges"], edges) or not np.array_equal(outer["labels"], labels):
                raise ValueError("Expression-control readout axes differ")
            scores["decoder"] = outer["beacon_decoder"]
        if suite == "sampled_pairs":
            dataset = next(d for d in bundle.manifest["datasets"] if d["dataset"] == c["dataset"])
            train = bundle.array(dataset["sampled"]["train"])
        else:
            split = bundle.arrays(reference["split"])
            train = split["train"]
            if not np.array_equal(split["test"][tuple(edges.T)], labels):
                raise ValueError("Prediction labels and frozen split differ")
        own = dict(scores)
        scores.update(popularity(train, edges))
        for method, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite saved scores")
            row = dict(meta, suite=suite, method=method, **pool_metrics(labels, values, edges))
            row.update(metrics(labels, values, edges[:, 0]))
            pools.append(row)
            if suite == "fixed_pools":
                completion.append(row)
            if suite == "optimization" and method == "beacon" or suite == "fixed_pools" and c["coverage"] == .8 and c["ratio"] == 5 and not c["corruption"] and meta["variant"] == "beacon" and method == "beacon":
                optimization.append(row)
            if suite == "sergio" and method == "beacon":
                metadata = bundle.json(reference["metadata"])
                sergio.append(dict(row, dataset=metadata["dataset"], ap_over_prevalence=row["average_precision"] / row["prevalence"],
                    hidden_fraction_of_unlabeled=metadata["hidden_reference_positives_sampled_as_unlabeled"] / metadata["train_unlabeled"]))
            if suite == "expression" and method in ("beacon", "graph_only", "decoder"):
                expression.append(dict(row, readout="beacon_decoder" if method == "decoder" else "beacon_gp", opt_seed=c["seed"]))
        if suite == "fixed_pools" and c["coverage"] == .8 and c["ratio"] == 5 and not c["corruption"]:
            key = (c["dataset"], c["split_seed"])
            stash = primary_scores.setdefault(key, {"edges": edges, "labels": labels, "scores": {}, "train": train})
            for method, values in scores.items():
                name = "beacon_snn" if c["snn_weight"] and method == "beacon" else method
                if meta["variant"] == "beacon" or name == c["control"] or name == "beacon_snn":
                    stash["scores"][name] = values
        if suite in ("fixed_pools", "sergio") and "calibrated_predictions" in reference and c["control"] == "beacon":
            calibrated = bundle.arrays(reference["calibrated_predictions"])
            valid = bundle.arrays(reference["validation_predictions"])
            metadata = dict(meta, suite="completion" if suite == "fixed_pools" else "simulator")
            for method in [*calibrated, "validation_prevalence_constant"]:
                forecasts = {"raw": prediction[method], "platt": calibrated[method]} if method in calibrated else {"constant": np.full(len(labels), valid["labels"].mean())}
                for form, values in forecasts.items():
                    probability.append(dict(metadata, method=method, form=form, **{k:v for k,v in probability_report(labels, values).items() if k != "quantile_reliability_bins"}))
                if method in calibrated:
                    cp = forecasts["platt"]
                    criteria = {"probability_entropy_order": cp * (1-cp)}
                    if method == "beacon":
                        criteria["latent_variance"] = prediction["latent_variance"]
                    for criterion, uncertainty in criteria.items():
                        for row in selective_report(labels, cp, uncertainty):
                            selective.append(dict(metadata, method=method, criterion=criterion, retained_coverage=row["coverage"], positive_retention=row["positives"] / labels.sum()))
    comparator_rows = []
    for comparator in bundle.manifest["comparators"]:
        if comparator["suite"] == "fixed_pools" and "dataset" in comparator and comparator.get("coverage") == .8:
            dataset, seed = comparator["dataset"], comparator["split_seed"]
            if (dataset, seed) not in primary_scores or "predictions" not in comparator["artifacts"]:
                continue
            seeds = (42,14,100) if comparator["method"] in ("genie3", "grnboost2") and seed == 42 else (seed,)
            for selected_seed in seeds:
                target = primary_scores[(dataset, selected_seed)]
                values = aligned_scores(bundle.arrays(comparator["artifacts"]["predictions"]), target["edges"], target["labels"], comparator["method"])
                for method, scores in values.items():
                    target["scores"][method] = scores
                    comparator_rows.append(dict(dataset=dataset, context=CONTEXT[dataset], split_seed=selected_seed, method=method,
                        **pool_metrics(target["labels"], scores, target["edges"])))
    for (dataset, seed), record in sorted(primary_scores.items()):
        item = next(d for d in bundle.manifest["datasets"] if d["dataset"] == dataset)
        # This evaluator intentionally uses pandas' float64 conversion, not the fit loader's float32.
        values = bundle.frame(item["expression"], index_col=0).to_numpy()
        raw = np.column_stack([np.log1p((record["train"] == 1).sum(axis=0)), np.log1p(values.mean(axis=1)), (values > 0).mean(axis=1)])
        scale = raw.std(axis=0)
        scale[scale == 0] = 1
        features = (raw - raw.mean(axis=0)) / scale
        pairs = match_pairs(record["edges"], record["labels"], features)
        pos, neg = pairs[:,0].astype(int), pairs[:,1].astype(int)
        for method, scores in record["scores"].items():
            values = (scores[pos] > scores[neg]).astype(float) + .5 * (scores[pos] == scores[neg])
            concordance.append(dict(dataset=dataset, split_seed=seed, method=method, pair_concordance=float(values.mean())))
    inducing = []
    for e in bundle.manifest["experiments"]:
        if e["suite"] != "inducing":
            continue
        for count, values in bundle.json(e["reference"]["record"])["fits"].items():
            inducing.append(dict(inducing_points=int(count), training_seconds=values["training_seconds"], peak_cuda_allocated_mb=values["peak_cuda_allocated_mb"],
                **{f"{pool}_{metric}":values[pool][metric] for pool in ("validation","test") for metric in ("average_precision","auprc_trapezoid")}))
    tables = {"summary/completion_runs.csv":pd.DataFrame(completion), "summary/optimization_seeds.csv":pd.DataFrame(optimization),
        "summary/sergio_runs.csv":pd.DataFrame(sergio), "summary/resources.csv":pd.DataFrame(resources),
        "completion_summary/hard_challenge_metrics.csv":pd.DataFrame(concordance),
        "probability_summary/probability_metrics.csv":pd.DataFrame(probability), "probability_summary/selective_metrics.csv":pd.DataFrame(selective),
        "summary/inducing_points.csv":pd.DataFrame(inducing), "expression":pd.DataFrame(expression), "pool_metrics":pd.DataFrame(pools),
        "comparators":pd.DataFrame(comparator_rows).drop_duplicates()}
    tables["coverage_comparators"] = bundle.frame(bundle.manifest["tables"]["coverage_comparators"]).query('method == "gnnlink"')
    return tables

def comparator_queries(tables):
    primary = tables["summary/completion_runs.csv"].query('coverage == .8 and ratio == 5 and corruption == 0 and variant == "beacon"')
    comp = tables["comparators"]
    add = queries.add
    for metric in ("auprc_trapezoid", "all_tf_ap", "auroc"):
        cm = comp.groupby(["method", "context"])[metric].mean()
        beacon = primary.query('method == "beacon"').groupby("context")[metric].mean()
        decoder = primary.query('method == "decoder"').groupby("context")[metric].mean()
        degree = primary.query('method == "degree_logistic"').groupby("context")[metric].mean()
        for (method, context), value in cm.items():
            add("BEACON minus comparator (80%)", f"BEACON - {method}", context, metric, None, beacon[context]-value)
        for context in queries.ORDER:
            add("BEACON minus comparator (80%)", "BEACON - topology control", context, metric, None, beacon[context]-degree[context])
            add("BEACON minus comparator (80%)", "BEACON GP - pair decoder", context, metric, None, beacon[context]-decoder[context])
        if metric == "auprc_trapezoid":
            best = cm[cm.index.get_level_values(0).isin(["gnnlink","gclink","scregulate","reggain"])].groupby(level=1).max()
            for context in queries.ORDER:
                add("BEACON minus comparator (80%)", "BEACON - best of GNNLink/GCLink/scRegulate/RegGAIN", context, metric, None, beacon[context]-best[context])
    wide = comp.pivot_table(index=["context","split_seed"], columns="method", values="auprc_trapezoid")
    beacon = primary.query('method == "beacon"').set_index(["context","split_seed"]).auprc_trapezoid
    for method in ("gnnlink","gclink","scregulate","reggain"):
        b, w = beacon.align(wide[method], join="inner")
        if len(b) != 12 or b.isna().any() or w.isna().any():
            raise ValueError(f"Expected 12 matched context-split pairs for {method}: beacon {list(beacon.index)}, comparator {list(wide[method].dropna().index)}")
        add("BEACON minus comparator (80%)", f"split wins over {method} (of 12)", "all", "auprc_trapezoid", None, int((b > w).sum()))

def sampled_queries(bundle, tables):
    beacon = tables["pool_metrics"].query('suite == "sampled_pairs" and method == "beacon"').set_index("dataset_id")
    source = bundle.frame(bundle.manifest["tables"]["sampled_degree"])
    degree = source.query('pool == "saved_sampled_test" and method == "training_degree_logistic"').set_index("dataset_id")
    beacon = beacon.loc[degree.index]
    if len(beacon) != 44 or not np.array_equal(beacon.pairs, degree.pairs) or not np.array_equal(beacon.positives, degree.positives):
        raise ValueError("Incomplete or unmatched sampled-pair reference")
    contexts = degree.dataset.str.rsplit("_",n=1).str[0].str.split("_",n=1).str[1]
    rng = np.random.default_rng(20_260_905)
    for metric in ("auroc","average_precision","auprc_trapezoid","macro_auroc","macro_average_precision"):
        delta = degree[metric]-beacon[metric]
        frame = pd.DataFrame({"delta":delta,"context":contexts}).dropna()
        blocks = frame.groupby("context").delta.agg(["sum","count"])
        draws = rng.integers(0,len(blocks),size=(10000,len(blocks)))
        boot = blocks["sum"].to_numpy()[draws].sum(axis=1)/blocks["count"].to_numpy()[draws].sum(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        values=dict(beacon_mean=beacon[metric].mean(),degree_mean=degree[metric].mean(),paired_difference=delta.mean(),
                    degree_wins=int((delta>0).sum()),beacon_wins=int((delta<0).sum()))
        values.update({"ci_low (degree - BEACON)":low,"ci_high (degree - BEACON)":high})
        for item,value in values.items():
            queries.add("Sampled-pair benchmark (44 settings)",item,"all",metric,None,value)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--out",type=Path,default=RELEASE/"results"/"evaluation")
    parser.add_argument("--responses",action="store_true")
    args=parser.parse_args()
    bundle=Bundle(args.config)
    bundle.verify()
    args.out.mkdir(parents=True,exist_ok=False)
    tables=replay(bundle)
    queries.TABLES=tables
    queries.rows=[]
    for name in ("primary","coverage","ratio_corruption","concordance","calibration","sergio","stability","expression_controls","resources"):
        getattr(queries,name)()
    comparator_queries(tables)
    sampled_queries(bundle,tables)
    pd.DataFrame(queries.rows).to_csv(args.out/"key_numbers.csv",index=False)
    for key,frame in tables.items():
        name=Path(key).stem
        frame.to_csv(args.out/(name+".csv"),index=False)
    (args.out/"manifest.json").write_text(json.dumps({"status":"rescored", "key_rows":len(queries.rows),
        "metric_only":["inducing reference metrics","resource receipts","sampled topology comparator metrics"],
        "prediction_refits":False},indent=2)+"\n")
    if args.responses:
        from evaluation.responses import evaluate
        evaluate(bundle,args.out/"responses")
        from evaluation.trrust import evaluate as trrust
        trrust(bundle,args.out/"trrust")

if __name__=="__main__":
    main()
