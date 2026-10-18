from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
from evaluation.metrics import binary_metrics, match_pairs, metrics, paired, pool_metrics, popularity, top_weights
from evaluation.probabilities import calibrate, probability_report
from evaluation.responses import label_bundle
from evaluation.specificity import COMPARATORS, EXPRESSION_ONLY, aligned
from evaluation import trrust

COMPONENTS = (8, 16, 32, 64, 128, 256)
REFERENCE = 64
SEED = 42
CONTEXTS = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
SERGIO = {201: "sparse", 205: "dense"}
COVERAGES = (.05, .1, .2, .4, .8)
SPLITS = (42, 14, 100)
EXTERNAL = {"gnnlink": "GNNLink", "reggain": "RegGAIN", "inferelator": "Inferelator", "degree_logistic": "Topology control",
            "expression_abs_correlation": "Expression correlation", "target_control_detection": "Target detection"}
TRRUST_METHODS = {"GNNLink": "gnnlink", "GENELink": "genelink", "GENIE3": "genie3", "Inferelator": "inferelator"}
SPLIT_RANGE = "observed three-split range (not a CI)"
CLUSTER = "95% 7-context cluster bootstrap; 10000 draws; seed 20_260_905"
COLUMNS = ["k", "endpoint", "label", "method", "unit", "value", "interval_method", "cluster", "direction", "compare"]
EXCLUDED = ()


def interval(values, method, clusters=None, units=None, saved=None):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Intervals need finite per-unit values")
    if method == "split_range":
        if len(values) != 3:
            raise ValueError("A split range needs the three splits")
        return float(values.min()), float(values.max()), SPLIT_RANGE
    if method.startswith("cluster_"):
        blocks = pd.DataFrame(dict(value=values, cluster=clusters)).groupby("cluster").value.agg(["sum", "count"])
        if len(blocks) != 7:
            raise ValueError("The sampled-pair interval resamples seven context clusters")
        rng = np.random.default_rng(20_260_905)
        for _ in range(1 if method == "cluster_auroc" else 3):
            indices = rng.integers(len(blocks), size=(10000, len(blocks)))
        boot = blocks["sum"].to_numpy()[indices].sum(axis=1) / blocks["count"].to_numpy()[indices].sum(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        return float(low), float(high), CLUSTER
    rng = np.random.default_rng(SEED)
    if method.startswith("binding_group_2"):
        rng.integers(int(method.rsplit(":", 1)[1]), size=(10000, int(method.rsplit(":", 1)[1])))
    elif method == "tcell_primary":
        for _ in range(2):
            rng.integers(len(values), size=(10000, len(values)))
    indices = rng.integers(len(values), size=(10000, len(values)))
    if method == "rpe1_saved":
        if saved is not None and np.array_equal(saved["tf_genes"].astype(str), np.asarray(units, str)):
            indices = saved["indices"]
        else:
            indices = np.random.default_rng(9071).integers(len(values), size=(10000, len(values)))
    result = paired(values, indices=indices)
    return result["ci_low"], result["ci_high"], f"95% paired regulator bootstrap; 10000 draws; {method.split(':')[0]}"


class Units:

    def __init__(self):
        self.rows = []

    def add(self, k, endpoint, label, method, unit, value, interval="split_range", cluster="", direction="higher", compare=True):
        if not np.isfinite(value):
            raise ValueError((endpoint, method, unit, value))
        self.rows.append(dict(k=k, endpoint=endpoint, label=label, method=method, unit=str(unit), value=float(value),
                              interval_method=interval, cluster=str(cluster), direction=direction, compare=compare))

    def frame(self):
        return pd.DataFrame(self.rows, columns=COLUMNS)


class Fits:

    def __init__(self, bundle):
        self.bundle = bundle
        self.inputs = set()
        section = bundle.manifest.get("factor_components")
        if section is None:
            raise ValueError("The bundle has no factor_components section")
        if tuple(section["components"]) != COMPONENTS or section["reference"] != REFERENCE:
            raise ValueError("Unexpected factor-component grid")
        self.section = section["fits"]

    def reference(self, suite, components, **condition):
        if components == REFERENCE:
            found = [e for e in self.bundle.manifest["experiments"] if e["suite"] == suite and e["condition"].get("seed") == SEED
                     and all(e["condition"].get(key) == value for key, value in condition.items())]
        else:
            found = [f for f in self.section if f["suite"] == suite and f["components"] == components and f["condition"].get("seed") == SEED
                     and all(f["condition"].get(key) == value for key, value in condition.items())]
        if len(found) != 1:
            raise ValueError(f"Expected one {suite} fit with {components} components for {condition}, found {len(found)}")
        return found[0]["reference"]

    def arrays(self, record):
        self.inputs.add(record)
        return self.bundle.arrays(record)

    def array(self, record):
        self.inputs.add(record)
        return self.bundle.array(record)

    def json(self, record):
        self.inputs.add(record)
        return self.bundle.json(record)


def seed_comparator(fits, suite, method, **keys):
    found = [c for c in fits.bundle.manifest["comparators"] if c.get("suite") == suite and c.get("method") == method
             and all(c.get(key) == value for key, value in keys.items())]
    seeded = [c for c in found if c.get("seed") == SEED]
    if len(seeded) == 1:
        return seeded[0]
    if len(found) == 1 and found[0].get("seed") is None:
        return found[0]
    raise ValueError(f"No seed-{SEED} comparator {method} {keys}")


def fixed_pools(fits, units):
    expression = {}
    for dataset, context in CONTEXTS.items():
        for coverage in COVERAGES:
            percent = int(round(100 * coverage))
            for split_seed in SPLITS:
                condition = dict(dataset=dataset, coverage=coverage, split_seed=split_seed, control="beacon")
                base = fits.reference("fixed_pools", REFERENCE, **condition)
                reference = fits.arrays(base["predictions"])
                edges, labels = reference["edges"], reference["labels"]
                train = fits.arrays(base["split"])["train"]
                scores = {"Topology control": reference["degree_logistic"], **popularity(train, edges)}
                methods = sorted({c["method"] for c in fits.bundle.manifest["comparators"] if c.get("suite") == "fixed_pools"
                                  and c.get("dataset") == dataset and c.get("coverage") == coverage and c.get("method") in COMPARATORS},
                                 key=list(COMPARATORS).index)
                for method in methods:
                    if method in EXPRESSION_ONLY:
                        data = fits.arrays(seed_comparator(fits, "fixed_pools", method, dataset=dataset, coverage=coverage)["artifacts"]["predictions"])
                        scores[COMPARATORS[method]] = aligned(data["eligible_edges"], data["eligible_scores"], edges, missing=0.)
                    elif any(c.get("split_seed") == split_seed for c in fits.bundle.manifest["comparators"] if c.get("suite") == "fixed_pools"
                             and c.get("method") == method and c.get("dataset") == dataset and c.get("coverage") == coverage):
                        entry = seed_comparator(fits, "fixed_pools", method, dataset=dataset, coverage=coverage, split_seed=split_seed)
                        data = fits.arrays(entry["artifacts"]["predictions"])
                        scores[COMPARATORS[method]] = aligned(data["edges"], data[method], edges)
                beacon = {}
                for k in COMPONENTS:
                    record = fits.reference("fixed_pools", k, **condition)
                    data = fits.arrays(record["predictions"])
                    if not np.array_equal(data["edges"], edges) or not np.array_equal(data["labels"], labels):
                        raise ValueError(f"Component fit axes differ: {condition}, {k}")
                    beacon[k] = data["beacon"]
                pooled = [(k, "BEACON", pool_metrics(labels, values, edges)) for k, values in beacon.items()]
                pooled += [(None, name, pool_metrics(labels, values, edges)) for name, values in scores.items()]
                for metric, title in (("auprc_trapezoid", "pooled AUPRC"), ("all_tf_ap", "all-TF AP")):
                    endpoint, label = f"completion/{dataset}/{percent}/{metric}", f"{context}, {percent}%: {title}"
                    for k, name, values in pooled:
                        units.add(k, endpoint, label, name, split_seed, values[metric])
                if coverage != .8:
                    continue
                if dataset not in expression:
                    item = next(d for d in fits.bundle.manifest["datasets"] if d["dataset"] == dataset)
                    fits.inputs.add(item["expression"])
                    expression[dataset] = fits.bundle.frame(item["expression"], index_col=0).to_numpy()
                values = expression[dataset]
                raw = np.column_stack([np.log1p((train == 1).sum(axis=0)), np.log1p(values.mean(axis=1)), (values > 0).mean(axis=1)])
                scale = raw.std(axis=0)
                scale[scale == 0] = 1
                pairs = match_pairs(edges, labels, (raw - raw.mean(axis=0)) / scale)
                positive, negative = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
                concord = lambda s: float(((s[positive] > s[negative]).astype(float) + .5 * (s[positive] == s[negative])).mean())
                endpoint, label = f"concordance/{dataset}", f"{context}: same-regulator concordance"
                for k, values in beacon.items():
                    units.add(k, endpoint, label, "BEACON", split_seed, concord(values))
                for name, values in scores.items():
                    units.add(None, endpoint, label, name, split_seed, concord(values))
                endpoint, label = f"calibration/{dataset}", f"{context}, 80%: calibrated Brier"
                valid = fits.arrays(base["validation_predictions"])
                calibrated, _ = calibrate(valid["labels"], {"Topology control": valid["degree_logistic"]}, labels,
                                          {"Topology control": reference["degree_logistic"]}, None)
                units.add(None, endpoint, label, "Topology control", split_seed, probability_report(labels, calibrated["Topology control"])["brier"], direction="lower")
                units.add(None, endpoint, label, "Validation prevalence constant", split_seed,
                          probability_report(labels, np.full(len(labels), valid["labels"].mean()))["brier"], direction="lower")
                for k in COMPONENTS:
                    record = fits.reference("fixed_pools", k, **condition)
                    validation = fits.arrays(record["validation_predictions"])
                    if not np.array_equal(validation["edges"], valid["edges"]) or not np.array_equal(validation["labels"], valid["labels"]):
                        raise ValueError(f"Component validation axes differ: {condition}, {k}")
                    calibrated, _ = calibrate(valid["labels"], {"BEACON": validation["beacon"]}, labels, {"BEACON": beacon[k]}, None)
                    units.add(k, endpoint, label, "BEACON", split_seed, probability_report(labels, calibrated["BEACON"])["brier"], direction="lower")


def sampled_pairs(fits, units):
    degree = fits.bundle.frame(fits.bundle.manifest["tables"]["sampled_degree"]).query(
        'pool == "saved_sampled_test" and method == "training_degree_logistic"').set_index("dataset_id")
    fits.inputs.add(fits.bundle.manifest["tables"]["sampled_degree"])
    datasets = sorted({e["condition"]["dataset"] for e in fits.bundle.manifest["experiments"] if e["suite"] == "sampled_pairs"})
    if len(datasets) != 44 or set(datasets) != set(degree.index):
        raise ValueError("Expected the 44 sampled-pair settings")
    for dataset in datasets:
        context = degree.loc[dataset, "dataset"].rsplit("_", 1)[0].split("_", 1)[1]
        reference = fits.arrays(fits.reference("sampled_pairs", REFERENCE, dataset=dataset)["predictions"])
        if len(reference["labels"]) != degree.loc[dataset, "pairs"] or reference["labels"].sum() != degree.loc[dataset, "positives"]:
            raise ValueError("Sampled-pair degree reference does not match the saved pool")
        for k in COMPONENTS:
            data = fits.arrays(fits.reference("sampled_pairs", k, dataset=dataset)["predictions"])
            if not np.array_equal(data["edges"], reference["edges"]) or not np.array_equal(data["labels"], reference["labels"]):
                raise ValueError(f"Component fit axes differ: sampled pairs {dataset}, {k}")
            values = metrics(data["labels"], data["beacon"], data["edges"][:, 0])
            for metric, title in (("auroc", "AUROC"), ("auprc_trapezoid", "AUPRC")):
                units.add(k, f"sampled_pair/{metric}", f"Sampled pairs: mean {title}", "BEACON", dataset, values[metric],
                          interval=f"cluster_{metric}", cluster=context)
                units.add(k, f"sampled_pair/difference/{metric}", f"Sampled pairs: {title} minus topology", "BEACON", dataset,
                          values[metric] - degree.loc[dataset, metric], interval=f"cluster_{metric}", cluster=context, compare=False)
        for metric, title in (("auroc", "AUROC"), ("auprc_trapezoid", "AUPRC")):
            units.add(None, f"sampled_pair/{metric}", f"Sampled pairs: mean {title}", "Topology control", dataset,
                      degree.loc[dataset, metric], interval=f"cluster_{metric}", cluster=context)


def sergio(fits, units):
    for dataset, density in SERGIO.items():
        for coverage in (.05, .2, .8):
            percent = int(round(100 * coverage))
            for split_seed in SPLITS:
                condition = dict(dataset=dataset, coverage=coverage, split_seed=split_seed, control="beacon")
                reference = fits.arrays(fits.reference("sergio", REFERENCE, **condition)["predictions"])
                for k in COMPONENTS:
                    data = fits.arrays(fits.reference("sergio", k, **condition)["predictions"])
                    if not np.array_equal(data["edges"], reference["edges"]) or not np.array_equal(data["labels"], reference["labels"]):
                        raise ValueError(f"Component fit axes differ: SERGIO {condition}, {k}")
                    values = pool_metrics(data["labels"], data["beacon"], data["edges"])
                    for metric, value in (("ap_over_prevalence", values["average_precision"] / values["prevalence"]), ("auroc", values["auroc"])):
                        units.add(k, f"sergio/{density}/{percent}/{metric}", f"SERGIO {density}, {percent}%: {metric}", "BEACON", split_seed,
                                  value, compare=False)


def perturbation(fits, units, context):
    base = fits.reference("external", REFERENCE, context=context, control="beacon")
    training = fits.arrays(base["training"])
    genes, tfs = training["genes"], training["tf_indices"]
    shape = (len(tfs), len(genes))
    discovery = fits.array(base["discovery_mask"])
    positive = training["train"] == 1
    scores = {"Topology control": fits.array(base["degree_logistic"]),
              "Prior target in-degree": np.broadcast_to(positive.sum(axis=0), shape),
              "Prior regulator out-degree": np.broadcast_to(positive.sum(axis=1)[tfs, None], shape)}
    for method in ("gnnlink", "reggain"):
        scores[EXTERNAL[method]] = fits.array(seed_comparator(fits, "external", method, context=context)["artifacts"]["scores"])
    beacon = {k: fits.array(fits.reference("external", k, context=context, control="beacon")["beacon"]) for k in COMPONENTS}
    for name, matrix in [*scores.items(), *beacon.items()]:
        if matrix.shape != shape:
            raise ValueError(f"Score axis mismatch: {context} {name}")
    fits.inputs.update((fits.bundle.manifest["labels"][context]["artifact"], fits.bundle.manifest["labels"][context]["calibration"]))
    labels = label_bundle(fits.bundle, context, genes, genes[tfs])
    mask = discovery[labels["selected"]] & labels["tested"]
    if context == "k562":
        prefix, title, interval_method = "k562/response", "K562", "regulator_seed42"
        names = [("average_precision", "response AP"), *[(f"top{n}_supported_fraction", f"top-{n} response fraction") for n in (50, 100, 200)]]
    else:
        condition = context.removeprefix("tcell_")
        prefix, title, interval_method = f"tcell/{condition}", f"T cells {condition}", "tcell_primary"
        names = [("average_precision", "AP"), ("top100_supported_fraction", "top-100 response fraction")]
    for i, tf in enumerate(labels["tf_genes"]):
        support = labels["support"][i, mask[i]]
        for k, method, matrix in [*((k, "BEACON", m) for k, m in beacon.items()), *((None, n, m) for n, m in scores.items())]:
            values = binary_metrics(support, matrix[labels["selected"][i], mask[i]])
            for metric, name in names:
                units.add(k, f"{prefix}/{metric}", f"{title}: {name}", method, tf, values[metric], interval=interval_method)
    if context != "k562":
        return
    data = fits.arrays(fits.bundle.manifest["prepared"]["binding"])
    if not np.array_equal(data["genes"], genes) or not np.array_equal(data["tf_indices"], tfs):
        raise ValueError("Binding axes differ from score matrix")
    lookup = {str(g): j for j, g in enumerate(labels["tf_genes"])}
    assayed = np.flatnonzero(data["assayed"])
    for i in assayed:
        tf = genes[tfs[i]]
        allowed = discovery[i] & data["annotated"]
        outcomes = [("binding", "binding AP", allowed, data["primary"][i], "binding_group_1")]
        if str(tf) in lookup:
            j = lookup[str(tf)]
            outcomes.append(("binding_and_response", "joint AP", allowed & labels["tested"][j], data["primary"][i] & labels["support"][j],
                             f"binding_group_2:{len(assayed)}"))
        for assay, name, eligible, support, method_interval in outcomes:
            for k, method, matrix in [*((k, "BEACON", m) for k, m in beacon.items()), *((None, n, m) for n, m in scores.items())]:
                units.add(k, f"k562/{assay}/average_precision", f"K562: {name}", method, tf,
                          binary_metrics(support[eligible], matrix[i, eligible])["average_precision"], interval=method_interval)


def rpe1(fits, units):
    prepared = fits.bundle.manifest["prepared"]["rpe1"]
    training = fits.arrays(prepared["training"])
    genes, tfs = training["genes"], training["tf_indices"]
    shape = (len(tfs), len(genes))
    discovery = fits.array(prepared["discovery_mask"])
    positive = training["train"] == 1
    scores = {"Prior target in-degree": np.broadcast_to(positive.sum(axis=0), shape),
              "Prior regulator out-degree": np.broadcast_to(positive.sum(axis=1)[tfs, None], shape)}
    methods = [c["method"] for c in fits.bundle.manifest["comparators"] if c.get("suite") == "rpe1"]
    for method in dict.fromkeys(methods):
        if method not in EXTERNAL:
            continue
        values = fits.array(seed_comparator(fits, "rpe1", method)["artifacts"]["scores"])
        scores[EXTERNAL[method]] = np.broadcast_to(values, shape) if values.shape == (len(genes),) else values
    beacon = {k: fits.array(fits.reference("rpe1", k, control="beacon")["beacon"]) for k in COMPONENTS}
    for name, matrix in [*scores.items(), *beacon.items()]:
        if matrix.shape != shape:
            raise ValueError(f"Score axis mismatch: RPE1 {name}")
    fits.inputs.update((fits.bundle.manifest["labels"]["rpe1"]["artifact"], fits.bundle.manifest["labels"]["rpe1"]["calibration"]))
    labels = label_bundle(fits.bundle, "rpe1", genes, genes[tfs])
    mask = discovery[labels["selected"]] & labels["tested"]
    names = (("top100_mean_abs_effect", "top-100 mean absolute effect"), ("average_precision", "response AP"))
    for i, tf in enumerate(labels["tf_genes"]):
        support = labels["support"][i, mask[i]]
        effects = np.abs(labels["standardized_effects"][i, mask[i]])
        for k, method, matrix in [*((k, "BEACON", m) for k, m in beacon.items()), *((None, n, m) for n, m in scores.items())]:
            values = matrix[labels["selected"][i], mask[i]]
            result = dict(top100_mean_abs_effect=float(top_weights(values) @ effects / 100),
                          average_precision=binary_metrics(support, values)["average_precision"])
            for metric, name in names:
                units.add(k, f"rpe1/{metric}", f"RPE1: {name}", method, tf, result[metric], interval="rpe1_saved")


def trrust_record(fits, record, pairs, full):
    value = fits.json(record)
    if value["context"] != "flu" or int(value["method_seed"]) != SEED:
        raise ValueError("Expected the influenza TRRUST record of task seed 42")
    frame = pd.DataFrame([(r["source"], r["target"], r["analysis_role"], float(r["probability"])) for r in value["records"]],
                         columns=[*trrust.KEYS, "score"])
    if frame.duplicated(trrust.KEYS).any() or set(frame[trrust.KEYS].itertuples(index=False, name=None)) != full:
        raise ValueError("TRRUST record does not cover the evaluation pairs once")
    return frame.set_index(trrust.KEYS).score.reindex(pd.MultiIndex.from_frame(pairs[trrust.KEYS])).to_numpy(dtype=np.float64)


def trrust_endpoints(fits, units):
    section = fits.bundle.manifest["trrust"]
    pairs = trrust.evaluated_pairs(section)
    full = {(r["source"], r["target"], r["analysis_role"]) for r in section["evaluation"]}
    scores = {name: trrust_record(fits, section["comparators"][key][0], pairs, full) for name, key in TRRUST_METHODS.items()}
    scores["GRNBoost2"] = trrust_record(fits, section["paired_grnboost2"][0], pairs, full)
    index = {gene: i for i, gene in enumerate(section["genes"])}
    edges = np.column_stack([pairs.source.map(index), pairs.target.map(index)]).astype(np.int64)
    fits.inputs.update((section["prior"], section["unmasked_prior"]))
    for name, values in trrust.baselines(trrust.training_graph(fits.bundle), edges).items():
        scores["Topology control" if name == "Degree logistic" else name] = values
    beacon = {k: trrust_record(fits, fits.reference("trrust", k, task=0, control="beacon")["record"], pairs, full) for k in COMPONENTS}
    original = fits.bundle.frame(section["unmasked_prior"], sep="\t", dtype=str)
    kept = trrust.exclude_overlap({"BEACON": pairs}, original)["BEACON"].index.to_numpy()
    for evaluation in trrust.EVALUATIONS:
        rows = np.arange(len(pairs)) if evaluation == "all" else kept
        part = pairs.iloc[rows].reset_index(drop=True)
        current = {**{("BEACON", k): v[rows] for k, v in beacon.items()}, **{(name, None): v[rows] for name, v in scores.items()}}
        per_tf = trrust.regulator_metrics(part, current)
        for metric, title in (("auroc", "same-regulator AUROC"), ("average_precision", "AP")):
            endpoint, label = f"trrust/{evaluation}/{metric}", f"TRRUST {evaluation.replace('_', ' ')}: {title}"
            for row in per_tf.itertuples():
                method, k = row.method
                units.add(k, endpoint, label, method, row.tf, getattr(row, metric), interval="regulator_seed42")


def per_unit(bundle):
    fits, units = Fits(bundle), Units()
    fixed_pools(fits, units)
    sampled_pairs(fits, units)
    sergio(fits, units)
    for context in ("k562", "tcell_resting", "tcell_stimulated"):
        perturbation(fits, units, context)
    rpe1(fits, units)
    trrust_endpoints(fits, units)
    fits.inputs.add(bundle.manifest["prepared"]["bootstrap"])
    return units.frame(), fits.inputs


def sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def status(low, high):
    return "above zero" if low > 0 else "below zero" if high < 0 else "includes zero"


def summarize(frame, saved=None):
    values, contrasts = [], []
    for endpoint, part in frame.groupby("endpoint", sort=True):
        beacon = part[part.method.eq("BEACON")]
        meta = beacon.iloc[0]
        if beacon.interval_method.nunique() != 1 or beacon.direction.nunique() != 1:
            raise ValueError(f"Inconsistent endpoint definition: {endpoint}")
        reference = beacon[beacon.k.eq(REFERENCE)].set_index("unit").sort_index()
        candidates = sorted(set(part.method) - {"BEACON"})
        if bool(meta["compare"]) and not candidates:
            raise ValueError(f"No comparator for {endpoint}")
        local = []
        for k in COMPONENTS:
            fit = beacon[beacon.k.eq(k)].set_index("unit").sort_index()
            if not fit.index.equals(reference.index) or not np.array_equal(fit.cluster.to_numpy(), reference.cluster.to_numpy()):
                raise ValueError(f"Reporting units differ: {endpoint}, {k}")
            low, high, kind = interval(fit.value, meta.interval_method, fit.cluster, fit.index, saved)
            local.append(dict(endpoint=endpoint, label=meta.label, k=k, value=float(fit.value.mean()), interval_low=low, interval_high=high,
                              interval_method=kind, direction=meta.direction, units=len(fit)))
            if not bool(meta["compare"]):
                continue
            for name in candidates:
                baseline = part[part.method.eq(name) & part.k.eq(k)].set_index("unit").sort_index()
                if not baseline.index.equals(fit.index):
                    raise ValueError(f"Comparator units differ: {endpoint}, {name}")
                delta = fit.value.to_numpy() - baseline.value.to_numpy()
                low, high, kind = interval(delta, meta.interval_method, fit.cluster, fit.index, saved)
                contrasts.append(dict(endpoint=endpoint, label=meta.label, k=k, baseline=name, baseline_value=float(baseline.value.mean()),
                                      beacon_value=float(fit.value.mean()), difference=float(delta.mean()), interval_low=low,
                                      interval_high=high, interval_method=kind, sign=sign(delta.mean()), interval_status=status(low, high),
                                      direction=meta.direction, units=len(fit)))
        low, high = min(r["value"] for r in local), max(r["value"] for r in local)
        values += [dict(r, minimum_across_k=low, maximum_across_k=high, range_across_k=high - low) for r in local]
    values, comparisons = pd.DataFrame(values), pd.DataFrame(contrasts)
    for _, part in comparisons.groupby(["endpoint", "baseline"]):
        ref = part[part.k.eq(REFERENCE)].iloc[0]
        index = part.index
        comparisons.loc[index, "sign_matches_k64"] = part.sign.eq(ref.sign)
        comparisons.loc[index, "interval_status_matches_k64"] = part.interval_status.eq(ref.interval_status)
        comparisons.loc[index, "sign_matches_at_every_k"] = bool(part.sign.eq(ref.sign).all())
        comparisons.loc[index, "interval_status_matches_at_every_k"] = bool(part.interval_status.eq(ref.interval_status).all())
        comparisons.loc[index, "difference_range_across_k"] = part.difference.max() - part.difference.min()
    return values, comparisons


def strongest(comparisons):
    rows = comparisons[comparisons.k.eq(REFERENCE)].copy()
    rows["key"] = np.where(rows.direction.eq("lower"), rows.baseline_value, -rows.baseline_value)
    return rows.sort_values(["endpoint", "key", "baseline"]).groupby("endpoint").head(1).set_index("endpoint").baseline


def tables(bundle):
    frame, inputs = per_unit(bundle)
    for prefix, method in EXCLUDED:
        frame = frame[~(frame.endpoint.str.startswith(prefix) & frame.method.eq(method))]
    frame = frame.reset_index(drop=True)
    comparators = frame[frame.k.isna()].drop(columns="k")
    beacon = frame[frame.k.notna()]
    expanded = pd.concat([beacon, *[comparators.assign(k=k) for k in COMPONENTS]], ignore_index=True)
    expanded["k"] = expanded.k.astype(int)
    saved = bundle.arrays(bundle.manifest["prepared"]["bootstrap"])
    endpoints, comparisons = summarize(expanded, saved)
    return dict(units=frame.assign(k=frame.k.astype("Int64")), endpoints=endpoints, comparisons=comparisons, inputs=sorted(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=RELEASE / "results" / "factor_components")
    args = parser.parse_args()
    bundle = Bundle(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    result = tables(bundle)
    for name in ("units", "endpoints", "comparisons"):
        result[name].to_csv(args.out / f"{name}.csv", index=False)
    best = strongest(result["comparisons"])
    (args.out / "manifest.json").write_text(json.dumps(dict(components=list(COMPONENTS), reference=REFERENCE, seed=SEED,
        endpoints=int(result["endpoints"].endpoint.nunique()), compared_endpoints=int(best.size), strongest=best.to_dict(),
        inputs=result["inputs"], bundle_manifest_sha256=bundle.manifest_sha256), indent=2) + "\n")


if __name__ == "__main__":
    main()
