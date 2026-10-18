import math
import json
import numpy as np
import pandas as pd
from evaluation.metrics import binary_metrics, expected_random_ap, paired, top_weights
from evaluation import seeds

CONTEXTS = ("k562", "tcell_resting", "tcell_stimulated", "rpe1")

def label_bundle(bundle, context, genes, tf_genes, *, definition=None):
    spec = bundle.manifest["labels"][context]
    if not spec.get("support_key"):
        raise ValueError("An explicit calibrated support key is required")
    calibration = bundle.json(spec["calibration"])
    if not calibration.get("passed") or calibration["mean_per_gene_yes_rate"] > .001:
        raise ValueError(f"Response calibration failed for {context}")
    if context.startswith("tcell"):
        fakes = calibration["per_fake"]
        if calibration["fakes"] != 200 or [r["task"] for r in fakes] != list(range(200)):
            raise ValueError("Incomplete T-cell null calibration")
        rate = float(np.mean(np.asarray([r["yes"] for r in fakes]) / np.asarray([r["tested"] for r in fakes])))
    else:
        if calibration["fake_knockdowns"] != 200 or calibration["scope"] != "per_perturbation":
            raise ValueError("Incorrect AD calibration family")
        rates = calibration["per_fake_yes_rates"]
        if len(rates) != 200:
            raise ValueError("Incomplete AD null calibration")
        rate = math.fsum(rates) / len(rates)
    if rate != calibration["mean_per_gene_yes_rate"]:
        raise ValueError("Calibration arithmetic differs from its receipt")
    data = bundle.arrays(spec["artifact"])
    if not np.array_equal(data["target_genes"], genes):
        raise ValueError("Response target axis differs from score matrix")
    if len(set(map(str, data["tf_genes"]))) != len(data["tf_genes"]):
        raise ValueError("Duplicate response regulator")
    lookup = {str(g): i for i, g in enumerate(tf_genes)}
    selected = np.array([lookup[str(g)] for g in data["tf_genes"]], dtype=int)
    support_key = spec["support_key"] if definition is None else spec["sensitivity_definitions"][definition]["support_key"]
    tested, support = data["tested"], data[support_key]
    shape = (len(selected), len(genes))
    if tested.shape != shape or support.shape != shape or not np.isin(tested, [0, 1]).all() or not np.isin(support, [0, 1]).all():
        raise ValueError("Invalid response support or tested mask")
    tested = tested.astype(bool)
    if context == "rpe1" and not np.isfinite(data["standardized_effects"][tested]).all():
        raise ValueError("Invalid RPE1 effects")
    return dict(data, selected=selected, tested=tested, support=support.astype(bool) & tested)

def context_fits(bundle, context, control="beacon"):
    if context == "rpe1":
        return seeds.experiments(bundle, "rpe1", control=control)
    return seeds.experiments(bundle, "external", context=context, control=control)


def ensemble_matrix(bundle, fits, role):
    return seeds.average([bundle.array(fit["reference"][role]) for fit in fits])


def score_matrices(bundle, context):
    fits = context_fits(bundle, context)
    reference = fits[0]["reference"]
    prepared = bundle.manifest["prepared"].get(context, {})
    training = bundle.arrays(prepared["training"] if context == "rpe1" else reference["training"])
    genes, tfs = training["genes"], training["tf_indices"]
    shape = (len(tfs), len(genes))
    scores = {"BEACON": ensemble_matrix(bundle, fits, "beacon")}
    for role, name in (("degree_logistic", "Degree logistic"), ("learned_logistic", "Learned logistic")):
        if role in reference:
            scores[name] = ensemble_matrix(bundle, fits, role)
    names = {"gnnlink": "GNNLink", "reggain": "RegGAIN", "reggain_alternative_orientation": "RegGAIN alternative orientation",
             "inferelator": "Inferelator"}
    suite = "rpe1" if context == "rpe1" else "external"
    methods = sorted({c["method"] for c in bundle.manifest["comparators"] if c["suite"] == suite and c.get("context", "rpe1") == context},
                     key=lambda m: [c.get("method") for c in bundle.manifest["comparators"]].index(m))
    for method in methods:
        keys = {} if context == "rpe1" else dict(context=context)
        values = seeds.average([bundle.array(c["artifacts"]["scores"]) for c in seeds.comparators(bundle, suite, method, **keys)])
        if values.shape == (len(genes),):
            values = np.broadcast_to(values, shape)
        scores[names.get(method, method)] = values
    if context == "k562":
        for control in ("graph_only", "permuted"):
            other = context_fits(bundle, context, control)
            other_train = bundle.arrays(other[0]["reference"]["training"])
            for key in ("genes", "tf_indices", "train"):
                if not np.array_equal(training[key], other_train[key]):
                    raise ValueError("External control training differs")
            scores[control] = ensemble_matrix(bundle, other, "beacon")
    for method, values in scores.items():
        if values.shape != shape:
            raise ValueError(f"Score axis mismatch: {method}")
    positive = training["train"] == 1
    scores["Prior target in-degree"] = np.broadcast_to(positive.sum(axis=0), shape)
    scores["Prior regulator out-degree"] = np.broadcast_to(positive.sum(axis=1)[tfs, None], shape)
    discovery = bundle.array(prepared["discovery_mask"] if context == "rpe1" else reference["discovery_mask"])
    expected = training["train"][tfs] == -1
    expected[np.arange(len(tfs)), tfs] = False
    if not np.array_equal(discovery, expected):
        raise ValueError("Discovery mask violates training/self exclusions")
    return training, scores, discovery


def evaluate(bundle, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for context in CONTEXTS:
        train, scores, discovery = score_matrices(bundle, context)
        genes, tfs = train["genes"], train["tf_indices"]
        primary = label_bundle(bundle, context, genes, genes[tfs])
        mask = discovery[primary["selected"]] & primary["tested"]
        rows, pooled = [], []
        definitions = [None, *bundle.manifest["labels"][context]["sensitivity_definitions"]] if context != "rpe1" else [None]
        for definition in definitions:
            labels = primary if definition is None else label_bundle(bundle, context, genes, genes[tfs], definition=definition)
            name = definition or ("primary" if context.startswith("tcell") else "configured")
            for method, matrix in scores.items():
                pooled.append(dict(condition=context, definition=name, method=method, **binary_metrics(labels["support"][mask], matrix[labels["selected"]][mask])))
            for i, tf in enumerate(labels["tf_genes"]):
                for method, matrix in scores.items():
                    values = matrix[labels["selected"][i], mask[i]]
                    extra = {}
                    if context == "rpe1":
                        weights = top_weights(values)
                        extra = dict(top100_mean_abs_effect=float(weights @ np.abs(labels["standardized_effects"][i, mask[i]]) / 100),
                                     top100_large_effect_yield=float(weights @ labels["support"][i, mask[i]] / 100))
                    rows.append(dict(condition=context, definition=name, tf=tf, method=method, **extra,
                                     **binary_metrics(labels["support"][i, mask[i]], values)))
        out = destination / context
        frame, _ = assay_tables(rows, ["condition", "definition"], out)
        pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
        receipt = dict(label_sha256=bundle.assets[bundle.manifest["labels"][context]["artifact"]]["original_sha256"], seeds=list(seeds.SEEDS))
        topk_comparisons(frame, ["condition", "definition"]).to_csv(out / "paired_topk.csv", index=False)
        if context.startswith("tcell"):
            signflip_table(frame, ["condition", "definition"]).to_csv(out / "signflip_exact.csv", index=False)
        if context == "rpe1":
            order = np.sort(primary["tf_genes"])
            saved = bundle.arrays(bundle.manifest["prepared"]["bootstrap"])
            indices = saved["indices"] if np.array_equal(saved["tf_genes"], order) else np.random.default_rng(9071).integers(len(order), size=(10000, len(order)))
            summary = dict(receipt, methods={}, beacon_minus={})
            for metric in ("top100_mean_abs_effect", "top100_large_effect_yield", "average_precision", "auprc_trapezoid"):
                wide = frame.pivot(index="tf", columns="method", values=metric).loc[order]
                if wide.isna().any().any():
                    raise ValueError("Incomplete RPE1 comparator cohort")
                for method in wide.columns:
                    summary["methods"].setdefault(method, {})[metric] = float(wide[method].mean())
                    if method != "BEACON":
                        summary["beacon_minus"].setdefault(method, {})[metric] = paired(wide.BEACON - wide[method], indices=indices)
            (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        else:
            (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if context == "k562":
            binding(bundle, train, scores, discovery, primary, destination / "binding")

def binding(bundle, train, scores, discovery, response, destination):
    data = bundle.arrays(bundle.manifest["prepared"]["binding"])
    genes, tfs = train["genes"], train["tf_indices"]
    if not np.array_equal(data["genes"], genes) or not np.array_equal(data["tf_indices"], tfs):
        raise ValueError("Binding axes differ from score matrix")
    lookup = {str(g): i for i, g in enumerate(response["tf_genes"])}
    scores = {k: v for k, v in scores.items() if k not in ("graph_only", "permuted")}
    rows, parts, counts = [], {}, []
    for window in ("primary", "wide"):
        for i in np.flatnonzero(data["assayed"]):
            tf = genes[tfs[i]]
            allowed = discovery[i] & data["annotated"]
            outcomes = [("binding", allowed, data[window][i])]
            if str(tf) in lookup:
                j = lookup[str(tf)]
                mask = allowed & response["tested"][j]
                support = response["support"][j]
                outcomes.append(("binding_and_response", mask, data[window][i] & support))
                counts.append(dict(window=window, tf=tf, tested_pairs=int(mask.sum()), both=int((mask & data[window][i] & support).sum())))
            for assay, eligible, support in outcomes:
                for method, matrix in scores.items():
                    rows.append(dict(window=window, assay=assay, tf=tf, method=method, **binary_metrics(support[eligible], matrix[i, eligible])))
                    parts.setdefault((window, assay, method), []).append((support[eligible], matrix[i, eligible]))
    assay_tables(rows, ["window", "assay"], destination)
    pd.DataFrame(counts).to_csv(destination / "cross_assay_counts.csv", index=False)
    pd.DataFrame([dict(window=w, assay=a, method=m, **binary_metrics(np.concatenate([p[0] for p in values]), np.concatenate([p[1] for p in values])))
                  for (w, a, m), values in parts.items()]).to_csv(destination / "pooled_metrics.csv", index=False)
    i, = [i for i in np.flatnonzero(data["assayed"]) if train["symbols"][tfs[i]] == "DNMT1"]
    j = lookup[str(genes[tfs[i]])]
    eligible = discovery[i] & data["annotated"] & response["tested"][j]
    idx = np.flatnonzero(eligible)
    support, bound = response["support"][j][idx], data["primary"][i][idx]
    counts = [dict(method="population", targets=len(idx), **joint_counts(np.ones(len(idx)), support, bound))]
    for method, matrix in scores.items():
        values = matrix[i, idx]
        weights = top_weights(values, 100)
        boundary = values == np.partition(values, len(values) - 100)[-100]
        counts.append(dict(method=method, targets=100, **joint_counts(weights, support, bound),
            joint_genes=";".join(train["symbols"][idx][(weights > 0) & support & bound]), ties_at_cutoff=int(boundary.sum())))
    pd.DataFrame(counts).to_csv(destination / "dnmt1_joint_top100.csv", index=False)

def joint_counts(weights, support, bound):
    support, bound = np.asarray(support, bool), np.asarray(bound, bool)
    weighted = lambda selected: float(np.asarray(weights, float) @ selected.astype(float))
    return dict(response=weighted(support), bound=weighted(bound), both=weighted(support & bound), neither=weighted(~support & ~bound),
                response_only=weighted(support & ~bound), binding_only=weighted(~support & bound))

def exact_signflip(delta):
    import itertools
    delta = np.asarray(delta, float)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(delta))))
    null = (signs * delta).mean(axis=1)
    observed, tol = delta.mean(), 1e-12
    return dict(mean=float(observed), n=len(delta), patterns=len(null), p_two_sided=float(np.mean(np.abs(null) >= abs(observed) - tol)),
                p_greater=float(np.mean(null >= observed - tol)), p_less=float(np.mean(null <= observed + tol)))


def signflip_table(frame, group_keys):
    rows = []
    for key, part in frame.groupby(group_keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in ("average_precision", "top100_supported_fraction"):
            wide = part.pivot(index="tf", columns="method", values=metric)
            for method in wide.columns:
                if method != "BEACON":
                    rows.append(dict(zip(group_keys, key), metric=metric, comparator=method, **exact_signflip(wide.BEACON - wide[method])))
    return pd.DataFrame(rows)


def topk_comparisons(frame, group_keys, seed=42):
    rows = []
    rng = np.random.default_rng(seed)
    for key, part in frame.groupby(group_keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for k in (10, 25, 50, 100, 200):
            wide = part.pivot(index="tf", columns="method", values=f"top{k}_supported_fraction")
            indices = rng.integers(len(wide), size=(10000, len(wide)))
            for method in wide.columns:
                if method != "BEACON":
                    r = paired(wide.BEACON - wide[method], indices=indices)
                    rows.append(dict(zip(group_keys, key), k=k, comparator=method, beacon_minus_comparator=r["mean"], ci_low=r["ci_low"],
                                     ci_high=r["ci_high"], beacon_wins=r["wins"], comparator_wins=r["losses"], ties=r["ties"]))
    return pd.DataFrame(rows)


def assay_tables(rows, group_keys, out, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["expected_random_ap"] = expected_random_ap(frame.pairs, frame.positives)
    frame.to_csv(out / "per_tf_metrics.csv", index=False)
    group = frame.groupby([*group_keys, "method"], dropna=False)
    aggregate = group.agg(tfs=("tf", "size"), tfs_with_response=("positives", lambda x: int((x > 0).sum())),
                         targets=("pairs", "sum"), responses=("positives", "sum"),
                         macro_ap_all_tf=("average_precision", "mean"), macro_auprc_trapezoid=("auprc_trapezoid", "mean"),
                         macro_auroc_defined=("auroc", "mean"), macro_prevalence=("prevalence", "mean"),
                         macro_expected_random_ap=("expected_random_ap", "mean"),
                         **{f"top{k}_response_fraction": (f"top{k}_supported_fraction", "mean") for k in (10, 25, 50, 100, 200)}).reset_index()
    aggregate.to_csv(out / "aggregate.csv", index=False)
    comparisons = []
    rng = np.random.default_rng(seed)
    for key, part in frame.groupby(group_keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        wide = part.pivot(index="tf", columns="method", values="average_precision")
        if wide.isna().any().any():
            raise ValueError("Methods must cover the identical complete regulator cohort")
        indices = rng.integers(len(wide), size=(10000, len(wide)))
        for method in wide.columns:
            if method == "BEACON":
                continue
            result = paired(wide.BEACON - wide[method], indices=indices)
            comparisons.append(dict(zip(group_keys, key), comparator=method,
                beacon_minus_comparator_macro_ap=result["mean"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                beacon_wins=result["wins"], comparator_wins=result["losses"], ties=result["ties"], tfs=len(wide)))
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    return frame, aggregate
