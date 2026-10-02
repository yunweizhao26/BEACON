"""Label-aware evaluation with mandatory calibrated response support."""
import math
import json
import numpy as np
import pandas as pd
from evaluation.metrics import binary_metrics, paired, top_weights

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

def score_matrices(bundle, experiment):
    context = experiment["condition"].get("context", "rpe1")
    reference = experiment["reference"]
    prepared = bundle.manifest["prepared"].get(context, {})
    training = bundle.arrays(prepared["training"] if context == "rpe1" else reference["training"])
    genes, tfs = training["genes"], training["tf_indices"]
    shape = (len(tfs), len(genes))
    scores = {"BEACON": bundle.array(reference["beacon"])}
    for role, name in (("degree_logistic", "Degree logistic"), ("learned_logistic", "Learned logistic")):
        if role in reference:
            scores[name] = bundle.array(reference[role])
    names = {"gnnlink": "GNNLink", "reggain": "RegGAIN", "reggain_alternative_orientation": "RegGAIN alternative orientation"}
    for comparator in bundle.manifest["comparators"]:
        if comparator["suite"] == ("rpe1" if context == "rpe1" else "external") and comparator.get("context", "rpe1") == context:
            values = bundle.array(comparator["artifacts"]["scores"])
            if values.shape == (len(genes),):
                values = np.broadcast_to(values, shape)
            scores[names.get(comparator["method"], comparator["method"])] = values
    if context == "k562":
        for control in ("graph_only", "permuted"):
            other, = [e for e in bundle.manifest["experiments"] if e["suite"] == "external" and e["condition"]["context"] == context and e["condition"]["control"] == control]
            other_train = bundle.arrays(other["reference"]["training"])
            for key in ("genes", "tf_indices", "train"):
                if not np.array_equal(training[key], other_train[key]):
                    raise ValueError("External control training differs")
            scores[control] = bundle.array(other["reference"]["beacon"])
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
    experiments = [e for e in bundle.manifest["experiments"] if e["suite"] in ("external", "rpe1") and e["condition"]["control"] == "beacon"]
    for experiment in experiments:
        context = experiment["condition"].get("context", "rpe1")
        train, scores, discovery = score_matrices(bundle, experiment)
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
        if context == "rpe1":
            order = np.sort(primary["tf_genes"])
            saved = bundle.arrays(bundle.manifest["prepared"]["bootstrap"])
            indices = saved["indices"] if np.array_equal(saved["tf_genes"], order) else np.random.default_rng(9071).integers(len(order), size=(10000, len(order)))
            summary = {"methods": {}, "beacon_minus": {}}
            for metric in ("top100_mean_abs_effect", "top100_large_effect_yield", "average_precision", "auprc_trapezoid"):
                wide = frame.pivot(index="tf", columns="method", values=metric).loc[order]
                if wide.isna().any().any():
                    raise ValueError("Incomplete RPE1 comparator cohort")
                for method in wide.columns:
                    summary["methods"].setdefault(method, {})[metric] = float(wide[method].mean())
                    if method != "BEACON":
                        summary["beacon_minus"].setdefault(method, {})[metric] = paired(wide.BEACON - wide[method], indices=indices)
            (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    support, bound = response["support"][j], data["primary"][i]
    def count(name, selected):
        return dict(method=name, targets=len(selected), response=int(support[selected].sum()), bound=int(bound[selected].sum()),
            both=int((support & bound)[selected].sum()), neither=int((~support & ~bound)[selected].sum()),
            response_only=int((support & ~bound)[selected].sum()), binding_only=int((~support & bound)[selected].sum()))
    counts = [count("population", idx)]
    for method, matrix in scores.items():
        top = idx[np.argsort(-matrix[i, idx], kind="stable")[:100]]
        counts.append(dict(count(method, top), joint_genes=";".join(train["symbols"][top[(support & bound)[top]]]),
            ties_at_cutoff=int((matrix[i, idx] == matrix[i, top[-1]]).sum())))
    pd.DataFrame(counts).to_csv(destination / "dnmt1_joint_top100.csv", index=False)

def assay_tables(rows, group_keys, out, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_tf_metrics.csv", index=False)
    group = frame.groupby([*group_keys, "method"], dropna=False)
    aggregate = group.agg(tfs=("tf", "size"), tfs_with_response=("positives", lambda x: int((x > 0).sum())),
                         targets=("pairs", "sum"), responses=("positives", "sum"),
                         macro_ap_all_tf=("average_precision", "mean"), macro_auprc_trapezoid=("auprc_trapezoid", "mean"),
                         macro_auroc_defined=("auroc", "mean"), macro_prevalence=("prevalence", "mean"),
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
