import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from beacon.pairs import classifier
from evaluation.metrics import binary_metrics, degree_features, expected_random_ap, paired, popularity
from evaluation.seeds import TRRUST_SEEDS, average, experiments

COMPARATORS = (("GNNLink", "gnnlink"), ("GENELink", "genelink"), ("GRNBoost2", "paired_grnboost2"),
               ("GENIE3", "genie3"), ("Inferelator", "inferelator"))
BASELINES = ("Prior target in-degree", "Prior regulator out-degree", "Degree logistic")
UNMODIFIED_MONITOR = ("GRNBoost2 (unmodified monitor)", "grnboost2_unmodified_monitor")
EVALUATIONS = ("all", "overlap_removed")
KEYS = ["source", "target", "analysis_role"]
TOPK = tuple(f"top{k}_supported_fraction" for k in (10, 25, 50, 100, 200))
METRICS = ("auroc", "average_precision", "auprc_trapezoid", "expected_random_ap", "prevalence", *TOPK)
COMPARED = ("auroc", "average_precision", "auprc_trapezoid", "top100_supported_fraction")
SIGN_FLIP = ("auroc", "average_precision", "top100_supported_fraction")
_CACHE = {}


def records(bundle, indices):
    rows = []
    for index in indices:
        value = bundle.json(index)
        if value["context"] != "flu":
            continue
        for row in value["records"]:
            rows.append(dict(method_seed=int(value["method_seed"]), source=row["source"], target=row["target"],
                             analysis_role=row["analysis_role"], score=float(row["probability"])))
    return pd.DataFrame(rows)


def exclude_overlap(frames, original, min_supported=5):
    original_pairs = set(original[["source_genesymbol", "target_genesymbol"]].itertuples(index=False, name=None))
    filtered = {}
    for method, frame in frames.items():
        overlap = np.fromiter(((s, t) in original_pairs for s, t in frame[["source", "target"]].itertuples(index=False, name=None)), dtype=bool, count=len(frame))
        drop = (frame.analysis_role == "supported").to_numpy() & overlap
        filtered[method] = frame.loc[~drop].copy()
    beacon = filtered["BEACON"]
    counts = beacon.loc[beacon.analysis_role == "supported", ["source", "target"]].drop_duplicates().groupby("source").target.nunique()
    eligible = sorted(counts[counts >= min_supported].index)
    return {method: frame[frame.source.isin(eligible)].copy() for method, frame in filtered.items()}


def bootstrap_difference(beacon, baseline, iterations=10000, seed=42):
    sources = sorted(set(beacon) & set(baseline))
    differences = np.asarray([beacon[s] - baseline[s] for s in sources])
    rng = np.random.default_rng(seed)
    boot = np.asarray([float(rng.choice(differences, len(differences), replace=True).mean()) for _ in range(iterations)])
    signs = rng.choice(np.asarray([-1., 1.]), size=(iterations, len(differences)))
    null = (signs * differences).mean(axis=1)
    p = (1 + int(np.sum(null >= differences.mean()))) / (iterations + 1)
    return float(differences.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), float(p)


def evaluated_pairs(section):
    regulators = set(section["regulators"])
    pairs = pd.DataFrame([(r["source"], r["target"], r["analysis_role"]) for r in section["evaluation"] if r["source"] in regulators], columns=KEYS)
    if set(pairs.source) != regulators or pairs.duplicated(["source", "target"]).any() or (pairs.source == pairs.target).any():
        raise ValueError("Evaluated TRRUST regulators need unique, non-self evaluation pairs")
    return pairs


def training_graph(bundle):
    section = bundle.manifest["trrust"]
    index = {gene: i for i, gene in enumerate(section["genes"])}
    prior = bundle.frame(section["prior"], dtype=str)
    by_id = {e: (index[s], index[t]) for e, s, t in prior[["edge_id", "source", "target"]].itertuples(index=False, name=None)}
    positive = np.asarray([by_id[e] for e in section["train_positive_edge_ids"]], dtype=np.int64).reshape(-1, 2)
    unlabeled = np.asarray(section["train_unlabeled"], dtype=np.int64).reshape(-1, 2)
    if set(map(tuple, positive.tolist())) & set(map(tuple, unlabeled.tolist())):
        raise ValueError("A TRRUST pair is both a known edge and an unlabeled pair")
    train = np.full((len(index), len(index)), -1, dtype=np.int8)
    train[tuple(positive.T)], train[tuple(unlabeled.T)] = 1, 0
    return train


def baselines(train, edges):
    scores = {name: values.astype(np.float64) for name, values in popularity(train, edges).items()}
    labeled = np.argwhere(train != -1)
    outgoing, incoming = np.log1p((train == 1).sum(axis=1)), np.log1p((train == 1).sum(axis=0))
    model = classifier(degree_features(labeled, outgoing, incoming), train[tuple(labeled.T)])
    scores["Degree logistic"] = model.predict_proba(degree_features(edges, outgoing, incoming))[:, 1]
    return scores


def aligned(bundle, indices, pairs, full):
    by_seed = {}
    target = pd.MultiIndex.from_frame(pairs[KEYS])
    for index in indices:
        value = bundle.json(index)
        if value["context"] != "flu":
            raise ValueError("TRRUST records must be the influenza context")
        frame = pd.DataFrame([(r["source"], r["target"], r["analysis_role"], float(r["probability"])) for r in value["records"]], columns=[*KEYS, "score"])
        seed = int(value["method_seed"])
        if seed in by_seed or frame.duplicated(KEYS).any() or set(frame[KEYS].itertuples(index=False, name=None)) != full:
            raise ValueError(f"TRRUST record {index} repeats a seed or does not cover the evaluation pairs once")
        by_seed[seed] = frame.set_index(KEYS).score.reindex(target).to_numpy(dtype=np.float64)
    if set(by_seed) != set(TRRUST_SEEDS):
        raise ValueError(f"Expected TRRUST seeds {TRRUST_SEEDS}, found {sorted(by_seed)}")
    return [by_seed[seed] for seed in TRRUST_SEEDS]


def scores(bundle):
    section = bundle.manifest["trrust"]
    pairs = evaluated_pairs(section)
    full = {(r["source"], r["target"], r["analysis_role"]) for r in section["evaluation"]}
    fits = experiments(bundle, "trrust", TRRUST_SEEDS, control="beacon")
    sources = {"BEACON": [fit["reference"]["record"] for fit in fits]}
    for name, key in COMPARATORS:
        sources[name] = section["paired_grnboost2"] if key == "paired_grnboost2" else section["comparators"][key]
    if UNMODIFIED_MONITOR[1] in section["comparators"]:
        sources[UNMODIFIED_MONITOR[0]] = section["comparators"][UNMODIFIED_MONITOR[1]]
    per_seed = {name: aligned(bundle, indices, pairs, full) for name, indices in sources.items()}
    index = {gene: i for i, gene in enumerate(section["genes"])}
    edges = np.column_stack([pairs.source.map(index), pairs.target.map(index)]).astype(np.int64)
    for name, values in baselines(training_graph(bundle), edges).items():
        per_seed[name] = [values] * len(TRRUST_SEEDS)
    inputs = sorted({i for indices in sources.values() for i in indices} | {section["prior"], section["unmasked_prior"]})
    return pairs, per_seed, inputs


def regulator_metrics(pairs, method_scores):
    y = pairs.analysis_role.eq("supported").to_numpy()
    sources = pairs.source.to_numpy()
    groups = [(tf, np.flatnonzero(sources == tf)) for tf in sorted(set(sources))]
    rows = []
    for method, values in method_scores.items():
        for tf, rows_ in groups:
            m = binary_metrics(y[rows_], values[rows_])
            rows.append(dict(method=method, tf=tf, **m, expected_random_ap=float(expected_random_ap(m["pairs"], m["positives"]))))
    return pd.DataFrame(rows)


def ranking(per_tf):
    table = per_tf.groupby("method", sort=False)[list(METRICS)].mean()
    table["ap_over_exact_chance"] = table.average_precision / table.expected_random_ap
    table["ap_over_prevalence"] = table.average_precision / table.prevalence
    counts = per_tf.groupby("method", sort=False).agg(regulators=("tf", "nunique"), supported=("positives", "sum"), pairs=("pairs", "sum"))
    return table.join(counts).reset_index()


def compare(per_tf, methods, *, sign_flip=True):
    wide = {metric: per_tf.pivot(index="tf", columns="method", values=metric) for metric in COMPARED}
    paired_rows, flip_rows = [], []
    for method in methods:
        for metric in COMPARED:
            table = wide[metric]
            paired_rows.append(dict(comparator=method, metric=metric, **paired((table["BEACON"] - table[method]).to_numpy())))
            if sign_flip and metric in SIGN_FLIP:
                d, low, high, p = bootstrap_difference(table["BEACON"].to_dict(), table[method].to_dict())
                flip_rows.append(dict(method=method, metric=metric, difference=d, ci_low=low, ci_high=high, one_sided_signflip_p=p))
    return pd.DataFrame(paired_rows), pd.DataFrame(flip_rows)


def regulator_tests(pairs, fits, permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    tfs = sorted(pairs.source.unique())
    supported = pairs.analysis_role.eq("supported").to_numpy()
    values, blocks = [], []
    for tf in tfs:
        rows = np.flatnonzero(pairs.source.to_numpy() == tf)
        rows = rows[np.argsort(pairs.target.to_numpy()[rows], kind="stable")]
        y = supported[rows]
        values.append(np.mean([binary_metrics(y, fit[rows])["auroc"] for fit in fits]))
        blocks.append((int(y.sum()), len(y), np.vstack([rankdata(fit[rows]) for fit in fits])))
    values = np.asarray(values)
    interval = paired(values, indices=rng.integers(len(values), size=(10000, len(values))))
    observed = values.mean()
    null = np.zeros(permutations)
    for m, n, ranks in blocks:
        for b in range(permutations):
            chosen = rng.permutation(n)[:m]
            null[b] += np.mean((ranks[:, chosen].sum(axis=1) - m * (m + 1) / 2) / (m * (n - m)))
    null /= len(tfs)
    exceeded = int(np.sum(null >= observed))
    return dict(macro_auroc=float(observed), ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                blocked_permutation_p=(1 + exceeded) / (permutations + 1), permutations=permutations, permutations_exceeding=exceeded)


def consensus_spearman(pairs, values):
    wide = pd.DataFrame(dict(source=pairs.source, target=pairs.target, score=values)).pivot(index="source", columns="target", values="score")
    full = np.full(wide.shape, np.nan)
    for k, row in enumerate(wide.to_numpy()):
        ok = np.isfinite(row)
        full[k, ok] = rankdata(row[ok]) / ok.sum()
    rho = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(len(full)):
            others = np.nanmean(np.delete(full, k, axis=0), axis=0)
            ok = np.isfinite(full[k]) & np.isfinite(others)
            rho.append(spearmanr(full[k, ok], others[ok])[0])
    return float(np.median(rho))


def endpoints(bundle, permutations=1000):
    key = (str(bundle.root), bundle.manifest_sha256, permutations)
    if key in _CACHE:
        return _CACHE[key]
    pairs, per_seed, inputs = scores(bundle)
    monitor = per_seed.pop(UNMODIFIED_MONITOR[0], None)
    mean = {name: average(values) for name, values in per_seed.items()}
    methods = [name for name in mean if name != "BEACON"]
    original = bundle.frame(bundle.manifest["trrust"]["unmasked_prior"], sep="\t", dtype=str)
    kept = exclude_overlap({"BEACON": pairs}, original)["BEACON"].index.to_numpy()
    original_pairs = set(original[["source_genesymbol", "target_genesymbol"]].itertuples(index=False, name=None))
    in_original = np.fromiter(((s, t) in original_pairs for s, t in pairs[["source", "target"]].itertuples(index=False, name=None)), dtype=bool, count=len(pairs))
    supported = pairs.analysis_role.eq("supported").to_numpy()
    out = {name: [] for name in ("cohorts", "per_tf_metrics", "pooled_metrics", "ranking", "paired_comparisons", "method_comparison",
                                 "beacon_tests", "consensus", "monitor", "per_seed_per_tf_metrics", "per_seed_ranking",
                                 "per_seed_comparisons", "per_seed_beacon_tests")}
    for evaluation in EVALUATIONS:
        rows = np.arange(len(pairs)) if evaluation == "all" else kept
        part = pairs.iloc[rows].reset_index(drop=True)
        label = dict(evaluation=evaluation)
        out["cohorts"].append(pd.DataFrame([dict(label, regulators=part.source.nunique(), supported=int(part.analysis_role.eq("supported").sum()),
            candidates=len(part), supported_in_original_source=int((supported & in_original).sum()), regulator_names=";".join(sorted(part.source.unique())))]))
        current = {name: values[rows] for name, values in mean.items()}
        per_tf = regulator_metrics(part, current)
        out["per_tf_metrics"].append(per_tf.assign(**label))
        out["pooled_metrics"].append(pd.DataFrame([dict(label, method=name, **binary_metrics(part.analysis_role.eq("supported").to_numpy(), values))
                                                   for name, values in current.items()]))
        consensus = {name: consensus_spearman(part, values) for name, values in current.items()}
        out["ranking"].append(ranking(per_tf).assign(median_consensus_spearman=lambda f: f.method.map(consensus), **label))
        paired_table, flips = compare(per_tf, methods)
        out["paired_comparisons"].append(paired_table.assign(**label))
        out["method_comparison"].append(flips.assign(**label))
        out["beacon_tests"].append(pd.DataFrame([dict(label, fits="seed_mean", **regulator_tests(part, [current["BEACON"]], permutations))]))
        if monitor is not None:
            pair = {"GRNBoost2": current["GRNBoost2"], UNMODIFIED_MONITOR[0]: average(monitor)[rows]}
            means = regulator_metrics(part, pair).groupby("method")[["auroc", "average_precision"]].mean()
            out["monitor"].append(pd.DataFrame([dict(label, metric=metric, corrected=means.loc["GRNBoost2", metric],
                unmodified=means.loc[UNMODIFIED_MONITOR[0], metric], corrected_minus_unmodified=means.loc["GRNBoost2", metric] - means.loc[UNMODIFIED_MONITOR[0], metric])
                for metric in ("auroc", "average_precision")]))
        for j, seed in enumerate(TRRUST_SEEDS):
            single = {name: values[j][rows] for name, values in per_seed.items()}
            seeded = regulator_metrics(part, single)
            out["per_seed_per_tf_metrics"].append(seeded.assign(seed=seed, **label))
            consensus = {name: consensus_spearman(part, values) for name, values in single.items()}
            out["per_seed_ranking"].append(ranking(seeded).assign(median_consensus_spearman=lambda f: f.method.map(consensus), seed=seed, **label))
            out["per_seed_comparisons"].append(compare(seeded, methods, sign_flip=False)[0].assign(seed=seed, **label))
        fits = [values[rows] for values in per_seed["BEACON"]]
        out["per_seed_beacon_tests"].append(pd.DataFrame([dict(label, fits="five_seeds", **regulator_tests(part, fits, permutations))]))
    result = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in out.items()}
    seeds = result["per_seed_ranking"]
    result["per_seed_range"] = (seeds.melt(id_vars=["evaluation", "method", "seed"], value_vars=["auroc", "average_precision", "top100_supported_fraction", "median_consensus_spearman"], var_name="metric")
                                .groupby(["evaluation", "method", "metric"], sort=False).value.agg(minimum="min", maximum="max").reset_index())
    best = []
    for (evaluation, seed), part in seeds.groupby(["evaluation", "seed"], sort=False):
        table = part.set_index("method")
        for metric in ("auroc", "average_precision"):
            others = table[metric].drop("BEACON")
            best.append(dict(evaluation=evaluation, seed=seed, metric=metric, beacon=table.loc["BEACON", metric], highest=table[metric].idxmax(),
                             highest_other=others.idxmax(), highest_other_value=others.max()))
    result["per_seed_best"] = pd.DataFrame(best)
    result["inputs"] = inputs
    _CACHE[key] = result
    return result


def evaluate(bundle, out):
    result = endpoints(bundle)
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        if name != "inputs":
            frame.to_csv(out / f"{name}.csv", index=False)
    return result
