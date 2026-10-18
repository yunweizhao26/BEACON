import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from beacon.data import Bundle
from comparators import adapters as models

CONTEXTS = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
UPSTREAM = dict(gnnlink="GNNLink", genelink="GENELink", gclink="GCLink", reggain="RegGAIN", scregnet="scRegNet")


def load_inputs(bundle, args):
    if args.kind in ("fixed_pools", "expression"):
        entry, = [e for e in bundle.manifest["experiments"] if e["suite"] == "fixed_pools"
                  and e["condition"]["dataset"] == args.dataset and e["condition"]["split_seed"] == args.split_seed
                  and e["condition"]["coverage"] == args.coverage and e["condition"]["control"] == "beacon"
                  and e["condition"]["seed"] == 42]
        split = bundle.arrays(entry["reference"]["split"])
        dataset, = [d for d in bundle.manifest["datasets"] if d["dataset"] == args.dataset]
        frame = bundle.frame(dataset["expression"], index_col=0)
        if args.kind == "expression":
            from experiments.expression import control_expression
            values, _ = control_expression(frame.to_numpy(dtype=np.float64), args.control)
            frame = pd.DataFrame(values, index=frame.index, columns=frame.columns)
        pools = {name: np.argwhere(split[name] != -1) for name in ("test", "valid")}
        labels = {name: split[name][tuple(edges.T)] for name, edges in pools.items()}
        return split, frame, pools, labels, args.seed, None

    if args.kind == "trrust":
        manifest = bundle.manifest["trrust"]
        task = manifest["tasks"][args.task]
        frame = bundle.frame(task["expression"], index_col=0)
        genes = np.asarray(frame.index.astype(str))
        if genes.tolist() != manifest["genes"]:
            raise ValueError("TRRUST gene order changed")
        lookup = {gene: i for i, gene in enumerate(genes)}
        prior = bundle.frame(manifest["prior"], dtype=str)
        by_id = {i: (lookup[s], lookup[t]) for i, s, t in prior[["edge_id", "source", "target"]].itertuples(index=False, name=None)}
        positive = np.asarray([by_id[i] for i in manifest["train_positive_edge_ids"]], dtype=np.int64)
        unlabeled = np.asarray(manifest["train_unlabeled"], dtype=np.int64)
        train = np.full((len(genes), len(genes)), -1, dtype=np.int8)
        train[tuple(positive.T)], train[tuple(unlabeled.T)] = 1, 0
        evaluation = manifest["evaluation"]
        edges = np.asarray([(int(r["source_index"]), int(r["target_index"])) for r in evaluation])
        if set(map(tuple, edges)) & set(map(tuple, np.argwhere(train >= 0))):
            raise ValueError("Evaluation pairs overlap training")
        return dict(train=train, genes=genes), frame, {"test": edges}, {}, int(task["seed"]), evaluation

    prepared = bundle.manifest["prepared"][args.context]
    if args.kind == "temporal":
        from preparation.temporal import saved
        training = bundle.arrays(saved(bundle, args.context, args.cutoff)["training"])
    elif args.context == "rpe1":
        training = bundle.arrays(prepared["training"])
    else:
        entry, = [e for e in bundle.manifest["experiments"] if e["suite"] == "external"
                  and e["condition"]["context"] == args.context and e["condition"]["control"] == "beacon" and e["condition"]["seed"] == 42]
        training = bundle.arrays(entry["reference"]["training"])
    expression = bundle.arrays(prepared["expression"])
    if not np.array_equal(training["genes"], expression["genes"]):
        raise ValueError("External expression gene order changed")
    genes, tfs = training["genes"], training["tf_indices"]
    values = expression["expression"]
    frame = pd.DataFrame(values if args.method == "gnnlink" else np.asarray(values, dtype=np.float32),
                         index=genes.astype(str), columns=[f"cell{i}" for i in range(values.shape[1])])
    pairs = np.column_stack([np.repeat(tfs, len(genes)), np.tile(np.arange(len(genes)), len(tfs))])
    return training, frame, {"test": pairs}, {}, args.seed, None


def foundation(bundle, args, genes):
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ("genes", "embeddings")}
    else:
        context = CONTEXTS[args.dataset] if args.kind == "fixed_pools" else args.context
        records = bundle.manifest.get("comparator_training", {}).get("geneformer_embeddings", {})
        if context not in records:
            raise ValueError(f"Missing Geneformer embeddings for {context}; add the bundle record or pass --embeddings")
        data = bundle.arrays(records[context])
    if not np.array_equal(data["genes"].astype(str), genes.astype(str)):
        raise ValueError("Geneformer embedding gene order differs from training")
    values = data["embeddings"]
    if values.ndim != 2 or len(values) != len(genes) or not np.isfinite(values).all():
        raise ValueError("Invalid Geneformer embedding matrix")
    return values


def select_and_refit(args, split, frame, pools, seed, embeddings=None):
    import torch
    device = torch.device(args.device)
    train, genes = split["train"], split["genes"]
    fit_train, held_edges, held_labels = models.internal_split(train)
    rule = models.RULES[args.method]
    result = dict(rule=rule, validation_pairs=len(held_labels), validation_positives=int(held_labels.sum()))
    matrix_kind = args.kind in ("external", "temporal")
    tfs = split.get("tf_indices")
    if args.method == "reggain" and matrix_kind:
        rows = {gene: i for i, gene in enumerate(tfs)}
        snapshots = models.reggain_snapshots(args.source, fit_train, genes, tfs, frame.to_numpy(), rule["grid"], device, seed)
        history = [dict(epoch=epoch, validation_ap=float(models.average_precision_score(
            held_labels, snapshots[epoch][0][[rows[i] for i in held_edges[:, 0]], held_edges[:, 1]]))) for epoch in rule["grid"]]
        epoch = max(history, key=lambda h: (h["validation_ap"], -h["epoch"]))["epoch"]
        del snapshots
        matrices = models.reggain_snapshots(args.source, train, genes, tfs, frame.to_numpy(), [epoch], device, seed)[epoch]
        values = {"test": matrices[0].ravel().astype(float)}
        if args.kind == "external" and args.context != "rpe1":
            values["alternative"] = matrices[1].ravel().astype(float)
        result.update(history=history, epochs=epoch)
        return values, result

    def fit(matrix, epochs, monitor=None):
        data = models.inputs(matrix, genes.tolist(), frame)
        if args.method == "gnnlink":
            if matrix_kind:
                data["expression"] = frame.to_numpy()
            values = models.gnnlink(args.source, data, seed, epochs, monitor)
            return lambda pairs: values[pairs[:, 0], pairs[:, 1]].astype(float)
        if args.method in ("genelink", "gclink"):
            return getattr(models, args.method)(args.source, data, seed, epochs, device, monitor)
        if args.method == "reggain":
            return models.reggain(args.source, matrix, genes, frame, seed, epochs, device)
        return models.train_scregnet(args.source, matrix, genes, tfs, frame, embeddings, device, epochs, monitor, seed=seed)

    fallback = held_labels.sum() < models.MIN_VALIDATION_POSITIVES and (
        args.kind == "fixed_pools" and args.method != "scregnet" or args.method == "gnnlink" and matrix_kind)
    history, checkpoint = [], None
    if fallback:
        epoch = rule["released"]
    elif not held_labels.any() or held_labels.all():
        raise ValueError("Internal validation requires both label classes")
    elif args.method == "reggain":
        for epoch in rule["grid"]:
            score = fit(fit_train, epoch)
            history.append(dict(epoch=epoch, validation_ap=float(models.average_precision_score(held_labels, score(held_edges)))))
        epoch = max(history, key=lambda h: (h["validation_ap"], -h["epoch"]))["epoch"]
    else:
        monitor = models.Monitor(rule, held_edges, held_labels)
        selected = fit(fit_train, rule["max"], monitor)
        if args.method == "scregnet":
            _, checkpoint = selected
        epoch, history = monitor.best(), monitor.history
    result.update(fallback=bool(fallback), history=history, epochs=epoch)
    early = models.score_all(checkpoint, pools["test"]) if args.method == "scregnet" and args.kind == "temporal" else None
    scorer = fit(train, epoch)
    if args.method == "scregnet":
        scorer, _ = scorer
    scores = {name: scorer(edges) for name, edges in pools.items()}
    if early is not None:
        values = scores["test"]
        use_refit = values.std() >= models.COLLAPSE_STD
        result.update(refit_std=float(values.std()), early_stopped_std=float(early.std()),
                      used="refit" if use_refit else "early-stopped selection model")
        if not use_refit:
            scores["test"] = early
        result["collapsed"] = bool(scores["test"].std() < models.COLLAPSE_STD)
    return scores, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("fixed_pools", "expression", "external", "temporal", "trrust"))
    p.add_argument("--method", choices=tuple(models.RULES), required=True)
    p.add_argument("--source", type=Path, required=True, help="Checkout of the upstream commit in comparators/manifest.json")
    p.add_argument("--config", type=Path)
    p.add_argument("--dataset", type=int, choices=tuple(CONTEXTS), default=1501)
    p.add_argument("--coverage", type=float, choices=(.05, .1, .2, .4, .8), default=.8)
    p.add_argument("--split-seed", type=int, choices=(42, 14, 100), default=42)
    p.add_argument("--control", choices=("real", "cell_shuffled", "gene_permuted", "random"), default="real")
    p.add_argument("--context", choices=("k562", "tcell_resting", "tcell_stimulated", "rpe1"))
    p.add_argument("--cutoff", type=int, choices=(2000, 2005, 2010))
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--seed", type=int, choices=(42, 14, 100), default=42,
                   help="Optimization seed; the held-out pairs that select the training length keep seed 42")
    p.add_argument("--embeddings", type=Path, help="Geneformer NPZ with genes and embeddings, in saved gene order")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.kind in ("external", "temporal") and not args.context or args.kind == "temporal" and args.cutoff is None:
        p.error("External/temporal fits require --context; temporal also requires --cutoff")
    if args.kind == "trrust" and args.method not in ("gnnlink", "genelink"):
        p.error("Epoch-standard TRRUST fits use GNNLink or GENELink")
    if args.kind == "expression" and args.method not in ("gnnlink", "genelink", "gclink"):
        p.error("Expression controls use GNNLink, GENELink or GCLink")
    if args.method == "scregnet" and args.kind == "fixed_pools" and (args.dataset != 1709 or args.coverage != .8):
        p.error("The released scRegNet fixed-pool setting is mHSC-E at 80% prior")
    args.source = args.source.resolve()
    upstream = json.loads(Path(__file__).with_name("manifest.json").read_text())["upstreams"][UPSTREAM[args.method]]
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != upstream["commit"]:
        raise ValueError(f"Expected upstream commit {upstream['commit']}, got {commit}")
    status = subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True,
                                     env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if status:
        raise ValueError("Upstream checkout has local changes; use the recorded clean checkout")
    bundle = Bundle(args.config)
    split, frame, pools, labels, seed, evaluation = load_inputs(bundle, args)
    if frame.index.astype(str).tolist() != split["genes"].astype(str).tolist():
        raise ValueError("Expression gene order differs from training")
    embeddings = foundation(bundle, args, split["genes"]) if args.method == "scregnet" else None
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    scores, summary = select_and_refit(args, split, frame, pools, seed, embeddings)
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Nonfinite comparator scores")
    if args.kind == "trrust":
        payload = dict(context="flu", method=args.method, method_seed=seed,
                       records=[dict(source=r["source"], target=r["target"], analysis_role=r["analysis_role"], probability=float(value))
                                for r, value in zip(evaluation, scores["test"])])
        (args.out / "records.json").write_text(json.dumps(payload) + "\n")
    elif args.kind in ("external", "temporal"):
        values = scores["test"].astype(np.float32).reshape(len(split["tf_indices"]), len(split["genes"]))
        np.save(args.out / f"{args.method}.npy", values)
        if "alternative" in scores:
            np.save(args.out / "reggain_alternative_orientation.npy", scores["alternative"].astype(np.float32).reshape(values.shape))
    else:
        values = dict(edges=pools["test"], labels=labels["test"], **{args.method: scores["test"]},
                      validation_edges=pools["valid"], validation_labels=labels["valid"], validation_scores=scores["valid"])
        np.savez_compressed(args.out / "predictions.npz", **values)
    summary.update(method=args.method, kind=args.kind, dataset=args.dataset, coverage=args.coverage,
                   split_seed=args.split_seed, context=args.context, cutoff=args.cutoff, control=args.control,
                   seed=seed, device=args.device, upstream_commit=commit, bundle_manifest_sha256=bundle.manifest_sha256,
                   seconds=time.perf_counter() - started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
