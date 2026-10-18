import numpy as np
from beacon.features import expression_features
from experiments.fixed_pools import fit_features, predict_pairs, seed_optimization


def score_all(projected, model, likelihood, genes, tfs, device):
    n = len(genes)
    scores = np.empty((len(tfs), n), dtype=np.float32)
    for row, source in enumerate(tfs):
        for start in range(0, n, 4096):
            targets = np.arange(start, min(n, start + 4096))
            edges = np.column_stack([np.full(len(targets), source), targets])
            scores[row, targets], _ = predict_pairs(projected, model, likelihood, edges, device=device, batch_size=4096, variance=False)
    return scores


def fit(data, *, seed, components=64, features=None, inducing_points=500, device="cuda"):
    seed_optimization(seed)
    if features is None:
        features = expression_features(np.asarray(data["expression"], dtype=np.float32), components=components)
    training, projected, model, likelihood = fit_features(features, data["train"], seed=seed, inducing_points=inducing_points, device=device)
    return {"beacon": {"values": score_all(projected, model, likelihood, data["genes"], data["tf_indices"], device)}}, training.training_log


def fit_frozen_features(bundle, data, *, seed, inducing_points=500, device="cpu"):
    features = bundle.array(bundle.manifest["prepared"]["factor_cache"]["features"])
    seed_optimization(seed)
    training, projected, model, likelihood = fit_features(features, data["train"], seed=seed, inducing_points=inducing_points, device=device)
    return {"beacon": {"values": score_all(projected, model, likelihood, data["genes"], data["tf_indices"], device)}}, training.training_log


if __name__ == "__main__":
    from experiments.run import main
    main("rpe1")
