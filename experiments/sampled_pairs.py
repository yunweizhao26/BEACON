"""Frozen sampled-pair prediction path."""
import numpy as np
from beacon.features import expression_features
from experiments.fixed_pools import seed_optimization, fit_features, predict_pairs

def fit(expression, split, *, seed=42, components=64, pca=False, features=None,
        inducing_points=500, device="cuda"):
    seed_optimization(seed)
    features = expression_features(expression, components=components, pca=pca) if features is None else features
    training, projected, model, likelihood = fit_features(features, split["train"], seed=seed,
        inducing_points=inducing_points, device=device)
    edges = np.argwhere(split["test"] != -1)
    labels = split["test"][tuple(edges.T)].astype(np.int8)
    scores, _ = predict_pairs(projected, model, likelihood, edges, device=device, variance=False)
    result = {"edges": edges, "labels": labels, "beacon": scores}
    return {"predictions": result}, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("sampled_pairs")
