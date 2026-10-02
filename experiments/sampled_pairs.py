"""Frozen sampled-pair and all-setting prediction paths."""
import numpy as np
from beacon.features import expression_features
from beacon.pairs import decoder_scores
from experiments.fixed_pools import seed_optimization, fit_features, predict_pairs

def fit(expression, split, *, seed=42, components=64, pca=False, features=None,
        inducing_points=500, device="cuda", all_settings=False):
    seed_optimization(seed)
    features = expression_features(expression, components=components, pca=pca) if features is None else features
    training, projected, model, likelihood = fit_features(features, split["train"], seed=seed,
        inducing_points=inducing_points, device=device, requested_encoder_epochs=50 if all_settings else 100,
        requested_gp_epochs=50 if all_settings else 200)
    edges = np.argwhere(split["test"] != -1)
    labels = split["test"][tuple(edges.T)].astype(np.int8)
    scores, _ = predict_pairs(projected, model, likelihood, edges, device=device, variance=False)
    result = {"edges": edges, "labels": labels, "gp" if all_settings else "beacon": scores}
    if all_settings:
        result["decoder"] = decoder_scores(training.encoder, features, edges)
    return {"predictions": result}, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("sampled_pairs")
