import time
import numpy as np
from beacon.features import expression_features, feature_control
from beacon.pairs import classifier, pair_features
from evaluation.metrics import degree_features
from experiments.fixed_pools import seed_optimization, fit_features, predict_pairs

def fit(data, *, seed=42, random_features=False, permuted_features=False, components=64,
        features=None, inducing_points=500, device="cuda"):
    expression, train, tfs = data["expression"], data["train"], data["tf_indices"]
    seed_optimization(seed)
    clock = time.perf_counter()
    fa = expression_features(expression, components=components) if features is None else features
    features = feature_control(fa, random_features=random_features, permuted_features=permuted_features)
    stages = {"factor_analysis": time.perf_counter() - clock}
    clock = time.perf_counter()
    training, projected, model, likelihood = fit_features(features, train, seed=seed, inducing_points=inducing_points, device=device)
    stages["representation_and_gp"] = time.perf_counter() - clock
    clock = time.perf_counter()
    edges = np.argwhere(train != -1)
    labels = train[tuple(edges.T)]
    outgoing, incoming = np.log1p((train == 1).sum(axis=1)), np.log1p((train == 1).sum(axis=0))
    degree = classifier(degree_features(edges, outgoing, incoming), labels)
    learned = classifier(pair_features(projected, edges), labels)
    stages["simple_classifiers"] = time.perf_counter() - clock
    clock = time.perf_counter()
    n = len(train)
    arrays = {name: np.empty((len(tfs), n), dtype=np.float32) for name in ("beacon", "latent_variance", "degree_logistic", "learned_logistic")}
    for row, source in enumerate(tfs):
        for start in range(0, n, 2048):
            targets = np.arange(start, min(n, start + 2048))
            edges = np.column_stack([np.full(len(targets), source), targets])
            scores, variance = predict_pairs(projected, model, likelihood, edges, device=device)
            if not np.isfinite(scores).all() or not np.isfinite(variance).all():
                raise ValueError("Nonfinite external predictions")
            arrays["beacon"][row, targets] = scores
            arrays["latent_variance"][row, targets] = variance
            arrays["degree_logistic"][row, targets] = degree.predict_proba(degree_features(edges, outgoing, incoming))[:, 1]
            arrays["learned_logistic"][row, targets] = learned.predict_proba(pair_features(projected, edges))[:, 1]
    stages["candidate_scoring"] = time.perf_counter() - clock
    training.training_log["stage_seconds"] = stages
    return {name: {"values": values} for name, values in arrays.items()}, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("external")
