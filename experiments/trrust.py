"""TRRUST held-out edges; all evaluation pairs and reversals stay masked."""
import numpy as np
import torch
import gpytorch
from beacon.features import expression_features
from experiments.fixed_pools import seed_optimization, fit_features

def fit(bundle, task_index, *, components=64, pca=False, features=None, inducing_points=500, device="cuda"):
    manifest = bundle.manifest["trrust"]
    task = manifest["tasks"][task_index]
    seed_optimization(task["seed"], cuda_all=True)
    frame = bundle.frame(task["expression"], index_col=0)
    genes = frame.index.astype(str).tolist()
    if genes != manifest["genes"]:
        raise ValueError("TRRUST expression gene order differs from frozen manifest")
    expression = frame.to_numpy(dtype=np.float32)
    gene_to_index = {gene: i for i, gene in enumerate(genes)}
    prior = bundle.frame(manifest["prior"], dtype=str)
    edge_by_id = {edge_id: (gene_to_index[source], gene_to_index[target]) for edge_id, source, target in prior[["edge_id", "source", "target"]].itertuples(index=False, name=None)}
    positive = np.asarray([edge_by_id[e] for e in manifest["train_positive_edge_ids"]], dtype=np.int64)
    unlabeled = np.asarray(manifest["train_unlabeled"], dtype=np.int64)
    train = np.full((len(genes), len(genes)), -1, dtype=np.int8)
    train[tuple(positive.T)], train[tuple(unlabeled.T)] = 1, 0
    evaluation = manifest["evaluation"]
    edges = np.asarray([(int(r["source_index"]), int(r["target_index"])) for r in evaluation], dtype=np.int64)
    used = set(map(tuple, np.argwhere(train != -1)))
    if set(map(tuple, edges)) & used:
        raise ValueError("Evaluated TRRUST pair occurs in training")
    if set(map(tuple, edges[:, ::-1])) & set(map(tuple, positive)):
        raise ValueError("Evaluated reciprocal TRRUST edge occurs in prior")
    features = expression_features(expression, components=components, pca=pca) if features is None else features
    training, projected, model, likelihood = fit_features(features, train, seed=task["seed"],
        inducing_points=inducing_points, device=device)
    # The frozen task scores the complete evaluation list in one batch, concatenating on the device.
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(.1):
        sources = torch.tensor(projected[edges[:, 0]], dtype=torch.float32, device=device)
        targets = torch.tensor(projected[edges[:, 1]], dtype=torch.float32, device=device)
        scores = likelihood(model(torch.cat([sources, targets], dim=1))).mean.cpu().numpy()
    return {"predictions": {"edges": edges, "probability": np.array([float(p) for p in scores]),
        "source": np.array([r["source"] for r in evaluation]), "target": np.array([r["target"] for r in evaluation]),
        "analysis_role": np.array([r["analysis_role"] for r in evaluation])}}, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("trrust")
