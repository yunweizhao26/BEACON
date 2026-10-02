"""Fixed pools, coverage, corruption, ratio, and component controls."""
import random
import numpy as np
import torch
import gpytorch
from beacon.features import expression_features, feature_control
from beacon.model import Training
from beacon.pairs import classifier, pair_features, decoder_scores, make_split
from evaluation.metrics import degree_features
from evaluation.probabilities import fit_nnpu, calibrate

def seed_optimization(seed, *, cuda_all=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_all:
        torch.cuda.manual_seed_all(seed)

def fit_features(features, train, *, seed=42, ratio=5, snn_weight=0.0,
                 without_encoder=False, inducing_points=500, device="cuda"):
    if inducing_points < 1 or int(inducing_points) != inducing_points:
        raise ValueError("inducing_points must be a positive integer")
    device = torch.device(device)
    training = Training(snn_weight=snn_weight)
    if without_encoder:
        projected = features
    else:
        encoder = training.fit_encoder(features, train, features.shape[1], 16,
                                       32, .001, ratio, 1., device)
        encoder.eval()
        with torch.no_grad():
            projected = encoder.get_embeddings(torch.tensor(features, dtype=torch.float32, device=device), combine_mode="avg").cpu().numpy()
    model, likelihood, _, _ = training.fit_gp(projected, train, device,
        inducing_points_num=inducing_points, batch_size=1024, run_seed=seed)
    model.eval()
    likelihood.eval()
    return training, projected, model, likelihood

def predict_pairs(projected, model, likelihood, edges, *, device="cuda", batch_size=2048, variance=True):
    scores, variances = [], []
    for offset in range(0, len(edges), batch_size):
        inputs = torch.tensor(pair_features(projected, edges[offset:offset + batch_size]), dtype=torch.float32, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(.1):
            latent = model(inputs)
            scores.append(likelihood(latent).mean.cpu().numpy())
            if variance:
                variances.append(latent.variance.cpu().numpy())
    return np.concatenate(scores), np.concatenate(variances) if variance else None

def fit(expression, split, *, seed=42, ratio=5, snn_weight=0.0,
        random_features=False, permuted_features=False, without_encoder=False,
        components=64, pca=False, inducing_points=500, diagnostics=False,
        decoder=True, logistic=True, nnpu=None,
        device="cuda", features=None, score_name="beacon", decoder_validation=True):
    seed_optimization(seed)
    fa = expression_features(expression, components=components, pca=pca) if features is None else features
    features = feature_control(fa, random_features=random_features, permuted_features=permuted_features)
    training, projected, model, likelihood = fit_features(features, split["train"], seed=seed, ratio=ratio,
        snn_weight=snn_weight, without_encoder=without_encoder, inducing_points=inducing_points, device=device)
    edges = {pool: np.argwhere(split[pool] != -1) for pool in ("train", "test", "valid")}
    labels = {pool: split[pool][tuple(edges[pool].T)] for pool in edges}
    test_gp, latent_variance = predict_pairs(projected, model, likelihood, edges["test"], device=device)
    valid_gp, valid_variance = predict_pairs(projected, model, likelihood, edges["valid"], device=device)
    outgoing = np.log1p((split["train"] == 1).sum(axis=1))
    incoming = np.log1p((split["train"] == 1).sum(axis=0))
    predictions = {score_name: test_gp, "degree_product": outgoing[edges["test"][:, 0]] + incoming[edges["test"][:, 1]]}
    validation = {score_name: valid_gp}
    for name, train_x, valid_x, test_x in ([
        ("degree_logistic", degree_features(edges["train"], outgoing, incoming), degree_features(edges["valid"], outgoing, incoming), degree_features(edges["test"], outgoing, incoming)),
        ("fa_logistic", pair_features(fa, edges["train"]), pair_features(fa, edges["valid"]), pair_features(fa, edges["test"])),
        ("learned_logistic", pair_features(projected, edges["train"]), pair_features(projected, edges["valid"]), pair_features(projected, edges["test"]))] if logistic else []):
        head = classifier(train_x, labels["train"])
        predictions[name] = head.predict_proba(test_x)[:, 1]
        validation[name] = head.predict_proba(valid_x)[:, 1]
    result = {}
    if nnpu is None:
        nnpu = diagnostics
    if nnpu:
        training_pairs = len(split["eligible_edges"]) - len(edges["test"]) - len(edges["valid"])
        (validation["learned_nnpu"], predictions["learned_nnpu"]), pu_info = fit_nnpu(
            pair_features(projected, edges["train"]), labels["train"], pair_features(projected, edges["valid"]),
            pair_features(projected, edges["test"]), labels["valid"].mean(), float(labels["train"].sum() / training_pairs), device)
    if diagnostics:
        calibrated, diagnostic_report = calibrate(labels["valid"], validation, labels["test"], predictions, latent_variance)
        training.training_log["diagnostics"] = dict(diagnostic_report, **({"nnpu_setup": pu_info} if nnpu else {}))
        result["calibrated_predictions"] = calibrated
    result["predictions"] = dict(edges=edges["test"], labels=labels["test"], latent_variance=latent_variance, **predictions)
    result["validation_predictions"] = dict(edges=edges["valid"], labels=labels["valid"], latent_variance=valid_variance, **validation)
    result["embeddings"] = {"values": projected}
    # The original wrapper computes validation decoder scores before test scores.
    if decoder and not without_encoder:
        pools = [("valid", "decoder_validation_predictions"), ("test", "decoder_predictions")] if decoder_validation else [("test", "decoder_predictions")]
        for pool, output in pools:
            result[output] = dict(edges=edges[pool], labels=labels[pool], decoder=decoder_scores(training.encoder, features, edges[pool]))
    return result, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("fixed_pools")
