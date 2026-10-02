"""RPE1 uses the frozen FA features and the complete post-FA RNG state."""
import random
import numpy as np
import torch
from beacon.features import expression_features
from experiments.fixed_pools import fit_features, predict_pairs

def restore_factor_state(bundle):
    cache = bundle.manifest["prepared"]["factor_cache"]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    state, python = bundle.arrays(cache["rng"]), bundle.json(cache["python_rng"])
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    random.setstate(tuples(python["state"]))
    np.random.set_state((python["numpy_bit_generator"], state["numpy_keys"], int(state["numpy_position"]), int(state["numpy_has_gauss"]), float(state["numpy_cached_gaussian"])))
    torch.set_rng_state(torch.from_numpy(state["torch_cpu_state"]))
    return bundle.array(cache["features"])

def fit(bundle, data, *, components=64, pca=False, features=None, inducing_points=500, device="cuda"):
    cached = restore_factor_state(bundle)
    if features is None:
        features = cached if components == 64 and not pca else expression_features(
            np.asarray(data["expression"], dtype=np.float32), components=components, pca=pca)
    training, projected, model, likelihood = fit_features(features, data["train"], device=device,
        inducing_points=inducing_points)
    n, tfs = len(data["genes"]), data["tf_indices"]
    scores = np.empty((len(tfs), n), dtype=np.float32)
    for row, source in enumerate(tfs):
        for start in range(0, n, 4096):
            targets = np.arange(start, min(n, start + 4096))
            edges = np.column_stack([np.full(len(targets), source), targets])
            scores[row, targets], _ = predict_pairs(projected, model, likelihood, edges, device=device, batch_size=4096, variance=False)
    return {"beacon": {"values": scores}}, training.training_log

if __name__ == "__main__":
    from experiments.run import main
    main("rpe1")
