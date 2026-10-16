"""Refits of the sparse GP with different numbers of inducing inputs."""
import numpy as np
import torch
from beacon.model import Training
from evaluation.metrics import metrics
from experiments.fixed_pools import predict_pairs

def inducing(bundle, experiment, *, device="cuda"):
    c = experiment["condition"]
    source = next(e for e in bundle.manifest["experiments"] if e["suite"] == "fixed_pools" and
        e["condition"]["dataset"] == c["dataset"] and e["condition"]["split_seed"] == c["split_seed"] and
        e["condition"]["coverage"] == .8 and e["condition"]["control"] == "beacon" and not e["condition"]["snn_weight"])
    reference = source["reference"]
    split = bundle.arrays(reference["split"])
    embeddings = bundle.array(reference["embeddings"])
    e_star = int(bundle.json(reference["training_log"])["encoder"]["e_star"])
    training = Training()
    cached = training.internal_split(split["train"])
    cached["e_star"] = e_star
    cached["selection"] = {"best_epoch": e_star}
    fits = {}
    for count in (128, 256, 500, 1000):
        model, likelihood, _, _ = training.fit_gp(embeddings, split["train"], torch.device(device),
            inducing_points_num=count, batch_size=1024, run_seed=42)
        model.eval()
        likelihood.eval()
        fit = {"gp_epochs": int(training.training_log["fits"][-1]["stop_epoch"]), "gp_stage": training.training_log["fits"][-1]["stage"]}
        for pool, key in (("validation", "valid"), ("test", "test")):
            edges = np.argwhere(split[key] != -1)
            scores, _ = predict_pairs(embeddings, model, likelihood, edges, device=device, variance=False)
            fit[pool] = metrics(split[key][tuple(edges.T)], scores, edges[:, 0])
        fits[str(count)] = fit
    return fits

if __name__ == "__main__":
    from experiments.run import main
    main()
