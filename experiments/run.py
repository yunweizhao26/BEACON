from pathlib import Path
import argparse
import json
import numpy as np
from beacon.data import Bundle, RELEASE, write_result

def execute(bundle, experiment, *, device="cuda", components=64, inducing_points=500):
    from experiments import fixed_pools, sampled_pairs, external, rpe1, trrust
    from experiments.expression import transform
    from experiments.sensitivity import inducing
    suite, condition, reference = experiment["suite"], experiment["condition"], experiment["reference"]
    common = dict(components=components, inducing_points=inducing_points, device=device)
    if suite == "inducing":
        return {}, {"metric_only": True, "fits": inducing(bundle, experiment, device=device)}
    if suite == "trrust":
        return trrust.fit(bundle, condition["task"], **common)
    if suite in ("external", "rpe1"):
        context = condition.get("context", "rpe1")
        prepared = bundle.manifest["prepared"][context]
        data = bundle.arrays(prepared["expression"])
        training = bundle.arrays(prepared["training"] if suite == "rpe1" else reference["training"])
        for key in ("genes", "symbols", "tf_indices"):
            if not np.array_equal(data[key], training[key]):
                raise ValueError(f"Prepared and training axes differ: {key}")
        data.update(training)
        if suite == "rpe1":
            return rpe1.fit(data, seed=condition["seed"], **common)
        return external.fit(data, seed=condition["seed"], random_features=condition["control"] == "graph_only",
                            permuted_features=condition["control"] == "permuted", **common)
    expression, genes, truth, tfs = bundle.dataset(condition["dataset"])
    if suite == "sampled_pairs":
        dataset = next(d for d in bundle.manifest["datasets"] if d["dataset"] == condition["dataset"])
        split = {k: bundle.array(v) for k, v in dataset["sampled"].items()}
        if split["train"].shape != truth.shape or np.any(truth[split["test"] == 1] != 1):
            raise ValueError("Sampled split and reference network disagree")
        return sampled_pairs.fit(expression, split, seed=condition["seed"], **common)
    split = fixed_pools.make_split(truth, tfs, condition["split_seed"], condition["coverage"])
    frozen = bundle.arrays(reference["split"])
    for key in ("train", "valid", "test", "eligible_edges"):
        if not np.array_equal(split[key], frozen[key]):
            raise ValueError(f"Frozen split differs: {key}")
    control = condition["control"]
    if suite == "expression":
        expression = transform(expression, control)
        control = "graph_only" if control == "random" else "beacon"
    arrays, log = fixed_pools.fit(expression, split, seed=condition["seed"],
        random_features=control == "graph_only", permuted_features=control == "permuted",
        without_encoder=control == "fa_gp", diagnostics=condition["diagnostics"], score_name=control, decoder_validation=suite != "expression", **common)
    if suite == "expression":
        decoder = arrays.pop("decoder_predictions")
        arrays["expression_predictions"] = dict(edges=decoder["edges"], labels=decoder["labels"],
            beacon_gp=arrays["predictions"][control], beacon_decoder=decoder["decoder"])
    return arrays, log

def main(suite=None, context=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--case", type=int, help="Setting identifier recorded in the bundle manifest")
    p.add_argument("--dataset", type=int, default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--inducing-points", type=int, default=500)
    args = p.parse_args()
    bundle = Bundle(args.config)
    cases = [e for e in bundle.manifest["experiments"] if
             (e["setting"] == args.case if args.case is not None else
              e["suite"] == (suite or "fixed_pools") and (args.dataset is None or e["condition"]["dataset"] == args.dataset) and (context is None or e["condition"].get("context") == context))]
    if not cases:
        raise ValueError("No matching release setting")
    experiment = cases[0]
    if suite and experiment["suite"] != suite:
        raise ValueError("Case does not belong to the requested experiment")
    import resource
    import time
    import torch
    from experiments import fixed_pools, external, rpe1, sampled_pairs, trrust
    started = time.perf_counter()
    arrays, log = execute(bundle, experiment, device=args.device, components=args.components,
                          inducing_points=args.inducing_points)
    output = args.out or RELEASE / "results" / experiment["suite"]
    cuda = args.device == "cuda" and torch.cuda.is_available()
    write_result(output, arrays, log, experiment["condition"], seconds=time.perf_counter() - started,
                 peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                 peak_cuda_allocated_mb=torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                 device=torch.cuda.get_device_name() if cuda else "cpu", threads=torch.get_num_threads())
    print(json.dumps({"output": str(output), "condition": experiment["condition"]}))

if __name__ == "__main__":
    main()
