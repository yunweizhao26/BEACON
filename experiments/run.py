"""Run one manifest setting into a new output directory."""
from pathlib import Path
import argparse
import json
import numpy as np
from beacon.data import Bundle, RELEASE, write_result

def execute(bundle, experiment, *, device="cuda", components=64, pca=False, scgpt=False, inducing_points=500):
    from experiments import fixed_pools, sampled_pairs, external, rpe1, trrust
    from experiments.expression import transform
    from experiments.sensitivity import scgpt_features, inducing
    suite, condition, reference = experiment["suite"], experiment["condition"], experiment["reference"]
    common = dict(components=components, pca=pca, inducing_points=inducing_points, device=device)
    if suite == "inducing":
        return {}, {"metric_only": True, "fits": inducing(bundle, experiment, device=device)}
    if suite == "trrust":
        if scgpt:
            common["features"] = scgpt_features(bundle, bundle.manifest["trrust"]["genes"], universe="trrust_" + bundle.manifest["trrust"]["tasks"][condition["task"]]["context"], pca=pca)
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
        if scgpt:
            common["features"] = scgpt_features(bundle, data["genes"], symbols=data["symbols"], universe={"k562":"k562_external", "rpe1":"rpe1_external"}.get(context, context), pca=pca)
        if suite == "rpe1":
            return rpe1.fit(bundle, data, **common)
        return external.fit(data, seed=condition["seed"], random_features=condition["control"] == "graph_only",
                            permuted_features=condition["control"] == "permuted", **common)
    expression, genes, truth, tfs = bundle.dataset(condition["dataset"])
    if scgpt:
        common["features"] = scgpt_features(bundle, genes, universe="beeline_DS" + str(condition["dataset"]), pca=pca)
    if suite == "sampled_pairs":
        dataset = next(d for d in bundle.manifest["datasets"] if d["dataset"] == condition["dataset"])
        split = {k: bundle.array(v) for k, v in dataset["sampled"].items()}
        if split["train"].shape != truth.shape or np.any(truth[split["test"] == 1] != 1):
            raise ValueError("Sampled split and reference network disagree")
        return sampled_pairs.fit(expression, split, seed=condition["seed"], **common)
    if suite == "all_settings":
        split = bundle.arrays(experiment["split"])
        if not np.array_equal(genes, split["genes"]):
            raise ValueError("All-setting split axes differ")
        return sampled_pairs.fit(expression, split, seed=condition["seed"], all_settings=True, **common)
    split = fixed_pools.make_split(truth, tfs, condition["split_seed"], condition["coverage"], condition["ratio"], condition["corruption"])
    frozen = bundle.arrays(reference["split"])
    for key in ("train", "valid", "test", "eligible_edges"):
        if not np.array_equal(split[key], frozen[key]):
            raise ValueError(f"Frozen split differs: {key}")
    control = condition["control"]
    if suite == "expression":
        expression = transform(expression, control)
        control = "graph_only" if control == "random" else "beacon"
    arrays, log = fixed_pools.fit(expression, split, seed=condition["seed"], ratio=condition["ratio"],
        snn_weight=condition["snn_weight"], random_features=control == "graph_only", permuted_features=control == "permuted",
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
    p.add_argument("--pca", action="store_true")
    p.add_argument("--scgpt", action="store_true")
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
    arrays, log = execute(bundle, experiment, device=args.device, components=args.components,
                          pca=args.pca, scgpt=args.scgpt, inducing_points=args.inducing_points)
    output = args.out or RELEASE / "results" / experiment["suite"]
    write_result(output, arrays, log, experiment["condition"])
    print(json.dumps({"output": str(output), "condition": experiment["condition"]}))

if __name__ == "__main__":
    main()
