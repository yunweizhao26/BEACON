"""BEACON fit on one temporal-holdout prior, with the seed and settings of the perturbation-context fits."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from beacon.data import Bundle
from preparation.temporal import dated_edges, references, setting


def fit(bundle, context, cutoff, device="cpu"):
    data, train, _, _ = setting(bundle, context, cutoff, dated_edges(*references(bundle)))
    data = dict(data, **train)
    if context == "rpe1":
        from experiments.rpe1 import fit as fit_context
        return fit_context(bundle, data, device=device)
    from experiments.external import fit as fit_context
    return fit_context(data, seed=42, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--context", choices=("k562", "tcell_resting", "tcell_stimulated", "rpe1"), required=True)
    parser.add_argument("--cutoff", type=int, choices=(2000, 2005, 2010), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    arrays, log = fit(Bundle(args.config), args.context, args.cutoff, args.device)
    for name, value in arrays.items():
        np.savez_compressed(args.out / f"{name}.npz", **value)
    (args.out / "run.json").write_text(json.dumps(dict(context=args.context, cutoff=args.cutoff, device=args.device,
                                                       seconds=time.time() - start, training_log=log), indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
