from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from beacon.data import Bundle, RELEASE
from evaluation.seeds import SEEDS, TRRUST_SEEDS, combine, experiments


def commands(bundle, args, config):
    python = [sys.executable, "-B", "-m"]
    extra = ["--config", str(config)] if config else []
    if args.suite == "temporal":
        return {seed: python + ["experiments.temporal", "--context", args.context, "--cutoff", str(args.cutoff), "--seed", str(seed),
                                "--device", args.device, "--out", str(args.out / f"seed_{seed}")] + extra for seed in SEEDS}
    if args.suite == "trrust":
        fits = experiments(bundle, "trrust", TRRUST_SEEDS, control="beacon")
    else:
        condition = {"fixed_pools": dict(dataset=args.dataset, coverage=args.coverage, split_seed=args.split_seed, control=args.control),
                     "sergio": dict(dataset=args.dataset, coverage=args.coverage, split_seed=args.split_seed, control="beacon"),
                     "sampled_pairs": dict(dataset=args.dataset), "external": dict(context=args.context, control=args.control),
                     "rpe1": {}}[args.suite]
        fits = experiments(bundle, args.suite, SEEDS, **condition)
    return {fit["condition"]["seed"]: python + ["experiments.run", "--case", str(fit["setting"]), "--device", args.device,
                                                 "--out", str(args.out / f"seed_{fit['condition']['seed']}")] + extra for fit in fits}


def average_outputs(directories, destination):
    destination.mkdir()
    for path in sorted(directories[0].glob("*.npz")):
        members = []
        for directory in directories:
            with np.load(directory / path.name, allow_pickle=False) as data:
                members.append({key: data[key] for key in data.files})
        np.savez_compressed(destination / path.name, **combine(members))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=("fixed_pools", "sampled_pairs", "sergio", "external", "rpe1", "temporal", "trrust"))
    p.add_argument("--config", type=Path)
    p.add_argument("--dataset", type=int)
    p.add_argument("--coverage", type=float, default=.8)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--control", default="beacon")
    p.add_argument("--context")
    p.add_argument("--cutoff", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config = args.config.resolve() if args.config else None
    bundle = Bundle(config)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pending = list(commands(bundle, args, config).items())
    running, seconds = {}, {}
    while pending or running:
        while pending and len(running) < max(1, args.parallel):
            seed, command = pending.pop(0)
            running[seed] = (subprocess.Popen(command, cwd=RELEASE, env=os.environ.copy()), time.perf_counter())
        time.sleep(.2)
        for seed, (process, start) in list(running.items()):
            if process.poll() is not None:
                if process.returncode:
                    raise SystemExit(f"Fit with seed {seed} failed with exit code {process.returncode}")
                seconds[seed] = time.perf_counter() - start
                del running[seed]
    fitted = time.perf_counter()
    order = TRRUST_SEEDS if args.suite == "trrust" else SEEDS
    average_outputs([args.out / f"seed_{seed}" for seed in order], args.out / "ensemble")
    finished = time.perf_counter()
    records = {}
    for seed in order:
        path = args.out / f"seed_{seed}" / ("run.json" if args.suite == "temporal" else "manifest.json")
        record = json.loads(path.read_text())
        records[str(seed)] = {k: record.get(k) for k in ("seconds", "peak_rss_mb", "peak_cuda_allocated_mb", "device", "threads")}
        log = record.get("training_log", {})
        encoder = log.get("encoder", {})
        records[str(seed)].update(stage_seconds=log.get("stage_seconds"), encoder_seconds={k: encoder.get(k) for k in (
            "step1_encoder_seconds", "step1_gp_seconds", "step2_encoder_seconds")}, final_gp_seconds=(log.get("fits") or [{}])[-1].get("seconds"))
    (args.out / "run.json").write_text(json.dumps(dict(suite=args.suite, seeds=list(order), parallel=args.parallel, device=args.device,
        threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        process_seconds={str(k): v for k, v in seconds.items()}, fits=records, fits_wall_seconds=fitted - started,
        averaging_seconds=finished - fitted, total_seconds=finished - started), indent=2) + "\n")


if __name__ == "__main__":
    main()
