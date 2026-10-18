import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
from evaluation.seeds import TRRUST_SEEDS

SETTING = ("suite", "method", "dataset", "coverage", "split_seed", "context", "task", "seed")
SUITES = ("fixed_pools", "external", "rpe1", "trrust")
METHODS = ("gnnlink", "genelink", "gclink", "reggain", "scregnet")
SECOND_READING = ("reggain_alternative_orientation",)
CONTEXT = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
COLUMNS = ("suite", "method", "context", "dataset", "coverage", "split_seed", "task", "seed", "device", "rule_every",
           "rule_max", "rule_patience", "rule_grid", "rule_released", "selected_epochs", "validation_pairs",
           "validation_positives", "fallback", "seconds", "selection_seconds", "refit_seconds")


def key(setting):
    return tuple(None if setting.get(k) is None else float(setting[k]) if k == "coverage" else
                 int(setting[k]) if k in ("dataset", "split_seed", "task", "seed") else str(setting[k]) for k in SETTING)


def entries(bundle):
    found = []
    for c in bundle.manifest["comparators"]:
        if c.get("suite") in SUITES and c.get("seed") is not None:
            (record,) = c["artifacts"].values()
            found.append(({k: c[k] for k in SETTING if k in c}, record))
    for method, records in bundle.manifest.get("trrust", {}).get("comparators", {}).items():
        if len(records) != len(TRRUST_SEEDS):
            raise ValueError(f"TRRUST {method}: expected one record per task seed")
        found += [(dict(suite="trrust", method=method, task=task, seed=seed), record)
                  for task, (seed, record) in enumerate(zip(TRRUST_SEEDS, records))]
    return found


def records(bundle, complete=True):
    section = bundle.manifest.get("comparator_runs")
    if not section:
        raise ValueError("The bundle has no comparator run records")
    expected = {}
    for setting, record in entries(bundle):
        if key(setting) in expected:
            raise ValueError(f"Duplicate comparator entry: {setting}")
        expected[key(setting)] = record
    seen = set()
    fits = bundle.json(section["fits"])
    for fit in fits:
        k = key(fit)
        if k not in expected:
            raise ValueError(f"Run record without a bundle comparator entry: {dict(zip(SETTING, k))}")
        if k in seen:
            raise ValueError(f"Two run records for one comparator entry: {dict(zip(SETTING, k))}")
        if bundle.assets[expected[k]]["original_sha256"] != fit["score_sha256"]:
            raise ValueError(f"Run record belongs to a different score file: {dict(zip(SETTING, k))}")
        seen.add(k)
    missing = [dict((n, v) for n, v in zip(SETTING, k) if v is not None) for k in expected if k not in seen]
    if complete and missing:
        raise ValueError(f"{len(missing)} comparator entries have no run record, for example {missing[:3]}")
    return fits


def label(fit):
    if fit["suite"] == "fixed_pools":
        return CONTEXT[int(fit["dataset"])]
    return {"rpe1": "rpe1", "trrust": "flu"}.get(fit["suite"], fit.get("context"))


def selected(bundle):
    rows = []
    for fit in records(bundle):
        if fit.get("selected_epochs") is None or fit["method"] in SECOND_READING:
            continue
        rule = fit.get("rule") or {}
        grid = rule.get("grid")
        rows.append({**{k: fit.get(k) for k in COLUMNS}, "context": label(fit), "rule_every": rule.get("every"),
                     "rule_max": rule.get("max"), "rule_patience": rule.get("patience"),
                     "rule_grid": " ".join(map(str, grid)) if grid else None, "rule_released": rule.get("released")})
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in ("dataset", "split_seed", "task", "seed", "rule_every", "rule_max", "rule_patience", "rule_released",
                   "selected_epochs", "validation_pairs", "validation_positives"):
        frame[column] = frame[column].astype("Int64")
    order = {"suite": SUITES, "method": METHODS, "seed": TRRUST_SEEDS}
    frame = frame.sort_values(["suite", "method", "dataset", "context", "coverage", "split_seed", "task", "seed"],
                              key=lambda s: s.map(lambda v: order[s.name].index(v)) if s.name in order else s, kind="stable")
    return frame.reset_index(drop=True)


def ranges(bundle):
    frame = selected(bundle)
    rows = []
    for (suite, method), part in frame.groupby(["suite", "method"], sort=False):
        groups = [("all", part)] + [(str(seed), part[part.seed == seed]) for seed in dict.fromkeys(part.seed.tolist())]
        for seed, members in groups:
            values = members.selected_epochs.astype(int)
            rows.append(dict(method=method, suite=suite, seed=seed, min_epochs=int(values.min()), max_epochs=int(values.max()),
                             fits=len(values)))
    return pd.DataFrame(rows)


def history(bundle):
    records(bundle)
    return bundle.frame(bundle.manifest["comparator_runs"]["validation_history"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=RELEASE / "results" / "epochs")
    args = parser.parse_args()
    bundle = Bundle(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    fits = selected(bundle)
    table = ranges(bundle)
    fits.to_csv(args.out / "selected_epochs.csv", index=False)
    table.to_csv(args.out / "epoch_ranges.csv", index=False)
    print(f"{len(records(bundle))} run records; {len(fits)} fits with selected epochs")
    print(table[table.seed.isin(["all", "42"])].to_string(index=False))


if __name__ == "__main__":
    main()
