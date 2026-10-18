from pathlib import Path
import argparse
import pandas as pd
from beacon.data import Bundle
from evaluation.epochs import SETTING, history, key, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    bundle = Bundle(parser.parse_args().config)
    trace = {}
    for row in history(bundle).to_dict("records"):
        setting = {k: None if not isinstance(row[k], str) and pd.isna(row[k]) else row[k] for k in SETTING}
        trace.setdefault(key(setting), []).append((row["validation_ap"], -int(row["epoch"])))
    checked = 0
    for fit in records(bundle):
        if fit.get("selected_epochs") is None:
            continue
        epochs, rule = int(fit["selected_epochs"]), fit["rule"]
        if "grid" in rule:
            assert epochs in rule["grid"], fit
        else:
            assert epochs % rule["every"] == 0 and 1 <= epochs <= rule["max"], fit
        assert -max(trace[key(fit)])[1] == epochs, fit
        if fit["suite"] == "fixed_pools":
            assert not fit.get("fallback", False) and fit["validation_positives"] >= 10, fit
        checked += 1
    print(f"{len(records(bundle))} run records; {checked} selected epoch counts follow their rules")


if __name__ == "__main__":
    main()
