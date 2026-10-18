import numpy as np
import pandas as pd
from evaluation import factor_components as fc
from evaluation.metrics import paired


def intervals():
    assert fc.interval([.3, -.1, .2], "split_range")[:2] == (-.1, .3)
    delta = np.random.default_rng(3).normal(.01, .05, 40)
    low, high, kind = fc.interval(delta, "regulator_seed42")
    reference = paired(delta, indices=np.random.default_rng(42).integers(40, size=(10000, 40)))
    assert (low, high) == (reference["ci_low"], reference["ci_high"]) and kind.endswith("regulator_seed42")
    rng = np.random.default_rng(42)
    for _ in range(2):
        rng.integers(40, size=(10000, 40))
    reference = paired(delta, indices=rng.integers(40, size=(10000, 40)))
    assert fc.interval(delta, "tcell_primary")[:2] == (reference["ci_low"], reference["ci_high"])
    rng = np.random.default_rng(42)
    rng.integers(72, size=(10000, 72))
    reference = paired(delta, indices=rng.integers(40, size=(10000, 40)))
    assert fc.interval(delta, "binding_group_2:72")[:2] == (reference["ci_low"], reference["ci_high"])
    saved = dict(tf_genes=np.array([f"g{i:02d}" for i in range(40)]), indices=np.zeros((10000, 40), dtype=int))
    assert fc.interval(delta, "rpe1_saved", units=saved["tf_genes"], saved=saved)[:2] == (delta[0], delta[0])
    clusters = np.repeat(list("abcdefg"), 3)[:20]
    values = np.arange(20, dtype=float)
    low, high, _ = fc.interval(values, "cluster_auroc", clusters)
    blocks = pd.DataFrame(dict(v=values, c=clusters)).groupby("c").v.agg(["sum", "count"])
    draws = np.random.default_rng(20_260_905).integers(7, size=(10000, 7))
    boot = blocks["sum"].to_numpy()[draws].sum(axis=1) / blocks["count"].to_numpy()[draws].sum(axis=1)
    assert (low, high) == tuple(np.quantile(boot, [.025, .975]))
    for method, count in (("split_range", 2), ("cluster_auroc", 20)):
        try:
            fc.interval(np.ones(count), method, np.repeat(list("ab"), 10) if count == 20 else None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid {method} cohort accepted")


def contrasts():
    rows = []
    for k in fc.COMPONENTS:
        for unit, value in zip(("42", "14", "100"), (.50, .52, .51)):
            rows.append(dict(k=k, endpoint="completion/1501/80/auprc_trapezoid", label="hESC, 80%: pooled AUPRC", method="BEACON", unit=unit,
                             value=value + (.1 if k == 8 else 0.), interval_method="split_range", cluster="", direction="higher", compare=True))
            for name, offset in (("GNNLink", (-.02, .02, 0.)), ("Degree", (0., 0., 0.)), ("RegGAIN", (-.3, -.3, -.3))):
                rows.append(dict(rows[-1] if name == "GNNLink" else rows[-2], method=name,
                                 value=.51 + dict(zip(("42", "14", "100"), offset))[unit]))
            rows.append(dict(k=k, endpoint="calibration/1501", label="hESC, 80%: calibrated Brier", method="BEACON", unit=unit, value=.01,
                             interval_method="split_range", cluster="", direction="lower", compare=True))
            for name, value in (("Topology control", .02), ("Validation prevalence constant", .015)):
                rows.append(dict(rows[-1], method=name, value=value))
            rows.append(dict(k=k, endpoint="sergio/sparse/80/auroc", label="SERGIO sparse, 80%: auroc", method="BEACON", unit=unit, value=.5,
                             interval_method="split_range", cluster="", direction="higher", compare=False))
    frame = pd.DataFrame(rows)
    values, comparisons = fc.summarize(frame)
    assert "sergio/sparse/80/auroc" in set(values.endpoint) and "sergio/sparse/80/auroc" not in set(comparisons.endpoint)
    best = fc.strongest(comparisons)
    assert best["completion/1501/80/auprc_trapezoid"] == "Degree" and best["calibration/1501"] == "Validation prevalence constant"
    row = comparisons.query('endpoint == "completion/1501/80/auprc_trapezoid" and baseline == "Degree" and k == 64').iloc[0]
    assert np.isclose(row.difference, 0.) and row.interval_status == "includes zero" and np.allclose((row.interval_low, row.interval_high), (-.01, .01))
    row = comparisons.query('endpoint == "completion/1501/80/auprc_trapezoid" and baseline == "Degree" and k == 8').iloc[0]
    assert row.interval_status == "above zero" and not row.interval_status_matches_k64 and not row.interval_status_matches_at_every_k
    assert np.isclose(row.difference_range_across_k, .1)
    assert comparisons.query('endpoint == "calibration/1501" and baseline == "Topology control"').interval_status.eq("below zero").all()
    assert values.query('endpoint == "completion/1501/80/auprc_trapezoid"').set_index("k").value.round(6).to_dict()[8] == .61
    assert np.isclose(values.range_across_k.max(), .1)


def main():
    intervals()
    contrasts()
    print("Factor-component checks passed")


if __name__ == "__main__":
    main()
