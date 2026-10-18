import itertools
import numpy as np
import pandas as pd
from scipy import stats
from evaluation import robustness as rb
from evaluation.metrics import pool_metrics


def sign_flip():
    assert rb.sign_flip_p(np.full(10, .2)) == 2 / 1024
    assert rb.sign_flip_p([3., 1., 2.]) == .25
    assert rb.sign_flip_p([1., -1.]) == 1.
    rng = np.random.default_rng(7)
    delta = rng.normal(.05, .1, 8)
    sums = [abs(np.dot(signs, delta)) for signs in itertools.product((1, -1), repeat=8)]
    assert rb.sign_flip_p(delta) == np.mean(np.array(sums) >= abs(delta.sum()) - 1e-12)
    assert rb.sign_flip_p(-delta) == rb.sign_flip_p(delta)


def t_interval():
    delta = np.array([.1, .3, .2, .4, .25])
    mean, low, high = rb.t_interval(delta)
    expected = stats.t.interval(.95, len(delta) - 1, loc=delta.mean(), scale=stats.sem(delta))
    assert np.isclose(mean, .25) and np.allclose((low, high), expected, rtol=0, atol=1e-12)
    _, low, high = rb.t_interval(np.array([1., 2., 3., 4.]))
    assert np.isclose(high - 2.5, stats.t.ppf(.975, 3) * np.std([1, 2, 3, 4], ddof=1) / 2)
    try:
        rb.t_interval([1.])
    except ValueError:
        pass
    else:
        raise AssertionError("One-split t interval accepted")


def verdicts():
    assert rb.split_verdict([.1, .2, .3]) == "holds"
    assert rb.split_verdict([-1., -2., -.5]) == "reverses"
    assert rb.split_verdict([1., -1., 2.]) == "ties"
    assert rb.split_verdict([0., 1., 1.]) == "ties"
    assert rb.interval_verdict(.01, .2) == "holds" and rb.interval_verdict(-.3, -.01) == "reverses"
    assert rb.interval_verdict(-.01, .02) == "ties"


def paired_rule():
    rng = np.random.default_rng(1)
    rows = []
    for part, splits in (("ten_splits", range(1, 11)), ("reverse_blocked", (42, 14, 100))):
        for split in splits:
            base = rng.uniform(.4, .6, len(rb.METRICS))
            offsets = {"BEACON": 0., "GNNLink": -.1, "Degree baseline": -.2 * (1 if split % 2 else -1) if part == "ten_splits" else -.01,
                       "Target popularity": -.3}
            for method, offset in offsets.items():
                rows.append(dict(part=part, dataset=1501, coverage=.8, split_seed=split, method=method,
                                 **{m: b + offset for m, b in zip(rb.METRICS, base)}))
    result = rb.paired(pd.DataFrame(rows)).set_index(["part", "method", "metric"])
    ten = result.loc["ten_splits"]
    assert (ten.loc["GNNLink"].verdict == "holds").all() and (ten.loc["Target popularity"].verdict == "holds").all()
    assert ten.loc[("GNNLink", "pooled_ap"), "sign_flip_p"] == 2 / 1024
    assert np.isnan(ten.loc[("GNNLink", "pooled_auprc"), "sign_flip_p"]) and np.isnan(ten.loc[("Target popularity", "pooled_ap"), "sign_flip_p"])
    assert (ten.loc["Degree baseline"].verdict == "ties").all() and ten.loc[("Degree baseline", "pooled_ap"), "beacon_higher"] == 5
    three = result.loc["reverse_blocked"]
    assert (three.loc["GNNLink"].verdict == "holds").all() and (three.loc["Degree baseline"].verdict == "holds").all()
    assert three.t_low.isna().all() and (three.splits == 3).all()


def setting():
    rng = np.random.default_rng(3)
    edges = np.array([(r, t) for r in range(4) for t in range(12) if r != t])
    labels = (rng.random(len(edges)) < .3).astype(np.int8)
    labels[[0, 12, 24, 36]] = 1
    known = np.array([[1, 0], [5, 6], [2, 5], [0, 7], [3, 7]])
    reverse = rb.reverse_known(known, edges)
    assert reverse.tolist() == [tuple(e) in {(0, 1), (6, 5), (5, 2), (7, 0), (7, 3)} for e in edges.tolist()]
    pop = rb.baselines(known, edges)
    assert np.array_equal(pop["Target popularity"], np.array([(known[:, 1] == t).sum() for t in edges[:, 1]], float))
    assert np.array_equal(pop["Regulator popularity"], np.array([(known[:, 0] == r).sum() for r in edges[:, 0]], float))
    beacon = [rng.random(len(edges)).astype(np.float32) for _ in range(3)]
    gnnlink = [rng.normal(size=len(edges)) for _ in range(3)]
    degree = rng.random(len(edges))
    rows = rb.evaluate_setting(dict(part="references", dataset=1, split_seed=42, coverage=.8), edges, labels, known,
                               {"BEACON": beacon, "GNNLink": gnnlink}, degree)
    frame = pd.DataFrame(rows).set_index("method")
    assert list(frame.index) == ["BEACON", "GNNLink", "Degree baseline", "Target popularity", "Regulator popularity"]
    mean = np.stack([b.astype(np.float64) for b in beacon]).mean(axis=0)
    assert frame.loc["BEACON", "pooled_ap"] == pool_metrics(labels.astype(int), mean, edges)["average_precision"]
    assert frame.loc["Degree baseline", "pooled_auprc"] == pool_metrics(labels.astype(int), degree, edges)["auprc_trapezoid"]
    assert (frame.reverse_excluded_pairs == reverse.sum()).all() and (frame.scoring.iloc[2:] == "deterministic").all()
    assert frame.loc["GNNLink", "scoring"] == "seed mean"


def alignment():
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    labels = np.array([1, 0, 0])
    values = rb.aligned(edges[::-1], np.array([3., 2., 1.]), edges, labels[::-1], labels)
    assert values.tolist() == [1., 2., 3.]
    try:
        rb.aligned(edges, np.zeros(3), edges, np.array([0, 0, 0]), labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched labels accepted")


def main():
    sign_flip()
    t_interval()
    verdicts()
    paired_rule()
    setting()
    alignment()
    print("Robustness core checks passed")


if __name__ == "__main__":
    main()
