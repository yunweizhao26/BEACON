import numpy as np

SEEDS = (42, 14, 100)
TRRUST_SEEDS = (42, 14, 100, 38, 47)
AXES = frozenset(("edges", "labels", "test_edges", "test_labels", "valid_edges", "valid_labels", "validation_edges",
                  "validation_labels", "eligible_edges", "genes", "symbols", "tf_indices", "source", "target",
                  "analysis_role"))


def average(values):
    first = np.asarray(values[0])
    if all(np.asarray(v).dtype == first.dtype and np.array_equal(first, v) for v in values[1:]):
        return first
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in values])
    if not np.isfinite(stack).all():
        raise ValueError("Nonfinite seed scores")
    return stack.mean(axis=0)


def combine(members):
    keys = list(members[0])
    if any(list(m) != keys for m in members[1:]):
        raise ValueError("Seed fits store different arrays")
    combined = {}
    for key in keys:
        values = [m[key] for m in members]
        if key in AXES or not np.issubdtype(np.asarray(values[0]).dtype, np.number):
            if not all(np.array_equal(values[0], v) for v in values[1:]):
                raise ValueError(f"Seed fits disagree on {key}")
            combined[key] = values[0]
        else:
            combined[key] = average(values)
    return combined


def matches(condition, required):
    return all(condition.get(key) == value for key, value in required.items())


def experiments(bundle, suite, seeds=SEEDS, **condition):
    found = [e for e in bundle.manifest["experiments"] if e["suite"] == suite and matches(e["condition"], condition)]
    by_seed = {e["condition"]["seed"]: e for e in found}
    if len(by_seed) != len(found) or set(by_seed) != set(seeds):
        raise ValueError(f"Expected seeds {seeds} for {suite} {condition}, found {sorted(by_seed)}")
    return [by_seed[seed] for seed in seeds]


def fit_arrays(bundle, fits, role):
    return combine([bundle.arrays(fit["reference"][role]) for fit in fits])


def comparators(bundle, suite, method, **keys):
    found = [c for c in bundle.manifest["comparators"] if c.get("suite") == suite and c.get("method") == method and matches(c, keys)]
    if len(found) == 1 and found[0].get("seed") is None:
        return found
    by_seed = {c.get("seed"): c for c in found}
    if len(by_seed) != len(found) or set(by_seed) != set(SEEDS):
        raise ValueError(f"Expected seeds {SEEDS} for comparator {method} {keys}, found {sorted(map(str, by_seed))}")
    return [by_seed[seed] for seed in SEEDS]


def comparator_arrays(bundle, suite, method, role="predictions", **keys):
    return combine([bundle.arrays(c["artifacts"][role]) for c in comparators(bundle, suite, method, **keys)])


def stored(bundle, value, seeds=SEEDS):
    if isinstance(value, int):
        return bundle.array(value)
    if set(value) != {str(seed) for seed in seeds}:
        raise ValueError(f"Expected seeds {seeds}, found {sorted(value)}")
    return average([bundle.array(value[str(seed)]) for seed in seeds])
