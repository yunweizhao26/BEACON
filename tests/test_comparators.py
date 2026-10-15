"""Internal-split identity and AP selection checks; no upstream fitting."""
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from beacon.model import Training
from comparators.adapters import internal_split, Monitor, RULES
from comparators.train import foundation


def main():
    train = np.full((40, 40), -1, dtype=np.int8)
    train.flat[:100] = 1
    train.flat[100:900] = 0
    fit, edges, labels = internal_split(train)
    reference = Training().internal_split(train)
    # Compare the exact BEACON held-out pairs, not just their class counts.
    assert np.array_equal(fit, reference["train"])
    assert np.array_equal(edges, reference["edges"])
    assert np.array_equal(labels, reference["labels"])
    assert (labels == 1).sum() == 10 and (labels == 0).sum() == 80
    assert (fit[tuple(edges.T)] == -1).all()
    np.testing.assert_array_equal(train[tuple(edges.T)], labels)
    monitor = Monitor(dict(every=2, patience=4), np.array([[0, 0], [0, 1]]), np.array([0, 1]))
    assert not monitor.check(1, lambda e: np.array([1., 0.]))
    assert not monitor.check(2, lambda e: np.array([0., 1.]))
    assert not monitor.check(4, lambda e: np.array([0., 1.]))
    assert monitor.check(6, lambda e: np.array([0., 1.]))
    assert monitor.best() == 2  # Earliest epoch wins an AP tie.
    assert RULES["reggain"]["grid"] == (50, 100, 200, 500, 1000, 2000)
    # The supplied mHSC-E archive has object metadata unrelated to training.
    # Read the numeric embeddings without enabling pickle for that metadata.
    genes = np.array(["A", "B"])
    embeddings = np.arange(6, dtype=np.float32).reshape(2, 3)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "embeddings.npz"
        np.savez(path, genes=genes, embeddings=embeddings, ensembl_ids=np.array([None, "ID"], dtype=object))
        actual = foundation(None, SimpleNamespace(embeddings=path), genes)
        np.testing.assert_array_equal(actual, embeddings)
        assert actual.dtype == embeddings.dtype
    print("Comparator split, selection and embedding-reader checks passed")


if __name__ == "__main__":
    main()
