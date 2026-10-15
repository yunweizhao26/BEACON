"""Small regression checks for the revised specificity estimator."""
import numpy as np
from evaluation.specificity import consensus_frame, evaluate_frame, specific_gain


def main():
    edges = np.array([(r, t) for r in range(3) for t in range(4)])
    labels = np.tile([False, True, False, True], 3)
    values = np.tile([1., 1., 2., 3.], 3) + np.repeat([0., 10., 100.], 4)
    reverse = np.tile([True, True, False, False], 3)
    frame = consensus_frame(edges, labels, values)
    assert np.array_equal(frame.consensus, consensus_frame(edges, ~labels, values).consensus)
    np.testing.assert_array_equal(frame.percentile[:4], [.375, .375, .75, 1.])
    result, detail = evaluate_frame(frame, reverse)
    assert result["gain"] == 1.
    assert result["excluded_positives"] == result["excluded_unlabeled"] == 3
    assert result["evaluated_pairs"] == 6 and len(detail) == 3
    assert specific_gain(edges, labels, values, reverse) == (result["within_ap"], result["target_only_ap"])

    edges = np.array([(r, t) for r in range(2) for t in range(3)])
    values = np.array([0., 1., 2., 3., 1., 2.])
    labels = np.array([False, True, True, True, False, True])
    frame = consensus_frame(edges, labels, values)
    np.testing.assert_allclose(frame.consensus, [1., 1/3, 2/3, 1/3, 2/3, 1.])
    # Exclusion must not change either regulator's original percentile pool.
    result, _ = evaluate_frame(frame, np.array([True, False, False, False, False, False]))
    assert result["evaluated_pairs"] == 5 and result["eligible_regulators"] == 1
    singleton = consensus_frame(np.array([[0, 1]]), np.array([True]), np.array([.5]))
    assert singleton.fallback.all() and singleton.consensus.iloc[0] == 1.
    try:
        consensus_frame(np.array([[0, 1], [0, 1]]), [False, True], [.1, .2])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate pairs accepted")
    print("Specificity regression checks passed")


if __name__ == "__main__":
    main()
