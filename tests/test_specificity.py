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
    external_consensus()
    print("Specificity regression checks passed")


def external_consensus():
    from scipy.stats import rankdata, spearmanr
    from sklearn.metrics import average_precision_score
    from evaluation.metrics import expected_random_ap
    from evaluation.specificity import external_statistics
    rng = np.random.default_rng(3)
    matrix = rng.random((4, 30))
    selected = np.array([0, 2, 3])
    mask = np.ones((3, 30), bool)
    mask[:, :2] = False
    support = np.zeros((3, 30), bool)
    support[0, [3, 7, 11]] = support[1, 5] = True
    agreement, residual = external_statistics(matrix, selected, mask, support)
    ranks = np.vstack([rankdata(matrix[i]) for i in selected])
    for k in range(3):
        others = ranks[[j for j in range(3) if j != k]].sum(axis=0)
        assert np.isclose(agreement[k], spearmanr(ranks[k][mask[k]], others[mask[k]])[0])
        expected = average_precision_score(support[k, mask[k]], (ranks[k] - others / 2)[mask[k]]) if support[k].any() else np.nan
        assert np.isclose(residual[k], expected, equal_nan=True)
    agreement, _ = external_statistics(np.vstack([matrix[0], matrix[0]]), np.array([0, 1]), mask[:2], support[:2])
    assert np.allclose(agreement, 1)
    try:
        external_statistics(matrix, np.array([0]), mask[:1], support[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("Single-regulator consensus accepted")
    selected, mask = np.arange(10), np.ones((10, 300), bool)
    support = np.zeros((10, 300), bool)
    for k in range(10):
        support[k, rng.choice(300, 1 + k % 3, replace=False)] = True
    draws = np.array([external_statistics(rng.random((10, 300)), selected, mask, support)[1].mean() for _ in range(300)])
    chance = expected_random_ap(mask.sum(axis=1), support.sum(axis=1)).mean()
    assert abs(draws.mean() - chance) < 4 * draws.std() / np.sqrt(len(draws)), (draws.mean(), chance)


if __name__ == "__main__":
    main()
