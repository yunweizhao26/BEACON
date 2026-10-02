"""Frozen metric definitions; PR area and AP are distinct."""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
def split_checks(train, valid, test):
    matrices = {"train": train, "valid": valid, "test": test}
    checks = {}
    for name, matrix in matrices.items():
        checks[f"{name}_invalid_labels"] = int((~np.isin(matrix, [-1, 0, 1])).sum())
        checks[f"{name}_self_pairs"] = int((np.diag(matrix) != -1).sum())
    for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        checks[f"{left}_{right}_overlap"] = int(((matrices[left] != -1) & (matrices[right] != -1)).sum())
    checks["heldout_positive_as_training_zero"] = int((((valid == 1) | (test == 1)) & (train == 0)).sum())
    return checks

def metrics(labels, scores, sources):
    result = {"pairs": len(labels), "positives": int(labels.sum()), "prevalence": float(labels.mean())}
    result["auroc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    result["average_precision"] = float(average_precision_score(labels, scores)) if labels.sum() else None
    precision, recall, _ = precision_recall_curve(labels, scores)
    result["auprc_trapezoid"] = float(auc(recall, precision))
    regulator_ap, regulator_auc = [], []
    for source in np.unique(sources):
        mask = sources == source
        if len(np.unique(labels[mask])) == 2:
            regulator_ap.append(average_precision_score(labels[mask], scores[mask]))
            regulator_auc.append(roc_auc_score(labels[mask], scores[mask]))
    result["macro_eligible_sources"] = len(regulator_ap)
    result["macro_average_precision"] = float(np.mean(regulator_ap)) if regulator_ap else None
    result["macro_auroc"] = float(np.mean(regulator_auc)) if regulator_auc else None
    # Fractional allocation at the cutoff avoids arbitrary gene-order effects for tied degree scores.
    for k in [10, 50, 100]:
        effective_k = min(k, len(labels))
        threshold = np.partition(scores, len(scores) - effective_k)[-effective_k]
        above, tied = scores > threshold, scores == threshold
        recovered = labels[above].sum() + (effective_k - above.sum()) * labels[tied].mean()
        result[f"top{k}_supported_fraction"] = float(recovered / effective_k)
    return result

def degree_features(edges, outgoing, incoming):
    source, target = edges.T
    return np.column_stack([outgoing[source], incoming[source], outgoing[target], incoming[target],
                            outgoing[source] * incoming[target]])

def fractional_topk(labels, scores, k):
    k = min(k, len(labels))
    threshold = np.partition(scores, len(scores) - k)[-k]
    above, tied = scores > threshold, scores == threshold
    return float((labels[above].sum() + (k - above.sum()) * labels[tied].mean()) / k)

def binary_metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or y.shape != scores.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Nonempty aligned binary labels and finite scores required")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p, r, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores) if y.any() else 0.
    return dict(average_precision=float(ap), auprc_trapezoid=float(auc(r, p)),
                auroc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
                prevalence=float(y.mean()), positives=int(y.sum()), pairs=len(y),
                **{f"top{k}_supported_fraction": fractional_topk(y, scores, k) for k in (10, 25, 50, 100, 200)})

def popularity(train, edges):
    """Only observed positive training edges contribute; no test degree or labels."""
    positive = np.asarray(train) == 1
    incoming, outgoing = positive.sum(axis=0), positive.sum(axis=1)
    return {"Prior target in-degree": incoming[edges[:, 1]], "Prior regulator out-degree": outgoing[edges[:, 0]]}

def paired(delta, seed=42, indices=None):
    delta = np.asarray(delta, float)
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Paired comparison needs a complete finite regulator cohort")
    if indices is None:
        indices = np.random.default_rng(seed).integers(len(delta), size=(10000, len(delta)))
    boot = delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return dict(mean=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                wins=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))

def top_weights(scores, k=100):
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) < k or not np.isfinite(scores).all():
        raise ValueError("Insufficient finite targets for top-K")
    threshold = np.partition(scores, len(scores) - k)[-k]
    above, tied = scores > threshold, scores == threshold
    weights = above.astype(float)
    weights[tied] = (k - above.sum()) / tied.sum()
    return weights

def pool_metrics(labels, scores, edges):
    values = binary_metrics(labels, scores)
    per_tf = []
    for tf in np.unique(edges[:, 0]):
        mask = edges[:, 0] == tf
        per_tf.append(average_precision_score(labels[mask], scores[mask]) if labels[mask].any() else 0.)
    return dict(values, all_tf_ap=float(np.mean(per_tf)))

def all_tf_ap(edges, labels, scores):
    """Mean AP over every eligible regulator in the pool, with zero for regulators without a positive."""
    from sklearn.metrics import average_precision_score
    values = []
    for tf in np.unique(edges[:, 0]):
        mask = edges[:, 0] == tf
        values.append(average_precision_score(labels[mask], scores[mask]) if labels[mask].any() else 0.)
    return float(np.mean(values))

def match_pairs(edges,labels,features,caliper=.5,seed=42):
    rng=np.random.default_rng(seed);pairs=[]
    for tf in np.unique(edges[:,0]):
        pos=np.flatnonzero((edges[:,0]==tf)&(labels==1));unknown=np.flatnonzero((edges[:,0]==tf)&(labels==0));available=np.ones(len(unknown),bool)
        for i in rng.permutation(pos):
            delta=features[edges[unknown,1]]-features[edges[i,1]]
            allowed=available&np.all(np.abs(delta)<=caliper,axis=1)
            if not allowed.any():continue
            distance=np.sum(delta*delta,axis=1);distance[~allowed]=np.inf;j=int(np.argmin(distance));available[j]=False
            pairs.append((i,int(unknown[j]),float(np.sqrt(distance[j]))))
    return np.asarray(pairs,dtype=float).reshape(-1,3)
