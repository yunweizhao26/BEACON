"""Frozen splitting and readout helpers."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import split_checks
def make_split(H, tf_indices, split_seed, coverage, ratio, corruption=0):
    """Reserve 10% of each reference-label class for validation and test once.

    Only revealed training positives are supplied to the model. Hidden
    training positives remain eligible for unlabeled training sampling.
    The full validation/test pools do not depend on coverage, ratio or corruption.
    """
    if not (0 < coverage <= .8 and ratio >= 1 and 0 <= corruption <= .2):
        raise ValueError('Expected 0 < coverage <= .8, ratio >= 1 and 0 <= corruption <= .2')
    n = len(H)
    allowed = np.zeros(H.shape, dtype=bool)
    allowed[tf_indices, :] = True
    np.fill_diagonal(allowed, False)
    edges = np.argwhere(allowed)
    labels = H[tuple(edges.T)].astype(np.int8)
    rng = np.random.default_rng(split_seed)
    positives = rng.permutation(np.flatnonzero(labels == 1))
    absent = rng.permutation(np.flatnonzero(labels == 0))
    if len(positives) < 20:
        raise ValueError('At least 20 reference positives are needed for this protocol')
    npart, upart = max(1, int(.1 * len(positives))), max(1, int(.1 * len(absent)))
    validation = np.concatenate([positives[:npart], absent[:upart]])
    test = np.concatenate([positives[npart:2*npart], absent[upart:2*upart]])
    training_p = positives[2*npart:]
    training_u = absent[2*upart:]
    n_revealed = min(len(training_p), max(1, int(round(coverage * len(positives)))))
    supplied = training_p[:n_revealed].copy()
    unknown = np.concatenate([training_p[n_revealed:], training_u])
    noise_rng = np.random.default_rng(split_seed + 101)
    n_corrupted = int(round(corruption * len(supplied)))
    removed = supplied[:n_corrupted].copy()
    replacements = np.array([], dtype=int)
    if n_corrupted:
        replacements = noise_rng.choice(training_u, n_corrupted, replace=False)
        supplied[:n_corrupted] = replacements
        unknown = np.concatenate([np.setdiff1d(unknown, replacements), removed])
    sampling_rng = np.random.default_rng(split_seed + 202)
    n_sampled = ratio * len(supplied)
    if n_sampled > len(unknown):
        raise ValueError('Training unlabeled pool is too small for the requested ratio')
    sampled = sampling_rng.choice(unknown, n_sampled, replace=False)
    train = np.full(H.shape, -1, dtype=np.int8)
    valid = train.copy()
    test_matrix = train.copy()
    train[tuple(edges[supplied].T)] = 1
    train[tuple(edges[sampled].T)] = 0
    valid[tuple(edges[validation].T)] = labels[validation]
    test_matrix[tuple(edges[test].T)] = labels[test]
    checks = split_checks(train, valid, test_matrix)
    assert not any(checks.values()), checks
    return {'train': train, 'valid': valid, 'test': test_matrix, 'eligible_edges': edges,
            'supplied_positive_edges': edges[supplied], 'sampled_unlabeled_edges': edges[sampled],
            'removed_prior_edges': edges[removed], 'replacement_prior_edges': edges[replacements],
            'hidden_reference_positives_sampled_as_unlabeled': int(labels[sampled].sum()),
            'reference_positives': int(labels.sum()), 'checks': checks}

def classifier(x, y):
    fitted = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
    return fitted.fit(x, y)

def pair_features(embedding, edges):
    return np.concatenate([embedding[edges[:, 0]], embedding[edges[:, 1]]], axis=1)

def decoder_scores(encoder, features, edges):
    """Score ordered pairs with the encoder's auxiliary pair decoder."""
    device = next(encoder.parameters()).device
    with torch.no_grad():
        z = encoder.encoder(torch.as_tensor(features, dtype=torch.float32, device=device))
        scores = []
        for start in range(0, len(edges), 8192):
            ij = torch.as_tensor(edges[start:start + 8192], device=device)
            scores.append(encoder.edge_logits(z[ij[:, 0]], z[ij[:, 1]]).cpu().numpy())
    return np.concatenate(scores)
