"""Reference-label PU control and held-out probability diagnostics."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler


def nnpu_risk(logits, labels, class_prior, observed_fraction):
    positive = labels == 1
    lp = torch.nn.functional.softplus(-logits[positive]).mean()
    ln_p = torch.nn.functional.softplus(logits[positive]).mean()
    ln_u = torch.nn.functional.softplus(logits[~positive]).mean()
    # Reconstruct the training marginal from observed-P and remaining-U strata.
    negative_risk = (observed_fraction - class_prior) * ln_p + (1 - observed_fraction) * ln_u
    return class_prior * lp + torch.clamp(negative_risk, min=0)


def fit_nnpu(train_x, labels, valid_x, test_x, class_prior, observed_fraction, device):
    """Linear nnPU head; prevalence estimated only from the full validation pool."""
    requested_prior = float(class_prior)
    class_prior = max(requested_prior, float(observed_fraction))
    if not 0 < class_prior < 1:
        raise ValueError('Class prior must be in (0,1) and at least the observed positive fraction')
    scaler = StandardScaler().fit(train_x)
    x = torch.tensor(scaler.transform(train_x), dtype=torch.float32, device=device)
    y = torch.tensor(labels, device=device)
    torch.manual_seed(42)
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(500):
        optimizer.zero_grad()
        logits = model(x).flatten()
        risk = nnpu_risk(logits, y, class_prior, observed_fraction)
        objective = risk + model.weight.square().sum() / (2 * len(x))
        objective.backward()
        optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for values in [valid_x, test_x]:
            values = torch.tensor(scaler.transform(values), dtype=torch.float32, device=device)
            predictions.append(model(values).flatten().sigmoid().cpu().numpy())
    return predictions, {'class_prior_from_validation': requested_prior, 'class_prior_used': float(class_prior), 'prior_floor_applied': class_prior > requested_prior,
                         'observed_fraction_of_training_pairs': float(observed_fraction),
                         'epochs': 500, 'learning_rate': .01, 'empirical_risk': float(risk.detach().cpu())}


def probability_report(labels, probabilities):
    labels = np.asarray(labels)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    if labels.shape != p.shape or p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected aligned, nonempty, finite probability and label vectors')
    cuts = np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1]))
    assignments = np.searchsorted(cuts, p, side='right')
    bins = []
    for group in np.unique(assignments):
        indices = assignments == group
        bins.append({'pairs': int(indices.sum()), 'positives': int(labels[indices].sum()),
                     'mean_probability': float(p[indices].mean()),
                     'supported_fraction': float(labels[indices].mean())})
    return {'brier': float(brier_score_loss(labels, p)), 'log_loss': float(log_loss(labels, p, labels=[0, 1])),
            'quantile_reliability_bins': bins, 'binning': 'Up to 10 quantile bins; identical probabilities remain together',
            'ece': float(sum(b['pairs'] * abs(b['mean_probability'] - b['supported_fraction']) for b in bins) / len(p))}


def selective_report(labels, probabilities, uncertainty):
    labels, p, uncertainty = map(np.asarray, (labels, probabilities, uncertainty))
    if labels.shape != p.shape or p.shape != uncertainty.shape or not np.isfinite(uncertainty).all():
        raise ValueError('Selective inputs must be aligned and finite')
    rows = []
    for coverage in [1., .9, .75, .5, .25]:
        count = max(1, int(round(coverage * len(labels))))
        cutoff = np.partition(uncertainty, count - 1)[count - 1]
        weights = (uncertainty < cutoff).astype(float)
        ties = uncertainty == cutoff
        fraction = (count - weights.sum()) / ties.sum()
        weights[ties] = fraction
        positive = float(np.sum(weights * labels)); negative = count - positive
        predicted = p >= .5
        rows.append({'coverage': coverage, 'pairs': count, 'positives': positive,
                     'error_rate': float(np.sum(weights * (predicted != labels)) / count),
                     'sensitivity': float(np.sum(weights * labels * predicted) / positive) if positive else None,
                     'false_positive_rate': float(np.sum(weights * (1-labels) * predicted) / negative) if negative else None,
                     'brier': float(np.sum(weights * (p-labels)**2) / count),
                     'cutoff_tie_pairs': int(ties.sum()), 'boundary_tie_weight': float(fraction)})
    return rows


def calibrate(valid_labels, valid_predictions, test_labels, test_predictions, latent_variance):
    """Platt scaling uses validation labels; test labels enter reporting only."""
    reports, calibrated = {}, {}
    for name, test_p in test_predictions.items():
        if name == 'degree_product':
            continue
        valid_p = np.clip(valid_predictions[name], 1e-7, 1 - 1e-7)
        test_p = np.clip(test_p, 1e-7, 1 - 1e-7)
        logit_valid = np.log(valid_p / (1 - valid_p))[:, None]
        logit_test = np.log(test_p / (1 - test_p))[:, None]
        fitted = LogisticRegression(C=1, max_iter=1000, random_state=42).fit(logit_valid, valid_labels)
        p = fitted.predict_proba(logit_test)[:, 1]
        calibrated[name] = p
        reports[name] = {'raw': probability_report(test_labels, test_p),
                         'platt': probability_report(test_labels, p),
                         'platt_slope': float(fitted.coef_[0, 0]), 'platt_intercept': float(fitted.intercept_[0])}
        criteria = {'probability_entropy_order': p * (1 - p)}
        if name == 'beacon':
            criteria['latent_variance'] = latent_variance
        reports[name]['selective'] = {}
        for criterion, uncertainty in criteria.items():
            reports[name]['selective'][criterion] = selective_report(test_labels, p, uncertainty)
    reports['validation_prevalence_constant'] = probability_report(test_labels, np.full(len(test_labels), valid_labels.mean()))
    return calibrated, reports

