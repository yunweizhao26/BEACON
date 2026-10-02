"""Expression features; reduction always uses random_state=0."""
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
def expression_features(data, *, components=64, pca=False, scale=False):
    if isinstance(components, bool) or not isinstance(components, int) or components < 1:
        raise ValueError("components must be a positive integer")
    x = np.array(data) if not isinstance(data, np.ndarray) else data
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite genes by cells expression")
    if scale:
        x = StandardScaler().fit_transform(x)
    model = (PCA if pca else FactorAnalysis)(n_components=components, random_state=0)
    return np.array(model.fit_transform(x))
def feature_control(features, *, random_features=False, permuted_features=False):
    if random_features and permuted_features:
        raise ValueError("Choose at most one feature control")
    if random_features:
        return np.random.default_rng(2718).normal(size=features.shape).astype(np.float32)
    if permuted_features:
        return features[np.random.default_rng(2718).permutation(len(features))]
    return features.copy()
