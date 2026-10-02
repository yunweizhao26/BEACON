"""Paired setting bootstrap, preserving the recorded draw order."""
import numpy as np

def paired(frame, left, right, metric, rng_seed=42):
    """Setting-level paired comparison of left minus right, using split means."""
    from scipy.stats import wilcoxon
    wide = frame.pivot_table(index="dataset", columns="score", values=metric)
    wide = wide[[left, right]].dropna()
    diff = (wide[left] - wide[right]).to_numpy()
    rng = np.random.default_rng(rng_seed)
    boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1) if len(diff) else np.array([np.nan])
    nonzero = diff[diff != 0]
    return {"left": left, "right": right, "metric": metric, "settings": len(diff), "left_wins": int((diff > 0).sum()),
            "right_wins": int((diff < 0).sum()), "ties": int((diff == 0).sum()), "within_0.03": int((np.abs(diff) < .03).sum()),
            "mean_difference": float(diff.mean()) if len(diff) else np.nan, "median_difference": float(np.median(diff)) if len(diff) else np.nan,
            "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
            "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else np.nan}
