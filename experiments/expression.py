"""Expression controls with original memory layout."""
import numpy as np
def control_expression(x, control):
    """Return the controlled genes x cells matrix and any permutation used."""
    rng = np.random.default_rng(2718)
    if control == "real":
        return x.copy(), None
    if control == "gene_permuted":
        permutation = rng.permutation(len(x))
        return x[permutation], permutation
    if control == "cell_shuffled":
        order = rng.random(x.shape).argsort(axis=1)
        return np.take_along_axis(x, order, axis=1), None
    if control == "random":
        return rng.choice(x.ravel(), size=x.shape, replace=True).astype(x.dtype), None
    raise ValueError(control)

def transform(expression, control, *, match_layout=True):
    if control in ("real", "random"):
        return expression
    changed, _ = control_expression(np.asarray(expression), control)
    changed = changed.astype(expression.dtype)
    if match_layout and np.asarray(expression).flags.f_contiguous:
        changed = np.asfortranarray(changed)
    return changed
if __name__ == "__main__":
    from experiments.run import main
    main("expression")
