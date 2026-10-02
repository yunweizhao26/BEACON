"""Calibrated AD labels for K562/RPE1; no primary effect-size threshold."""
from functools import lru_cache
import math
import h5py
from numba import njit, prange
import numpy as np
import pandas as pd
from scipy import stats
ALPHA = .05
MAX_NULL_RATE = .001
def require(condition, message):
    if not condition:
        raise ValueError(message)


def column(group, name):
    """Read metadata only; decode both AnnData references and modern categories."""
    node = group[name]
    if isinstance(node, h5py.Group):
        cats = node["categories"]
        values = cats.asstr()[:] if h5py.check_string_dtype(cats.dtype) else cats[:]
        codes = node["codes"][:]
        require(np.all(codes >= 0), f"Missing categories: {node.name}")
        return values[codes]
    if "categories" in node.attrs:
        cats = group.file[node.attrs["categories"]]
        values = cats.asstr()[:] if h5py.check_string_dtype(cats.dtype) else cats[:]
        codes = node[:]
        require(np.all(codes >= 0), f"Missing categories: {node.name}")
        return values[codes]
    return node.asstr()[:] if h5py.check_string_dtype(node.dtype) else node[:]


def read_rows(dataset, rows):
    """At most 256 selected cells per HDF5 read; never load the full raw matrix."""
    rows = np.asarray(rows, dtype=np.int64)
    require(len(rows) > 0 and np.all(np.diff(rows) > 0), "Raw row indices must be sorted and unique")
    require(isinstance(dataset, h5py.Dataset) and dataset.ndim == 2,
            "Expected the inspected dense, backed release X; unsupported storage must be audited")
    result = np.empty((len(rows), dataset.shape[1]), dtype=np.float64)
    for start in range(0, len(rows), 256):
        result[start:start + 256] = dataset[rows[start:start + 256], :]
    require(np.isfinite(result).all() and (result >= 0).all(), "Invalid raw counts")
    return result


def standardize(values, mean, sd):
    residual = values - mean
    # Release semantics (LABEL_SPEC amendment 1): residual / 0 is +/-inf, 0 / 0 is 0.
    infinite = (sd == 0) & (residual != 0)
    return np.where(infinite, np.sign(residual) * np.inf, residual / np.where(sd == 0, 1., sd))


def bh(p):
    """Benjamini-Hochberg on exactly the supplied family; no silent NaN removal."""
    p = np.asarray(p, dtype=float)
    require(p.size > 0 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid BH family")
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    adjusted = np.minimum.accumulate((flat[order] * len(flat) / np.arange(1, len(flat) + 1))[::-1])[::-1]
    result = np.empty_like(flat)
    result[order] = np.minimum(adjusted, 1)
    return result.reshape(p.shape)


def correct(p, scope):
    return bh(p) if scope == "global" else np.stack([bh(row) for row in p])


@lru_cache(maxsize=1024)
def ad_sigma(n, m):
    """Scholz-Stephens finite-sample variance, the same expression as SciPy."""
    N, k = n + m, 2
    require(min(n, m) > 0 and N > 3, "AD needs nonempty samples and total size > 3")
    H = 1 / n + 1 / m
    hs = (1 / np.arange(N - 1, 1, -1, dtype=float)).cumsum()
    h, g = hs[-1] + 1, (hs / np.arange(2, N)).sum()
    a = (4*g - 6)*(k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k*k + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k*k + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k*k - 4*h*k
    return np.sqrt((a*N**3 + b*N**2 + c*N + d) / ((N - 1)*(N - 2)*(N - 3)))


@njit(parallel=True, cache=False)
def ad_statistic(case, control_sorted, sigma):
    """Merge sorted samples and accumulate midrank tie blocks in O(n+m) per gene.

    Inputs are gene x cell, float64. Control sorting is cached across real TFs.
    Degenerate pooled constants return NaN; caller explicitly assigns P=1.
    No fastmath: tie handling and summation must match the reference test.
    """
    genes, n = case.shape
    m = control_sorted.shape[1]
    N = n + m
    result = np.empty(genes)
    for gene in prange(genes):
        x, y = np.sort(case[gene]), control_sorted[gene]
        i, j, total, distinct = 0, 0, 0.0, 0
        while i < n or j < m:
            value = x[i] if j == m or (i < n and x[i] <= y[j]) else y[j]
            ii, jj = i, j
            while i < n and x[i] == value:
                i += 1
            while j < m and y[j] == value:
                j += 1
            tie = i - ii + j - jj
            B = ii + jj + tie / 2.0
            denom = B*(N - B) - N*tie/4.0
            if denom > 0:
                dx = N*(ii + (i - ii)/2.0) - B*n
                dy = N*(jj + (j - jj)/2.0) - B*m
                total += tie/N * (dx*dx/n + dy*dy/m) / denom
            distinct += 1
        result[gene] = (total*(N - 1)/N - 1)/sigma if distinct > 1 else np.nan
    return result


def probability(statistic, table):
    require(np.isfinite(statistic[~np.isnan(statistic)]).all(), "Infinite AD statistic")
    # This is linear interpolation in P, equivalent to interp1d(kind='linear').
    p = np.interp(statistic, table[:, 0], table[:, 1], left=1., right=np.finfo(float).tiny)
    return np.where(np.isnan(statistic), 1., np.clip(p, np.finfo(float).tiny, 1.))


def matched_indices(control_gems, template_gems, rng):
    """Exact counts per GEM, without replacement, with at least two references left."""
    selected = []
    gems, sizes = np.unique(template_gems, return_counts=True)
    for gem, count in zip(gems, sizes):
        available = np.flatnonzero(control_gems == gem)
        require(len(available) >= count + 2,
                f"Cannot match GEM {gem}: need {count} fake cells plus two remaining controls; have {len(available)}")
        selected.extend(rng.choice(available, int(count), replace=False).tolist())
    return np.sort(np.array(selected, dtype=np.int64))


def calibration_summary(q, effects, tested):
    require(q.shape == effects.shape == tested.shape and tested.any(axis=1).all(), "Invalid calibration axes")
    require(np.isfinite(q).all() and not np.isnan(effects[tested]).any(), "Undefined calibration")
    finite = tested & np.isfinite(effects)
    yes = (q < ALPHA) & tested
    rates = yes.sum(axis=1) / tested.sum(axis=1)
    mean_rate = math.fsum(rates.tolist()) / len(rates)
    return dict(passed=mean_rate <= MAX_NULL_RATE,
                fraction_with_any_yes=float(yes.any(axis=1).mean()),
                mean_per_gene_yes_rate=mean_rate,
                per_fake_yes_rates=rates.tolist(), per_fake_yes_counts=yes.sum(axis=1).tolist(),
                sensitivity_floor=float(np.quantile(np.abs(effects[finite]), .999, method="linear")),
                sensitivity_quantile=.999, sensitivity_pairs=int(finite.sum()),
                infinite_effect_pairs=int((tested & ~finite).sum()),
                maximum_allowed_mean_per_gene_yes_rate=MAX_NULL_RATE)


def make_labels(schema, p, q, effects, floor):
    """Preserve frozen tested masks, expose discovery/self exclusions explicitly."""
    shape = schema["tested"].shape
    require(p.shape == q.shape == effects.shape == shape, "Label schema axes differ")
    mask = schema["evaluation_mask"]
    require(np.isfinite(p[schema["tested"]]).all() and np.isfinite(q[schema["tested"]]).all(), "Missing tested P values")
    require(np.isfinite(effects[schema["tested"]]).all(), "Missing tested effects")
    response = (q < ALPHA) & mask
    return dict(tf_genes=schema["tf_genes"], target_genes=schema["target_genes"], constructs=schema["constructs"],
                tested=schema["tested"].copy(), evaluation_mask=mask.copy(), response=response,
                standardized_effects=effects, pvalues=p, adjusted_pvalues=q,
                response_sensitivity=response & (np.abs(effects) >= floor),
                sensitivity_floor=np.array(floor))


def response_labels(schema, pvalues, effects, null_pvalues, null_effects, null_tested,
                    *, work, family_genes, genes):
    """All columns in each perturbation's BH family must be present before subsetting."""
    require(pvalues.shape == (len(work), len(family_genes)), "Incomplete AD test family")
    require(effects.shape == (len(work), len(genes)), "Effect axes differ from raw genes")
    all_q = correct(pvalues, "per_perturbation")
    calibration = calibration_summary(correct(null_pvalues, "per_perturbation"), null_effects, null_tested)
    require(len(null_pvalues) == 200 and calibration["passed"], "The 200-fake null calibration must pass")
    calibration.update(fake_knockdowns=200, scope="per_perturbation")
    rows = np.array([list(work).index(c) for c in schema["constructs"]])
    gene_pos = {g: i for i, g in enumerate(genes)}
    cols = np.array([gene_pos[g] for g in schema["target_genes"]])
    family_pos = {g: i for i, g in enumerate(family_genes)}
    covered = np.array([g in family_pos for g in schema["target_genes"]])
    fcols = np.array([family_pos[g] for g in schema["target_genes"][covered]])
    p = np.full(schema["tested"].shape, np.nan)
    q = np.full_like(p, np.nan)
    p[:, covered], q[:, covered] = pvalues[np.ix_(rows, fcols)], all_q[np.ix_(rows, fcols)]
    data = make_labels(schema, p, q, effects[np.ix_(rows, cols)], calibration["sensitivity_floor"])
    for key in ("target_symbols", "discovery_mask"):
        if key in schema:
            data[key] = schema[key]
    return data, calibration

def test_counts(case, control, table):
    """Inputs are standardized genes by cells arrays; probabilities use kSamples interpolation."""
    case = np.ascontiguousarray(case, dtype=np.float64)
    control = np.sort(np.asarray(control, dtype=np.float64), axis=1)
    return probability(ad_statistic(case, control, ad_sigma(case.shape[1], control.shape[1])), table)

def real_response(raw, bulk, plan, meta, construct, controls_sorted, table):
    """AD probabilities and effects using the released within-GEM normalization."""
    idx=int(np.flatnonzero(meta["constructs"]==construct)[0])
    rows=np.flatnonzero(meta["qc"] & (meta["codes"]==idx))
    with h5py.File(raw,"r") as handle:x=read_rows(handle["X"],rows)
    library=x.sum(axis=1);require((library>0).all(),"Zero perturbation library")
    family={g:i for i,g in enumerate(meta["genes"])}
    indices=np.array([family[g] for g in plan["family_genes"]])
    x=x[:,indices]*(float(meta["median_library"])/library[:,None])
    gems=meta["gems"][rows]
    x=standardize(x,meta["mean"][np.ix_(gems,indices)],meta["sd"][np.ix_(gems,indices)])
    require(not np.isnan(x).any(),"Undefined normalized expression")
    with h5py.File(bulk,"r") as handle:effects=handle["X"][idx,:].astype(float)
    recomputed=x.mean(axis=0);released_effect=~np.isfinite(recomputed)
    effects[indices]=np.where(released_effect,effects[indices],recomputed)
    statistic=np.empty(len(indices))
    for start in range(0,len(indices),64):
        cols=indices[start:start+64]
        statistic[start:start+64]=ad_statistic(np.ascontiguousarray(x[:,start:start+64].T),np.ascontiguousarray(controls_sorted[cols]),ad_sigma(len(rows),controls_sorted.shape[1]))
    return dict(pvalues=probability(statistic,table),standardized_effects=effects)

def fake_response(plan,meta,construct,scaled,table,task):
    require(0<=task<200,"Expected one of 200 fake knockdowns")
    idx=int(np.flatnonzero(meta["constructs"]==construct)[0])
    template=np.flatnonzero(meta["qc"] & (meta["codes"]==idx))
    rng=np.random.default_rng(np.random.SeedSequence([42,task]))
    selected=matched_indices(meta["control_gems"],meta["gems"][template],rng)
    keep=np.ones(len(meta["controls"]),dtype=bool);keep[selected]=False
    genes={g:i for i,g in enumerate(meta["genes"])};indices=np.array([genes[g] for g in plan["family_genes"]])
    statistic,effect=np.empty(len(indices)),np.empty(len(indices))
    for start in range(0,len(indices),64):
        cols=indices[start:start+64];block=np.asarray(scaled[:,cols]).copy()
        for gem in range(len(meta["gem_labels"])):
            belongs=meta["control_gems"]==gem;ref=block[belongs & keep]
            require(len(ref)>=2,"Insufficient remaining GEM controls")
            mean,sd=ref.mean(axis=0),ref.std(axis=0,ddof=1)
            block[belongs]=standardize(block[belongs],mean,sd)
        x=np.ascontiguousarray(block[selected].T);y=np.sort(block[keep].T,axis=1)
        statistic[start:start+len(cols)]=ad_statistic(x,y,ad_sigma(len(selected),int(keep.sum())))
        effect[start:start+len(cols)]=x.mean(axis=1)-y.mean(axis=1)
    tested=np.array(plan["family_genes"])!=construct.rsplit("_",1)[-1]
    return dict(pvalues=probability(statistic,table),standardized_effects=effect,tested=tested,
                selected_control_rows=meta["controls"][selected],matched_gems=meta["control_gems"][selected])


def prepare_controls(raw, bulk, destination, line):
    """Create exact control normalization caches from public HDF5 count matrices."""
    destination.mkdir(parents=True, exist_ok=False)
    with h5py.File(bulk, "r") as handle:
        constructs = column(handle["obs"], "gene_transcript").astype(str)
        core_ids = constructs[column(handle["obs"], "core_control").astype(bool)]
        reported = column(handle["obs"], "num_cells_filtered").astype(float)
        bulk_genes = column(handle["var"], "gene_id").astype(str)
    with h5py.File(raw, "r") as handle:
        genes = column(handle["var"], "gene_id").astype(str)
        require(np.array_equal(genes, bulk_genes), "Raw and bulk gene axes differ")
        require(len(set(genes)) == len(genes) and len(set(constructs)) == len(constructs), "Ambiguous source axes")
        obs = handle["obs"]
        group = column(obs, "gene_transcript").astype(str)
        gems = column(obs, "gem_group")
        umi = column(obs, "UMI_count")
        adjusted = column(obs, "core_adjusted_UMI_count")
        mito = column(obs, "mitopercent")
        qc = (adjusted >= (2000 if line == "k562" else 3000)) & (mito < (.25 if line == "k562" else .11))
        qc &= np.isfinite(umi) & (umi > 0)
        require(handle["X"].shape == (len(group), len(genes)), "Unexpected raw X shape")
        rows = np.flatnonzero(qc & np.isin(group, core_ids))
        require(len(rows) > 1000, "Too few passing core controls")
        gem_labels, gem_index = np.unique(gems, return_inverse=True)
        control_gems = gem_index[rows]
        require(np.bincount(control_gems, minlength=len(gem_labels)).min() >= 2, "GEM without two controls")
        ids = {v: i for i, v in enumerate(constructs)}
        require(set(group[qc]) <= set(ids), "Raw construct missing from bulk")
        codes = pd.Categorical(group, categories=constructs).codes
        counts = np.bincount(codes[qc], minlength=len(constructs))
        # The release leaves num_cells_filtered empty only for individual non-targeting guides (71 in K562).
        finite = np.isfinite(reported)
        require(all("non-targeting" in c for c in constructs[~finite]), "Unreported cell counts outside non-targeting guides")
        require(np.array_equal(counts[finite], reported[finite].astype(int)), "QC cell counts disagree with the release; stop before testing")
        # The author's equalize_UMI_counts uses matrix row sums, not obs UMI_count.
        controls = read_rows(handle["X"], rows)
        library = controls.sum(axis=1)
        require((library > 0).all(), "Zero retained-gene library size")
        median_library = float(np.median(library))
        controls *= median_library / library[:, None]
    mu = np.empty((len(gem_labels), len(genes)))
    sd = np.empty_like(mu)
    for gem in range(len(gem_labels)):
        selected = controls[control_gems == gem]
        mu[gem], sd[gem] = selected.mean(axis=0), selected.std(axis=0, ddof=1)
    zero_sd = int((sd == 0).sum())
    np.save(destination / "controls_scaled.npy", controls)
    for start in range(0, len(controls), 512):
        subset = control_gems[start:start + 512]
        controls[start:start + 512] = standardize(controls[start:start + 512], mu[subset], sd[subset])
    sorted_controls = np.lib.format.open_memmap(destination / "controls_sorted.npy", mode="w+",
                                               dtype="float64", shape=(len(genes), len(rows)))
    for start in range(0, len(genes), 64):
        sorted_controls[start:start + 64] = np.sort(controls[:, start:start + 64].T, axis=1)
    sorted_controls.flush()
    del controls, sorted_controls
    np.savez_compressed(destination / "metadata.npz", genes=genes, constructs=constructs, counts=counts,
              codes=codes, qc=qc, gems=gem_index, gem_labels=gem_labels, controls=rows,
              control_gems=control_gems, mean=mu, sd=sd, median_library=np.array(median_library))
