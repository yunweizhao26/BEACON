"""Donor-adjusted joint DESeq2 and donor-consistent response support."""
import math
import json
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
DONORS = ("Donor1", "Donor2")
NFAKES = 200

def calibrate(bulk, samples, cache, tf_symbols, tf_genes, *, workers=4):
    """Fit all 200 null contrasts with the same donor and cell-count matching."""
    cells=sparse.csr_matrix((cache["data"],cache["indices"],cache["indptr"]),shape=tuple(cache["shape"]))
    keep=bulk["counts"].sum(axis=0)>=10
    rows=[];effects=[]
    for task in range(NFAKES):
        template,selected=sample_fake(samples,tf_symbols,cache["donors"],task)
        counts,libraries,meta=add_fake(bulk,samples,cells,cache["full_rna_library_counts"],cache["donors"],selected)
        assert np.array_equal(keep,counts.sum(axis=0)>=10)
        targets=np.append(tf_symbols,"FAKE")
        _,lfc,tested,q=joint_fit(counts,meta,bulk["genes"],targets,keep,workers)
        primary=labels(tested[-1],q[-1],lfc[-1],donor_effects(counts,libraries,meta,"FAKE"))
        evaluation=tested[-1] & (bulk["genes"]!=tf_genes[template])
        assert evaluation.any()
        rows.append(dict(task=task,yes=int(primary[evaluation].sum()),tested=int(evaluation.sum())))
        absolute=np.abs(lfc[-1]);effects.extend(absolute[np.isfinite(absolute) & (bulk["genes"]!=tf_genes[template])].tolist())
    receipt=gate_arithmetic([r["yes"] for r in rows],[r["tested"] for r in rows])
    receipt.update(fakes=NFAKES,per_fake=rows,data_floor=float(np.quantile(effects,.999,method="linear")))
    if not receipt["passed"]:raise ValueError("T-cell null calibration failed")
    return receipt

def response_labels(bulk, samples, tf_symbols, tf_genes, calibration, *, workers=4):
    """Global BH includes every contrast and self pair before discovery exclusions."""
    if not calibration["passed"] or calibration["fakes"]!=NFAKES:raise ValueError("Calibration required")
    counts=bulk["counts"];keep=counts.sum(axis=0)>=10
    pvalues,lfc,tested,q=joint_fit(counts,samples,bulk["genes"],tf_symbols,keep,workers)
    effects=np.stack([donor_effects(counts,bulk["full_rna_library_counts"],samples,target) for target in tf_symbols],axis=1)
    return dict(tf_genes=tf_genes,tf_symbols=tf_symbols,target_genes=bulk["genes"],tested=tested,
        pvalues=pvalues,log2foldchange=lfc,adjusted_pvalues=q,donor_log2foldchange=effects,
        primary_support=labels(tested,q,lfc,effects),fixed_floor_0p5=labels(tested,q,lfc,effects,.5),data_floor=labels(tested,q,lfc,effects,calibration["data_floor"]))
def fit(counts,metadata,workers):
    dds=DeseqDataSet(counts=counts,metadata=metadata,design='~donor+perturbation',refit_cooks=True,n_cpus=workers,quiet=True)
    dds.deseq2();return dds


def contrast(dds,target,workers):
    stat=DeseqStats(dds,contrast=['perturbation',target,'NO-TARGET'],alpha=.05,cooks_filter=True,independent_filter=False,n_cpus=workers,quiet=True)
    stat.summary();return stat.results_df

def bh_family(pvalues, lfc):
    """The complete global family, including self-pairs before exclusions."""
    tested = np.isfinite(pvalues) & np.isfinite(lfc)
    assert tested.any() and np.all((pvalues[tested] >= 0) & (pvalues[tested] <= 1))
    q = np.full_like(pvalues, np.nan, dtype=float)
    q[tested] = multipletests(pvalues[tested], method="fdr_bh")[1]
    return tested, q


def labels(tested, q, lfc, donor_effects, floor=0.):
    concordant = np.all(np.sign(donor_effects) == np.sign(lfc)[None], axis=0)
    return tested & (q < .05) & (np.abs(lfc) >= floor) & concordant


def sample_fake(samples, targets, cell_donors, task):
    rng = np.random.default_rng(np.random.SeedSequence([42, task]))
    template = int(rng.integers(len(targets)))
    chosen = []
    for donor in DONORS:
        n = int(samples.loc[(samples.donor == donor) & (samples.gene == targets[template]), "cells"].iloc[0])
        pool = np.flatnonzero(cell_donors == donor)
        assert 0 < n < len(pool), (targets[template], donor, n, len(pool))
        chosen.extend(rng.choice(pool, size=n, replace=False).tolist())
    selected = np.array(chosen, dtype=int)
    assert len(np.unique(selected)) == len(selected)
    return template, selected


def add_fake(bulk, samples, cells, libraries, donors, selected):
    assert selected.ndim == 1 and len(np.unique(selected)) == len(selected)
    assert np.all((selected >= 0) & (selected < cells.shape[0]))
    counts = bulk["counts"].copy()
    full = bulk["full_rna_library_counts"].copy()
    samples = samples.copy()
    fake_counts, fake_libraries, rows = [], [], []
    for donor in DONORS:
        chosen = selected[donors[selected] == donor]
        removed = np.asarray(cells[chosen].sum(axis=0)).ravel()
        removed_library = libraries[chosen].sum()
        control, = np.flatnonzero((samples.donor == donor) & (samples.gene == "NO-TARGET"))
        counts[control] -= removed
        full[control] -= removed_library
        samples.loc[samples.index[control], "cells"] -= len(chosen)
        fake_counts.append(removed); fake_libraries.append(removed_library)
        rows.append(dict(sample=donor + "__FAKE", donor=donor, gene="FAKE", cells=len(chosen)))
    counts = np.vstack([counts, fake_counts])
    full = np.concatenate([full, fake_libraries])
    meta = pd.concat([samples, pd.DataFrame(rows).set_index("sample")])
    assert np.all(counts >= 0) and np.all(full > 0) and meta.index.is_unique
    assert np.all(meta.cells > 0)
    assert np.array_equal(counts.sum(axis=0), bulk["counts"].sum(axis=0))
    assert full.sum() == bulk["full_rna_library_counts"].sum()
    return counts, full, meta


def donor_effects(counts, libraries, samples, target):
    effects = []
    for donor in DONORS:
        t, = np.flatnonzero((samples.donor == donor) & (samples.gene == target))
        c, = np.flatnonzero((samples.donor == donor) & (samples.gene == "NO-TARGET"))
        effects.append(np.log2((1e6 * counts[t] / libraries[t] + .5) / (1e6 * counts[c] / libraries[c] + .5)))
    return np.array(effects)


def joint_fit(counts, samples, genes, targets, keep, workers):
    # Reuse the original formula/design builder and target-vs-NO-TARGET contrast.
    frame = pd.DataFrame(counts[:, keep], index=samples.index, columns=genes[keep])
    dds = fit(frame, samples[["donor", "gene"]].rename(columns={"gene": "perturbation"}), workers)
    pvalues = np.full((len(targets), len(genes)), np.nan)
    lfc = pvalues.copy()
    for i, target in enumerate(targets):
        result = contrast(dds, target, workers).reindex(genes[keep])
        pvalues[i, keep] = result.pvalue
        lfc[i, keep] = result.log2FoldChange
    tested, q = bh_family(pvalues, lfc)
    return pvalues, lfc, tested, q


def gate_arithmetic(yes, tested):
    yes, tested = np.asarray(yes), np.asarray(tested)
    assert yes.shape == tested.shape and yes.size and np.all(tested > 0)
    assert np.all((yes >= 0) & (yes <= tested))
    mean = float(np.mean(yes / tested))
    return dict(mean_per_gene_yes_rate=mean, fraction_fakes_with_any_yes=float(np.mean(yes > 0)),
                threshold=.001, passed=bool(mean <= .001))


def control_cache(condition, bulk, samples, *, config=None):
    """Read only archived NO-TARGET CSC columns selected by the frozen QC."""
    import h5py
    from beacon.data import configuration, sha256
    root = configuration(config)["data_root"]
    prepared = root / f"prepared/tcell_{condition}"
    controls = pd.read_csv(prepared / "training_controls.csv.gz").set_index("barcode")
    qc = pd.read_csv(prepared / "cell_qc.csv.gz").set_index("barcode")
    expected = qc.loc[qc.passing & (qc.crispr == "NT")]
    assert controls.index.is_unique and np.array_equal(controls.index, expected.index)
    assert np.array_equal(controls.donor, expected.donor)
    assert (controls.gene == "NO-TARGET").all() and controls.passing.all()
    assert set(controls.donor) == set(DONORS)
    gene_qc = pd.read_csv(prepared / "gene_qc.csv")
    assert np.array_equal(gene_qc.loc[gene_qc.retained, "gene_id"], bulk["genes"])
    filename = "HuTcellsCRISPRaPerturbSeq_" + ("Resting" if condition == "resting" else "Re-stimulated") + ".h5Seurat"
    source = root / "raw/tcells" / filename
    digest = sha256(source)
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert digest == manifest["inputs"][str(source)]
    values, indices, indptr, libraries = [], [], [0], []
    with h5py.File(source, "r") as handle:
        symbols = handle["assays/RNA/features"].asstr()[:]
        assert np.array_equal(symbols, gene_qc.symbol)
        barcodes = pd.Index(handle["cell.names"].asstr()[:])
        assert barcodes.is_unique
        columns = barcodes.get_indexer(controls.index)
        assert np.all(columns >= 0)
        counts = handle["assays/RNA/counts"]
        assert tuple(counts.attrs["dims"]) == (len(symbols), len(barcodes))
        pointers = counts["indptr"][:]
        mapping = np.full(len(symbols), -1, dtype=int)
        mapping[gene_qc.retained.to_numpy()] = np.arange(len(bulk["genes"]))
        for column in columns:
            start, end = pointers[column:column + 2]
            raw = counts["data"][int(start):int(end)]
            assert np.isfinite(raw).all() and np.all(raw > 0) and np.all(raw == np.floor(raw))
            raw = raw.astype(np.int64)
            retained = mapping[counts["indices"][int(start):int(end)]]
            keep = retained >= 0
            values.append(raw[keep]); indices.append(retained[keep])
            indptr.append(indptr[-1] + int(keep.sum()))
            libraries.append(int(raw.sum()))
    matrix = sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), np.array(indptr)), shape=(len(controls), len(bulk["genes"])))
    libraries = np.array(libraries, dtype=np.int64)
    assert np.array_equal(libraries, controls.nCount_RNA.to_numpy())
    for donor in DONORS:
        cells = controls.donor.to_numpy() == donor
        row, = np.flatnonzero((samples.donor == donor) & (samples.gene == "NO-TARGET"))
        assert cells.sum() == samples.iloc[row].cells
        assert np.array_equal(np.asarray(matrix[cells].sum(axis=0)).ravel(), bulk["counts"][row])
        assert libraries[cells].sum() == bulk["full_rna_library_counts"][row]
        assert samples.loc[(samples.donor == donor) & (samples.gene != "NO-TARGET"), "cells"].max() < cells.sum()
    return dict(data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                shape=np.array(matrix.shape), full_rna_library_counts=libraries,
                donors=controls.donor.to_numpy(dtype=str), barcodes=controls.index.to_numpy(dtype=str),
                target_genes=bulk["genes"]), dict(path=str(source), sha256=digest)
