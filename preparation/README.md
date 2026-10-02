# Preparing public inputs

These scripts require the raw public data listed in [sources.md](../data/sources.md). They are documented preparation code, separate from the prediction identity test. The processed bundle is the identity input. Raw archives and original count matrices are not bundled, and no preparation was executed during this extraction.

All paths resolve from `data_root` in `config.toml`. Work on compute nodes. Preparation writes under `data_root/prepared`; retain its manifests. Preserve source gene ordering, dtype and the frozen source restrictions rather than replacing public-source snapshots with current API results.

| Step | Required input below `data_root` | Entry point |
|---|---|---|
| K562 controls, QC and eligibility | `raw/k562/raw.h5ad`, `bulk.h5ad`; `raw/transcription_factors.csv` | `python -m preparation.k562` |
| T-cell controls and donor pseudobulks | `raw/tcells/`: the resting/re-stimulated h5Seurat objects, guide calls, souporcell clusters/donor calls, `features.tsv.gz`, `protocol.json`; TF catalogue | `python -m preparation.tcells` |
| RPE1 controls and metadata | `raw/rpe1/raw.h5ad`, `bulk.h5ad`, `protocol.md`, `metadata/gene_identity_metadata.tsv`; TF catalogue | `python -m preparation.rpe1` |
| Curated-prior mapping and unlabeled sampling | Prepared control expression plus `raw/prior/allowed_prior.tsv` | `python -m preparation.prior k562` (also RPE1 and each T-cell condition) |
| ENCODE binding | Frozen `raw/binding/experiments.json`, `annotation.gtf.gz`, `checksums.txt`; prepared K562 training axes | `python -m preparation.binding` |
| TRRUST pair manifest | `raw/trrust/expression.csv`, `relationships.tsv`, `prior.tsv` | `python -m preparation.trrust` |
| SERGIO | `raw/sergio/{sparse,dense}/`: headerless `network.csv`, `expression.npy`, generator `manifest.json` | `python -m preparation.sergio --density sparse` (then dense) |

ENCODE preparation retains frozen metadata selection rules, checks selected peak MD5s, and downloads only selected peak files. The checksum table must identify the descriptive local `annotation.gtf.gz`. RPE1 source-file checks use the recorded source manifest. SERGIO preserves the source noisy counts without adding a log transformation.

`labels.py` contains the K562/RPE1 Anderson–Darling implementation, within-GEM normalization, complete per-perturbation BH, matched fake-knockdown sampling and calibration. `prepare_controls(raw, bulk, destination, line)` writes the control caches and metadata. `real_response(...)` recomputes complete test-family P values and effects; `fake_response(...)` uses independent `SeedSequence([42, task])` streams, matches each GEM count, removes fake cells from their reference and recomputes normalization. Use exactly 200 fake records. Template selection uses `default_rng(42).choice(sorted(unique_constructs), 200, replace=True)` over the recorded construct union. Pass the complete test family before subsetting to the frozen evaluation schema. `response_labels(...)` rejects failed calibration; primary support is BH q < .05 without an effect floor, while the sensitivity floor is the .999 linear quantile of finite null effects. The frozen tested mask and discovery/self exclusions remain distinct.

Build the AD probability grid with R 4.5.3 and kSamples 1.2.12:

```bash
Rscript preparation/ad_probability.R /path/to/data_root/prepared/ad_probability.csv
```

This is the recorded reconstruction of the published kSamples probability extension; the authors' original interpolation grid was not supplied. Keep the generated provenance record. The released K562 AD/BH table supplies the independent normalization/correction validation reference.

`tcell_labels.py` contains the DESeq2 design `~donor+perturbation`, target-versus-control contrasts, global BH over the complete finite family, and donor-consistent directions. Install `preparation/requirements.txt` into the pinned environment. `joint_fit` takes the prepared pseudobulk counts, sample metadata, genes and targets; `response_labels` requires the passed calibration receipt. `control_cache(condition, bulk, samples)` builds a CSR control-cell record from the source h5Seurat columns and frozen QC, checks gene/barcode/donor axes and reproduces donor pseudobulk totals. `calibrate` takes that record and rebuilds all 200 joint fake fits, keeping fake cells out of their remaining donor controls and preserving full-RNA library counts. The data-floor population includes finite Cook-filtered effects, as in the recorded producer.

The AD `response_labels` call requires explicit `work`, `family_genes` and raw `genes` axes. It applies BH on the complete family, then maps constructs and covered genes into the frozen schema; untested genes outside the family retain NaN probabilities. The label kernels expose array/metadata APIs rather than recreating the research scheduler and publication workflow. Reconstructing the raw input/schema orchestration still requires the frozen construct/tested-mask metadata and the recorded source-only protocols. That preparation path has not been validated end to end and is not claimed as a raw-data identity test.

Unresolved provenance: the complete raw-to-processed BEELINE recipe, original sampled-split invocation, initial ENCODE metadata query, curated-prior writer, original TRRUST download URL and Geneformer checkpoint location were not recovered. Prepared inputs and their records are retained. Raw TRRUST expression extraction and upstream SERGIO simulation should use the source recipes/metadata associated with those accessions; the release preparation starts from their processed expression/generator outputs.
