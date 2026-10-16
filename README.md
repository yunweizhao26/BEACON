# BEACON

BEACON predicts regulator–target links from expression and a partial regulatory prior. It combines factor-analysis features, a shared MLP encoder and an additive sparse variational Gaussian process.

## Install

Use Python 3.10.20 and the pinned CUDA environment for numerical reproduction.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Quickstart

From the release folder, on an allocated GPU node, fit the hESC fixed-pool setting:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m experiments.fixed_pools --dataset 1501 --out results/quickstart
```

Output directories must be absent. Experiment settings are listed in the data bundle’s `manifest.json`; `--case` selects a setting. Runners cover fixed pools, sampled pairs, expression controls, SERGIO, K562, T cells, RPE1, influenza-stimulated TRRUST and sensitivity analyses.

## Data

Set `data_root`, `bundle_url` and `bundle_sha256` in [config.toml](config.toml), then download:

```bash
python -m data.download
```

The Zenodo deposit, [doi:10.5281/zenodo.23108369](https://doi.org/10.5281/zenodo.23108369), is not yet published. The URL and checksum in `config.toml` await the revised bundle; do not treat them as a public reproduction artifact. The bundle is a [plain directory tree](data/README.md) of NPZ, CSV, TSV and JSON files with one top-level manifest. Arrays preserve dtype, values and memory order. See [public sources](data/sources.md), [raw preparation](preparation/README.md) and [comparator training](comparators/README.md).

## Tables and figures

The revised analyses require the epoch-standard comparator scores and their recomputed plotting records. Repacking the data bundle is separate from this source patch. See [required bundle updates](data/README.md#revised-paper-bundle).

Figures use the lighter paper palette directly. Reverse-edge checks exclude every test pair whose reverse was revealed, including unlabeled pairs. Fixed-pool specificity ranks each regulator's full test pool with average ties, computes a leave-one-out target consensus before exclusion, and evaluates both APs on the retained pairs. Its identical-order simulation band is empirical; unequal target pools do not guarantee a ratio of one. External specificity retains its aggregate definition and adds each regulator's own prior-support strata.

Recompute key numbers and response metrics from saved predictions:

```bash
python -m evaluation.report --responses --out results/evaluation
python -m evaluation.matched --out results/matched
python -m evaluation.specificity --out results/specificity
```

One command per main-text output:

```bash
python -m figures.run tables --only reference_dimensions --out results/reference_dimensions
python -m figures.run main --only beacon_overview --out results/overview
python -m figures.run main --only prior_completion --out results/prior_completion
python -m figures.run tables --only ablations --out results/ablations
python -m figures.run main --only experimental_validation --out results/experimental_validation
python -m figures.run main --only cross_context_validation --out results/cross_context
python -m figures.run main --only specificity_regimes --out results/specificity_regimes
```

Temporal holdout (Figure 4 and Supplementary Tables S16 and S17). The priors hold the CollecTRI edges first reported up to 2000, 2005 or 2010, and the saved scores of every method are in the data bundle:

```bash
python -m preparation.temporal --check
python -m evaluation.temporal --out results/temporal
python -m figures.temporal --source results/temporal --out results/temporal_figure
python -m experiments.temporal --context k562 --cutoff 2005 --out results/temporal_beacon_k562_2005
```

The first command rebuilds every prior, candidate mask and label set and compares them with the saved arrays. The last command refits BEACON on one prior.

Supplementary figures:

```bash
python -m figures.run supplementary --only runtime_scaling --out results/runtime_scaling
python -m figures.run supplementary --only precision_by_evidence_type --out results/precision_by_evidence_type
python -m figures.run supplementary --only precision_by_dataset --out results/precision_by_dataset
python -m figures.run supplementary --only score_distribution --out results/score_distribution
python -m figures.run supplementary --only score_distribution_by_evidence_type --out results/score_distribution_by_evidence_type
python -m figures.run supplementary --only auroc --out results/auroc
python -m figures.run supplementary --only auprc --out results/auprc
python -m figures.run supplementary --only network_complexity --out results/network_complexity
python -m figures.run supplementary --only benchmark_shortcut_audit --out results/benchmark_shortcut_audit
```

Supplementary tables and review checks:

```bash
python -m figures.run tables --only prior_edge_counts --out results/prior_edge_counts
python -m figures.run tables --only simulator_truth --out results/simulator_truth
python -m figures.run tables --only trrust_comparison --out results/trrust_comparison
python -m figures.run tables --only scregnet_fixed_pool --out results/scregnet_fixed_pool
python -m figures.run tables --only expression_controls --out results/expression_controls
python -m figures.run tables --only gp_inducing_sensitivity --out results/gp_inducing_sensitivity
python -m figures.run tables --only optimization_seeds --out results/optimization_seeds
python -m figures.run tables --only tcell_eligibility --out results/tcell_eligibility
python -m figures.run sensitivity --only factor_components --out results/factor_components
python -m figures.run checks --out results/review_checks
python -m figures.run tables --only reciprocal_edges --out results/table_s8
python -m figures.run tables --only specificity_fixed --out results/table_s14
python -m figures.run tables --only specificity_external --out results/table_s15
```

Factor sensitivity retains the prespecified 64-component setting.

## Tests

Small CPU checks:

```bash
python -m tests.test_bundle
python -m tests.test_equivalence --unit
python -m tests.test_evaluation --unit
python -m tests.test_specificity
python -m tests.test_comparators
```

On L40S with the pinned environment and thread settings above:

```bash
python -m tests.test_equivalence --out results/equivalence
python -m tests.test_evaluation --out results/evaluation_test
```

The identity test checks 16 prediction cases bitwise and inducing metrics separately. The evaluation test checks 1,176 in-scope key numbers at absolute tolerance 1e-9. The historical L40S test results predate this patch; the revised source and repacked bundle require fresh validation. Optional [Slurm wrappers](tests/equivalence.sbatch) accept a configurable Python and require your account and partition.

## License

MIT. See [LICENSE](LICENSE).
