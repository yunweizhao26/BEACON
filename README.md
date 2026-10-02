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

Output directories must be absent. Experiment settings are listed in the data bundle’s `manifest.json`; `--case` selects a setting. Runners cover fixed pools, sampled pairs, expression controls, SERGIO, K562, T cells, RPE1, TRRUST and sensitivity analyses.

## Data

Set `data_root`, `bundle_url` and `bundle_sha256` in [config.toml](config.toml), then download:

```bash
python -m data.download
```

The publication URL and checksum are pending. The bundle is a [plain directory tree](data/README.md) of NPZ, CSV, TSV and JSON files with one top-level manifest. Arrays preserve dtype, values and memory order. See [public sources](data/sources.md), [raw preparation](preparation/README.md) and [comparator configurations](comparators/README.md).

## Tables and figures

Recompute key numbers and response metrics from saved predictions:

```bash
python -m evaluation.report --responses --out results/evaluation
```

One command per main-text output:

```bash
python -m figures.run tables --only reference_dimensions --out results/reference_dimensions
python -m figures.run main --only beacon_overview --out results/overview
python -m figures.run main --only prior_completion --out results/prior_completion
python -m figures.run main --only component_evidence --out results/components
python -m figures.run main --only experimental_validation --out results/experimental_validation
python -m figures.run main --only cross_context_validation --out results/cross_context
```

Supplementary figures:

```bash
python -m figures.run supplementary --only runtime_scaling --out results/runtime_scaling
python -m figures.run supplementary --only completion_sensitivity --out results/completion_sensitivity
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
python -m figures.run sensitivity --only input_features --out results/input_features
python -m figures.run checks --out results/review_checks
```

Factor sensitivity retains the prespecified 64-component setting.

## Tests

Small CPU checks:

```bash
python -m tests.test_bundle
python -m tests.test_equivalence --unit
python -m tests.test_evaluation --unit
```

On L40S with the pinned environment and thread settings above:

```bash
python -m tests.test_equivalence --out results/equivalence
python -m tests.test_evaluation --out results/evaluation_test
```

The identity test checks 18 prediction cases bitwise and inducing metrics separately. The evaluation test checks 1,528 in-scope key numbers at absolute tolerance 1e-9. Full numerical validation remains pending. Optional [Slurm wrappers](tests/equivalence.sbatch) accept a configurable Python and require your account and partition.

## License

MIT. See [LICENSE](LICENSE).
