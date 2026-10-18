# BEACON

BEACON predicts regulator–target links from expression and a partial regulatory prior. It combines factor-analysis features, a shared MLP encoder and an additive sparse variational Gaussian process.

## Install

Use Python 3.10.20 and the pinned CUDA environment for numerical reproduction.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Quickstart

BEACON is the mean of three fits with optimization seeds 42, 14 and 100: each candidate pair's score is the mean of the three fits' predictive probabilities. From the release folder, on an allocated GPU node, fit and average the three seeds of the hESC fixed-pool setting:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m experiments.ensemble fixed_pools --dataset 1501 --coverage .8 --split-seed 42 --out results/quickstart
python -m experiments.ensemble external --context k562 --parallel 3 --out results/k562
```

`results/quickstart/seed_<seed>/` holds each fit and `results/quickstart/ensemble/` the per-pair mean. Output directories must be absent. Every single fit is a setting of the data bundle's `manifest.json`, and `python -m experiments.run --case <setting>` reruns one. Runners cover fixed pools, sampled pairs, expression controls, SERGIO, K562, T cells, RPE1, influenza-infected TRRUST (five task seeds), the temporal holdout and sensitivity analyses. Comparators are refitted with `python -m comparators.train ... --seed 14` (see [comparator training](comparators/README.md)).

## Data

`bundle_parts` and `bundle_sha256` in [config.toml](config.toml) point to the archived data bundle. Set `data_root` if needed, then download:

```bash
python -m data.download
```

The data bundle is archived at Zenodo, [doi:10.5281/zenodo.23108369](https://doi.org/10.5281/zenodo.23108369), as `beacon_data.tar.gz` split into 11 consecutive parts. The downloader joins the parts in order and checks the SHA-256 of the joined archive. To join them by hand, run `cat beacon_data.tar.gz.part* > beacon_data.tar.gz`. The bundle is a [plain directory tree](data/README.md) of NPZ, CSV, TSV and JSON files with one top-level manifest. Arrays preserve dtype, values and memory order. See [public sources](data/sources.md), [raw preparation](preparation/README.md) and [comparator training](comparators/README.md). [encode_k562_binding_accessions.tsv](data/encode_k562_binding_accessions.tsv) lists the 72 ENCODE experiments and peak files behind the K562 promoter-binding labels.

## Tables and figures

Every figure, table body and key number is computed from the bundle's per-seed score files by the commands below; each stochastic method enters as its seed ensemble (`evaluation.seeds`).

Figures use the lighter paper palette directly. Reverse-edge checks exclude every test pair whose reverse was a known edge, including unlabeled pairs. Fixed-pool specificity ranks each regulator's full test pool with average ties, computes a leave-one-out target consensus before exclusion, and evaluates both APs on the retained pairs. Its identical-order simulation band is empirical; unequal target pools do not guarantee a ratio of one. External specificity compares each regulator's ranking with a leave-one-out consensus of the other evaluated regulators, reports each regulator's own prior-support strata, and uses the mean over 100 random score matrices as its random reference. Mean per-regulator AP is compared with the exact expected AP of a random ranking of each regulator's targets; pooled AP is compared with prevalence. Targets tied at a top-K boundary count fractionally in every top-K endpoint. The T-cell comparisons add exact sign-flip tests over the seven regulators.

Recompute key numbers and response metrics from saved predictions:

```bash
python -m evaluation.report --responses --out results/evaluation
python -m evaluation.matched --out results/matched
python -m evaluation.specificity --out results/specificity
python -m evaluation.robustness --out results/robustness
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

Temporal holdout (Figure 4 and Supplementary Tables S16 and S17). The priors hold the CollecTRI edges first reported up to 2000, 2005 or 2010, and the saved scores of every method and seed are in the data bundle:

```bash
python -m preparation.temporal --check
python -m evaluation.temporal --out results/temporal
python -m figures.temporal --source results/temporal --out results/temporal_figure
python -m experiments.ensemble temporal --context k562 --cutoff 2005 --device cpu --out results/temporal_beacon_k562_2005
```

The first command rebuilds every prior, candidate mask and label set and compares them with the saved arrays. `evaluation.temporal` also writes `seeds.csv`, the single fits of BEACON and scRegNet and the scRegNet fit with permuted Geneformer embeddings, which `figures.temporal` lists in Supplementary Table S16. The temporal BEACON fits run on CPU.

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

Supplementary tables and robustness checks:

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
python -m figures.run checks --out results/robustness_checks
python -m figures.run tables --only reciprocal_edges --out results/reciprocal_edges
python -m figures.run tables --only specificity_fixed --out results/specificity_fixed
python -m figures.run tables --only specificity_external --out results/specificity_external
python -m figures.run tables --only robustness_reverse_blocked --out results/robustness_reverse_blocked
python -m figures.run tables --only robustness_ten_splits --out results/robustness_ten_splits
python -m figures.run tables --only robustness_references --out results/robustness_references
```

Factor sensitivity retains the prespecified 64-component setting. Table S13 compares single fits with optimization
seed 42 for BEACON at each number of components and for every comparator:

```bash
python -m evaluation.factor_components --out results/factor_components
python -m figures.run sensitivity --only factor_components --out results/factor_components_table
```

Comparator training epochs selected on the internal validation pairs, from the run records in the bundle, and the
single-kernel design check (Table S9, seed 42), and the runtime breakdown of the three-seed K562 and T-cell fits:

```bash
python -m evaluation.epochs --out results/epochs
python -m figures.run tables --only design_selection --out results/design_selection
python -m figures.run tables --out results/tables   # includes tab:runtime_stages
```

## Tests

Small CPU checks:

```bash
python -m tests.test_bundle
python -m tests.test_equivalence --unit
python -m tests.test_evaluation --unit
python -m tests.test_specificity
python -m tests.test_comparators
python -m tests.test_robustness
python -m tests.test_trrust --unit
python -m tests.test_factor_components
python -m tests.test_epochs
```

On L40S with the pinned environment and thread settings above:

```bash
python -m tests.test_equivalence --out results/equivalence
python -m tests.test_evaluation --out results/evaluation_test
```

The identity test refits the bundle's identity cases, single fits with one optimization seed, and compares their predictions bitwise; inducing metrics are checked separately. The evaluation test recomputes every key number from the bundle and compares it with `tables/key_numbers.csv` at absolute tolerance 1e-9; it also runs on CPU. Optional [Slurm wrappers](tests/equivalence.sbatch) accept a configurable Python and require your account and partition.

## License

MIT. See [LICENSE](LICENSE).
