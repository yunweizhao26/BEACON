# Data bundle

Set `data_root` in `config.toml`; relative paths resolve from that file. Data stay outside the repository. Every bundled file can be opened directly with NumPy, pandas or a text editor.

```text
beacon_data/
  manifest.json
  DATA_LICENSES.md
  beeline/string/hESC_500/
    expression.npz
    network.csv
    transcription_factors.csv
    sampled/train.npz
    sampled/test.npz
  predictions/fixed_pools/hESC/coverage_80/split_42/
    seed_42/predictions.npz
    seed_14/predictions.npz
    seed_100/predictions.npz
  predictions/k562/beacon.npz                    the seed-42 K562 fit; seed_14/ and seed_100/ hold the other two
  predictions/rpe1/seed_42/beacon.npz
  comparators/fixed_pools/gnnlink/hESC/coverage_80/split_42/seed_42.npz
  comparators/k562/gnnlink/seed_14.npz
  temporal/cutoff_2005/k562/beacon_seed_100.npz
  trrust/beacon/seed_47.json
  robustness/...                                 reverse-blocked training, ten splits and additional references
  labels/k562.npz
  labels/k562/encode_binding_accessions.tsv
  prepared/rpe1/factor_cache/features.npz
  raw/prior/dorothea.tsv
  timing/three_seeds/k562.json
  tables/key_numbers.csv
```

This is an excerpt. The bundle contains 44 BEELINE panels, two SERGIO inputs, fixed splits, prepared external inputs, calibrated labels, the fits of BEACON and of every comparator, and the tables computed from them by the release code.

**Seeds.** BEACON and every stochastic comparator are stored as one file per fit with optimization seed 42, 14 or 100; the release code scores the per-pair mean of the three fits (`evaluation.seeds`), and TRRUST uses its five task seeds. Manifest experiments carry `condition["seed"]`, and comparator entries carry `seed`; a deterministic score (degree, popularity, expression and detection baselines) has one entry without a seed. The sampled-pair benchmark, SERGIO, the hardware check, the inducing-point refits and the expression controls are single fits with seed 42 (the expression controls also hold seed-14 and seed-100 fits of the real input). `manifest["seeds"]` lists the seed sets.

**Tables.** Everything under `tables/` and `plotting/` except the sampled-pair, factor-component and T-cell eligibility records is written by the release code from the bundle's score files (`evaluation.report --responses`, `evaluation.specificity`, `evaluation.temporal`, `evaluation.robustness`, `figures.run checks`); `tests.test_evaluation` recomputes the key numbers and compares every row at absolute tolerance 1e-9.

Paths describe scientific conditions: context, reference, panel size, method, coverage, split and optimization seed. Each manifest entry records its relative `path`, `conditions`, `original_sha256` (the digest of the source file the asset was packed from), output `sha256`, sizes and array shapes/dtypes. Shared references resolve to one file. There is one top-level manifest. Experimental records use plain `setting` identifiers. Key numbers contain `section`, `item`, `context`, `metric` and `value`. `documents` lists the license file and the ENCODE accession list, `timing` the runtime records and `raw` the inputs of `preparation.prior_filter`.

Arrays use `np.savez_compressed`. Conversion checks compare shape, dtype, value bytes and C/Fortran memory layout before writing. Expression CSVs retain their pandas-parsed values plus gene/cell axes; the model converts to float32 at the original loader step. Small CSV/JSON/TSV files remain ordinary text. Compressed TSV inputs are decompressed before packing.

```python
from pathlib import Path
import numpy as np

root = Path("../beacon_data")
with np.load(root / "labels/k562.npz", allow_pickle=False) as labels:
    print(labels.files)
```

`beacon.data.Bundle` resolves manifest references and verifies checksums on reads. `Bundle.verify()` checks every listed file. Evaluation receipts identify the top-level manifest by SHA-256.

`data.pack.pack(files, root, destination, metadata=...)` packs a supplied list of trusted local files. Each item supplies `source`, `path`, `category`, `kind` and optional conditions. The returned plan includes source locations; the bundle manifest contains only public file metadata. An optional transform callback prepares file contents before writing. Use an absent destination outside the source tree. The packer validates paths and checksums, preserves array bytes and layout, and writes the manifest last.

For publication, archive the contents with `manifest.json` and category folders at the archive root. Configure its URL, or the URLs of its parts in order, and its SHA-256, then run `python -m data.download`. The downloader checks the archive checksum, rejects links/escaping paths, verifies every listed file and rejects missing or extra files before installing into an absent destination. Zenodo record [10.5281/zenodo.23108369](https://doi.org/10.5281/zenodo.23108369) holds the archive as `beacon_data.tar.gz` in 11 consecutive parts, and `config.toml` lists their URLs and the SHA-256 of the joined archive.

## Contents relative to the earlier release bundle

- Every stochastic method has its seed-14 and seed-100 fits next to the seed-42 fit; GNNLink scores use the per-gene z-score of the upstream loader and scRegNet scores the dropout-free prediction pass.
- TRRUST uses the corrected sampling of `preparation.trrust` (unique known edges, unlabeled pairs from catalogue TFs, evaluation pairs and reverses masked) and evaluates the 24 catalogue regulators.
- Inferelator 3.0 uses 50 bootstraps in every setting.
- The RPE1 fits seed the optimization and compute the factor-analysis features from the prepared input; the frozen features remain for the temporal RPE1 fits.
- The unreported fixed-pool fits with the SNN loss or a corrupted prior and the derived research tables are not included; the reciprocal-edge and regulator-similarity checks are under `tables/robustness_checks/`.
- Host names, Slurm job IDs and research paths are removed from every record.
- `DATA_LICENSES.md` gives the license and citation of every redistributed third-party source.

Table S15 distinguishes evaluated regulators from those with a response-positive tested target. Prior bins are 0, 1–2, 3–5, 6–10 and 11+ edges. Each regulator's consensus averages the ranks of the other evaluated regulators and is computed over the full evaluated population before assigning bins. Table S15 and Figure 3 are recomputed from full test-pool scores; their consensus cannot be reconstructed from already-excluded pairs.
