# Data bundle

Set `data_root` in `config.toml`; relative paths resolve from that file. Data stay outside the repository. Every bundled file can be opened directly with NumPy, pandas or a text editor.

```text
beacon_data/
  manifest.json
  beeline/string/hESC_500/
    expression.npz
    network.csv
    transcription_factors.csv
    sampled/train.npz
    sampled/test.npz
  predictions/fixed_pools/hESC/coverage_80/split_42/seed_42/
    predictions.npz
    validation_predictions.npz
    training_log.json
  predictions/k562/beacon.npz
  comparators/fixed_pools/gnnlink/hESC/coverage_80/split_42/predictions.npz
  labels/k562.npz
  labels/k562/calibration.json
  prepared/k562/control_expression.npz
  prepared/rpe1/factor_cache/
    features.npz
    rng.npz
    python_rng.json
  sensitivity/factor_sensitivity/endpoints.csv
  tables/key_numbers.csv
```

This is an excerpt. The bundle contains 44 BEELINE panels, two SERGIO inputs, fixed splits, prepared external inputs, calibrated labels, model and comparator predictions, RPE1 FA/RNG caches and sensitivity metrics.

Paths describe scientific conditions: context, reference, panel size, method, coverage, corruption, ratio, split and optimization seed. Each manifest entry records its relative `path`, `conditions`, `original_sha256`, output `sha256`, sizes and array shapes/dtypes. Shared references resolve to one file. There is one top-level manifest. Experimental records use plain `setting` identifiers. Key numbers contain `section`, `item`, `context`, `metric` and `value`.

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

For publication, archive the contents with `manifest.json` and category folders at the archive root. Configure its URL and SHA-256, then run `python -m data.download`. The downloader checks the archive checksum, rejects links/escaping paths, verifies every listed file and rejects missing or extra files before installing into an absent destination. Zenodo record [10.5281/zenodo.23108369](https://doi.org/10.5281/zenodo.23108369) is not yet published. Its DOI remains reserved; the configured archive checksum must be refreshed after repacking.

## Revised paper bundle

The reader deliberately uses ordinary manifest records; it does not search for local research overlays. Before publication:

- Replace the GNNLink, GENELink, GCLink, RegGAIN and scRegNet score fields for the epoch-standard fixed-pool, expression-control, external, temporal and TRRUST settings. Preserve pair order, labels, axes, dtype and shape. Fixed-pool RegGAIN uses the existing `test_reggain`/`valid_reggain` schema. Matrix scores remain NPZ `values` arrays. TRRUST replacements change only `records[].probability`.
- Replace comparator-dependent plotting/table records, including pool metrics, matched-target concordance, external response summaries, TRRUST summaries, reciprocal checks and key numbers. Replacing raw scores alone leaves cached figures stale. Recompute reciprocal and specificity analyses with the revised source. Keep the sampled-pair benchmark and unavailable alternative-orientation fields explicitly identified as retained evidence.
- Add the saved Geneformer embedding NPZs for mHSC-E and the four external contexts, preserving gene order and the unmapped-gene zero vectors. Register asset IDs under `comparator_training.geneformer_embeddings` with keys `mHSC-E`, `k562`, `tcell_resting`, `tcell_stimulated` and `rpe1`. Each NPZ contains `genes` and `embeddings`; include checkpoint/dictionary provenance and upstream training receipts.
- Update every changed asset checksum, size and array metadata, retain shared record aliases, and refresh the archive checksum only after the reproduction checks pass.

Table S15 distinguishes evaluated regulators from those with a response-positive tested target. Prior bins are 0, 1–2, 3–5, 6–10 and 11+ edges. The consensus uses the full evaluated population before assigning bins. Table S15 and Figure 3 are recomputed from full test-pool scores; their consensus cannot be reconstructed from already-excluded pairs.
