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
  scgpt/genes.tsv
  sensitivity/factor_sensitivity/endpoints.csv
  tables/key_numbers.csv
```

This is an excerpt. The bundle contains 44 BEELINE panels, two SERGIO inputs, fixed splits, prepared external inputs, calibrated labels, model and comparator predictions, RPE1 FA/RNG caches, scGPT features and sensitivity metrics.

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

For publication, archive the contents with `manifest.json` and category folders at the archive root. Configure its URL and SHA-256, then run `python -m data.download`. The downloader checks the archive checksum, rejects links/escaping paths, verifies every listed file and rejects missing or extra files before installing into an absent destination. Publication URL and checksum remain pending.
