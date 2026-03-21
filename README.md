BEACON
Run from `beacon/`:

```bash
pip install -r requirements.txt
python main.py
```

Current default reproduction settings in `main.py`:

```text
embedding_types = ['fa']
use_snn_options = [True]
model_types = ['standard']
output_dims = [16]
num_epochs_list = [50]
train_ratios = [0.8]
dataset_ids = [1701]
batch_sizes = [32]
learning_rates = [1e-3]
negative_ratios = [5]
temperatures = [1]
neg_sampling_options = ['random']
```

Paths are set up to run from `beacon/` and use local datasets in `data/raws` and `data/processed`. Intermediate splited data (train/valid/test) are stored at `data/splits`.

Run outputs now use plain files (no `run_artifacts.py` helper):

- `logs/.../version_*/run_manifest.json`: run configuration and metadata
- `logs/.../version_*/metrics_summary.json`: AUROC/AUPRC summary
- `logs/.../version_*/gp_reports/`: train/valid/test GP plots and reports
- `logs/.../sampler_overview.tsv`: one-line summary per run
