# Comparator training and evidence

Saved predictions are in the data bundle. The adapters in [train.py](train.py) use the pinned upstream implementations; upstream source is not vendored. [manifest.json](manifest.json) preserves checkout commits, package versions, complete recorded environments and the scRegNet configuration.

GNNLink, GENELink, GCLink, RegGAIN and scRegNet select training length with BEACON's internal validation rule. Hold out the same stratified 10% of labeled training pairs with seed 42, drawing positives first. Remove them from fitting, choose the earliest epoch with the highest held-out average precision (AP), and refit on all labeled training pairs. Benchmark validation/test pairs and assay labels never select epochs.

| Method | Check interval | Maximum / grid | Patience | Selected fixed-pool epochs |
|---|---:|---|---:|---|
| GNNLink | 10 | 10,000 | 300 epochs | 10–3,270 |
| GENELink | 1 | 100 | 20 epochs | 1–35 |
| GCLink contrastive stage | 1 | 100 | 20 epochs | 2–36 |
| RegGAIN | Each grid value | 50, 100, 200, 500, 1,000, 2,000 | Entire grid | 50–2,000 |
| scRegNet | 5 | 300 | 50 epochs | 5, 15, 25 for split seeds 42, 14, 100 |

Every stochastic comparator is fitted with optimization seeds 42, 14 and 100 (`--seed`), and the paper scores the per-pair mean of the three fits' scores, as for BEACON. The seed replaces 42 in each method's own seeding; the held-out pairs that select the training length keep seed 42. TRRUST fits use the five task seeds. GNNLink z-scores each gene across cells, as the upstream loader does, and scRegNet predicts without dropout.

GCLink retains 20 supervised pretraining epochs before its selected contrastive stage. Fixed-pool RegGAIN restarts its released pipeline at each grid value; external and temporal RegGAIN use snapshots of one selection trajectory. The recorded fixed-pool/GNNLink adapters fall back to the upstream epoch count if fewer than 10 held-out positives exist; no reported fixed-pool fit triggered this fallback.

In temporal scRegNet fits, retain the best held-out-AP selection checkpoint. Use it if the refit's score standard deviation is below 1e-4. This guard was added after observing constant refits and uses no evaluation labels. It applied to the 2005 priors in K562, resting T cells and RPE1 and the 2010 prior in K562. The K562 2005 selection checkpoint also remained constant. Fixed-pool scRegNet uses the refit.

## Run a fit

Install the method's recorded environment and check out its pinned upstream commit. Pass the checkout root as `--source`; the GENELink adapter resolves its `Code/` subdirectory and reshapes scalar labels to the model's single-column output. Run one method per process because upstream module names overlap.

```bash
python -m comparators.train fixed_pools --method gnnlink --source ../GNNLink \
  --dataset 1501 --coverage .8 --split-seed 42 --device cpu --out results/gnnlink
python -m comparators.train temporal --method scregnet --source ../scRegNet \
  --context k562 --cutoff 2005 --embeddings ../embeddings/k562.npz \
  --device cpu --out results/scregnet_k562
```

Use `--config` to select the data bundle. The runner also accepts `expression`, `external` and `trrust` settings; see `--help`. Output directories must be absent. Results include aligned scores, AP history, selected epoch, device, input manifest hash and the upstream commit. Re-running on a different device does not promise bitwise score identity.

scRegNet requires gene-aligned NPZ arrays named `genes` and `embeddings`. A repacked bundle can register these under `comparator_training.geneformer_embeddings`, keyed by `mHSC-E` or assay context. Otherwise pass `--embeddings`. Saved scores alone cannot reconstruct these embeddings.

## Upstream configurations and environments

| Comparator, upstream and pinned commit | Configuration and adapter behavior | Recorded environment |
|---|---|---|
| [GNNLink](https://github.com/sdesignates/GNNLink)<br>`3ecba9c8edec55aee4c6a351ad9a8602534ac2ab` | TensorFlow compatibility graph/session, seed reset, validation-selected epochs; supplied positives build adjacency plus identity, supplied unlabeled pairs provide the negative mask; read the complete logits matrix and select frozen eligible edges. Raw expression and the separately reported FA input are retained. | Python 3.10.20, TensorFlow CPU 2.20.0, NumPy 2.2.6, Torch 2.11.0; Torch is also imported by upstream utilities. |
| [GCLink](https://github.com/Yoyiming/GCLink)<br>`e0f3582b7f142367d255d47c279121a3a6595a0d` | Released attention model; 20 pretraining epochs and validation-selected contrastive epochs, batch 256, learning rate .003, edge removal .2, contrastive weight .5, temperature .2. Frozen positive/unlabeled pools replace the upstream split. | Core environment: Torch 2.11.0, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2. |
| [scRegulate](https://github.com/YDaiLab/scRegulate)<br>`de8efa5e24a2297c22636ba682aad69281bf630c` | Maximum 10,000 epochs; batch floor(.85 × cells), at least one; freeze 4,500 epochs; learning rate .0001; alpha max 1 and scale .0025; patience 4,500; min targets/TFs zero. Import the training modules through a package stub. Zero-target ULM estimates become zero, undefined P values become one. | Core environment plus AnnData .11.4, Scanpy 1.11.5; complete receipt in manifest. |
| [RegGAIN](https://github.com/SDU-Math-SunLab/RegGAIN)<br>`177018c13fc0fa3a47483d69f55af1806f5cc417` | Validation-selected grid epochs, learning rate .001, k=50, adjacency powers [0,1,2]; first widths [80,80,10], subsequent widths 40/40/5 and 16/16/2; pos=10; edge weights .6/.3/.3/.3 and node weights .5/.2/.2/.2. Primary scores retain the released orientation; the alternative orientation is separate evidence. | Core plus torch-geometric 2.8.0.post1, AnnData and Scanpy. |
| [GENELink](https://github.com/zpliulab/GENELink)<br>`4eb816af27f5fe0cb26f9230286401801b251129` | Validation-selected epochs, batch 256; hidden widths 128/64/32, output 16, two three-head attention layers, alpha .2, dot decoder, concatenation; Adam .003, StepLR gamma .99. Supplied training pairs and frozen eligible pool; no refitted test split. | Core Torch environment. |
| [GENIE3](https://github.com/aertslab/arboreto) | Arboreto random forests with eligible TFs, per-target optimization seed and upstream parameters. Full-network scores reused across fixed test pools; absent edges score zero. | Arboreto .1.6; NumPy 2.2.6, scikit-learn 1.7.2. Original distributed scheduler dependencies are not needed to rescore saved predictions. |
| [GRNBoost2](https://github.com/aertslab/arboreto) | Arboreto SGBM parameters, optimization seed per target, self predictor removed, importances multiplied by tree count. Joblib target scheduling replaces Dask. The [paired monitor](monitor.py) measures before/after loss on the **same** out-of-bag mask and applies the 25-round mean-improvement rule. Standard and paired-monitor outputs are identified separately. | Arboreto .1.6, scikit-learn 1.7.2, Joblib 1.5.3, NumPy 2.2.6. |
| [Inferelator 3.0](https://github.com/flatironinstitute/inferelator)<br>`9a0e0a8d41a5bef5abf414775ac014f63f180278` | Single-cell TFA/BBSR, no gold standard, supplied prior and TF list, Joblib scheduling. 50 bootstraps, with the optimization seed as bootstrap seed. Target-by-regulator confidence orientation is explicitly mapped to ordered pairs; missing pairs score zero. | Package 0.6.3 at the recorded source commit, plus core NumPy/SciPy/Pandas/scikit-learn environment. “3.0” is the manuscript method name; the checkout pin is authoritative. |
| [scRegNet](https://github.com/sindhura-cs/scRegNet)<br>`30d0215efd99c40ceaceb37d161fc0a86a236e0b` | Reported mHSC-E fixed pool and temporal priors. Geneformer penultimate token features averaged over detected cells; released graph convolution network with binary cross-entropy. Batch 133, hidden 256, RMSprop learning rate .0012056391735792479, weight decay 4.6205759922095423e-7, dropout .5095606629916417. Held-out AP every five epochs up to 300, patience 50 epochs; refit with the temporal collapse guard above; dropout zero for prediction. | Core Torch; the Transformers package version and checkpoint download location were not traced in the inspected receipts; saved embeddings/predictions remain the reproducible evidence. |

External adapters preserve the source restrictions of the curated prior and exclude all supplied training pairs and self pairs when reporting discovery metrics. Comparator configuration does not authorize selecting hyperparameters from test labels. No install or fit is required for the release evaluator.

## Temporal holdout

The temporal-holdout scores in the data bundle (`temporal/`) come from the configurations above, applied to the priors of edges first reported up to 2000, 2005 or 2010 in K562, resting and re-stimulated primary T cells and RPE1. All five comparators trained on labeled pairs use the internal AP selection and full-pair refit described above. GRNBoost2 and GENIE3 use arboreto's upstream per-target regressions and are fitted once per context and seed, because they do not use the prior. Inferelator uses 50 bootstraps. scRegNet uses Geneformer embeddings of each context's control cells, with each cell limited to its 4,094 top-ranked genes, computed with Transformers 4.57.6. RegGAIN ran on CPUs, and GENELink and GCLink ran on CPUs or GPUs with unchanged configurations; each run record (`*_run.json`) states the device. The scRegNet guard and remaining K562 2005 constant-score case are described above.
