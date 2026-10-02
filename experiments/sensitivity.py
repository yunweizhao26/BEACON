"""Factor/PCA/scGPT inputs, optimization seeds, and inducing counts."""
import numpy as np
import torch
from beacon.model import Training
from evaluation.metrics import metrics
from experiments.fixed_pools import predict_pairs

def inducing(bundle, experiment, *, device="cuda"):
    c = experiment["condition"]
    source = next(e for e in bundle.manifest["experiments"] if e["suite"] == "fixed_pools" and
        e["condition"]["dataset"] == c["dataset"] and e["condition"]["split_seed"] == c["split_seed"] and
        e["condition"]["coverage"] == .8 and e["condition"]["control"] == "beacon" and not e["condition"]["snn_weight"])
    reference = source["reference"]
    split = bundle.arrays(reference["split"])
    embeddings = bundle.array(reference["embeddings"])
    e_star = int(bundle.json(reference["training_log"])["encoder"]["e_star"])
    training = Training()
    cached = training.internal_split(split["train"])
    cached["e_star"] = e_star
    cached["selection"] = {"best_epoch": e_star}
    fits = {}
    for count in (128, 256, 500, 1000):
        model, likelihood, _, _ = training.fit_gp(embeddings, split["train"], torch.device(device),
            inducing_points_num=count, num_epochs=50, batch_size=1024, run_seed=42)
        model.eval()
        likelihood.eval()
        fit = {"gp_epochs": int(training.training_log["fits"][-1]["stop_epoch"]), "gp_stage": training.training_log["fits"][-1]["stage"]}
        for pool, key in (("validation", "valid"), ("test", "test")):
            edges = np.argwhere(split[key] != -1)
            scores, _ = predict_pairs(embeddings, model, likelihood, edges, device=device, variance=False)
            fit[pool] = metrics(split[key][tuple(edges.T)], scores, edges[:, 0])
        fits[str(count)] = fit
    return fits

def scgpt_features(bundle, genes, *, universe, symbols=None, pca=False):
    metadata = bundle.manifest["scgpt"]
    factors = bundle.arrays(metadata["principal_features" if pca else "factor_features"])
    keys = ["gene_key", "ensembl_id", "symbol"]
    union = bundle.frame(metadata["genes"], sep="\t", keep_default_na=False, dtype={k:str for k in keys})
    mapping = bundle.frame(metadata["universes"][universe], sep="\t", keep_default_na=False, dtype={k:str for k in keys})
    manifest = bundle.json(metadata["manifest"])
    if manifest["embedding_source"] != "scGPT":
        raise ValueError("Wrong embedding producer")
    genes = np.asarray(genes, str)
    symbols = genes if symbols is None else np.asarray(symbols, str)
    assert len(genes) == len(mapping)
    ens = np.char.startswith(genes, "ENSG")
    renamed = {(r["source_ensembl_id"], r["census_ensembl_id"]) for r in manifest.get("renamed_ensembl_ids", [])}
    assert all(g.split(".")[0] == m or (g.split(".")[0],m) in renamed for g,m in zip(genes[ens],mapping.ensembl_id.to_numpy()[ens]))
    assert np.array_equal(symbols[~ens], mapping.symbol.to_numpy()[~ens])
    selected = union.set_index("gene_key").index.get_indexer(mapping.gene_key)
    assert (selected >= 0).all() and np.array_equal(union.iloc[selected][keys], mapping[keys])
    if pca:
        # The original runner reads the exported decimal table then casts float32.
        exported = bundle.frame(metadata["exported"], sep="\t", keep_default_na=False)
        assert np.array_equal(exported[keys], union[keys])
        result = exported[[f"f{i:02d}" for i in range(64)]].to_numpy(np.float32)[selected]
    else:
        assert np.array_equal(factors["genes"], union.gene_key)
        result = factors["features"][selected]
    if result.shape != (len(genes), 64) or not np.isfinite(result).all():
        raise ValueError("Invalid aligned scGPT features")
    return result

if __name__ == "__main__":
    from experiments.run import main
    main()
