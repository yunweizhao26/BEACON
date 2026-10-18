import argparse
import json
import numpy as np
import pandas as pd
from beacon.data import configuration


def evaluation_records(genes, trrust, min_targets=5):
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    gene_set = set(genes)
    trrust = trrust[trrust["source"].isin(gene_set) & trrust["target"].isin(gene_set) & (trrust["source"] != trrust["target"])]
    target_counts = trrust.groupby("source")["target"].nunique()
    eligible_sources = target_counts[target_counts >= min_targets].index.tolist()
    if not eligible_sources:
        raise SystemExit("no eligible regulators")
    trrust_pairs = set(map(tuple, trrust[["source", "target"]].itertuples(index=False, name=None)))
    evaluation = [dict(source=source, target=target, source_index=gene_to_index[source], target_index=gene_to_index[target],
                       analysis_role="supported" if (source, target) in trrust_pairs else "low_effect",
                       orthogonal_edge_support=int((source, target) in trrust_pairs),
                       llcb_effect=0.0, llcb_absolute_effect=0.0, llcb_pip=0.0, llcb_lfsr=0.0)
                  for source in sorted(eligible_sources) for target in genes if target != source]
    if not any(record["analysis_role"] == "supported" for record in evaluation):
        raise SystemExit("no positive evaluation edges")
    return evaluation


def build(genes, trrust, prior, candidate_tfs, *, min_targets=5, negative_ratio=5, sampling_seed=42):
    genes = list(map(str, genes))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    evaluation = evaluation_records(genes, trrust, min_targets)
    masked = {(r["source_index"], r["target_index"]) for r in evaluation}
    masked |= {(target, source) for source, target in masked}
    prior = prior[prior["source_genesymbol"].isin(gene_to_index) & prior["target_genesymbol"].isin(gene_to_index)]
    pairs = {(gene_to_index[s], gene_to_index[t]) for s, t in prior[["source_genesymbol", "target_genesymbol"]].itertuples(index=False, name=None)}
    train_positive = np.asarray(sorted(p for p in pairs if p[0] != p[1] and p not in masked), dtype=np.int64).reshape(-1, 2)
    sources = np.asarray(sorted(gene_to_index[g] for g in set(map(str, candidate_tfs)) if g in gene_to_index), dtype=np.int64)
    if not len(sources):
        raise SystemExit("no candidate regulators among the expression genes")
    eligible = np.zeros((len(genes), len(genes)), dtype=bool)
    eligible[sources] = True
    np.fill_diagonal(eligible, False)
    eligible[tuple(train_positive.T)] = False
    if masked:
        eligible[tuple(np.asarray(sorted(masked)).T)] = False
    candidates = np.flatnonzero(eligible)
    size = int(len(train_positive) * negative_ratio)
    if size > len(candidates):
        raise SystemExit("too few candidate unlabeled pairs")
    chosen = np.random.default_rng(sampling_seed).choice(candidates, size=size, replace=False)
    train_unlabeled = np.column_stack(np.unravel_index(chosen, eligible.shape)).astype(np.int64)
    train_positive_edge_ids = [f"edge_{source:05d}_{target:05d}" for source, target in train_positive.tolist()]
    return dict(genes=genes, train_positive_edge_ids=train_positive_edge_ids, train_unlabeled=train_unlabeled.tolist(),
                evaluation=evaluation, candidate_regulators=len(sources)), train_positive


def build_any_source(genes, trrust, prior, *, min_targets=5, negative_ratio=5, sampling_seed=42):
    genes = list(map(str, genes))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    gene_set = set(genes)

    trrust = trrust.copy()
    trrust = trrust[
        trrust["source"].isin(gene_set)
        & trrust["target"].isin(gene_set)
        & (trrust["source"] != trrust["target"])
    ]
    target_counts = trrust.groupby("source")["target"].nunique()
    eligible_sources = target_counts[target_counts >= min_targets].index.tolist()
    if not eligible_sources:
        raise SystemExit("no eligible regulators")
    trrust_pairs = set(map(tuple, trrust[["source", "target"]].itertuples(index=False, name=None)))

    evaluation = []
    for source in sorted(eligible_sources):
        for target in genes:
            if target == source:
                continue
            evaluation.append({
                "source": source,
                "target": target,
                "source_index": gene_to_index[source],
                "target_index": gene_to_index[target],
                "analysis_role": "supported" if (source, target) in trrust_pairs else "low_effect",
                "orthogonal_edge_support": int((source, target) in trrust_pairs),
                "llcb_effect": 0.0,
                "llcb_absolute_effect": 0.0,
                "llcb_pip": 0.0,
                "llcb_lfsr": 0.0,
            })
    if not any(record["analysis_role"] == "supported" for record in evaluation):
        raise SystemExit("no positive evaluation edges")

    evaluation_pairs = {(record["source"], record["target"]) for record in evaluation}
    evaluation_pairs |= {(target, source) for source, target in evaluation_pairs}

    prior = prior.copy()
    prior = prior[
        prior["source_genesymbol"].isin(gene_set)
        & prior["target_genesymbol"].isin(gene_set)
    ]
    train_pairs = [
        (source, target)
        for source, target in prior[["source_genesymbol", "target_genesymbol"]].itertuples(
            index=False, name=None
        )
        if (source, target) not in evaluation_pairs
    ]
    train_positive_edge_ids = [
        f"edge_{gene_to_index[source]:05d}_{gene_to_index[target]:05d}"
        for source, target in sorted(train_pairs)
    ]
    train_positive = np.asarray(
        [(gene_to_index[source], gene_to_index[target]) for source, target in sorted(train_pairs)],
        dtype=np.int64,
    )

    rng = np.random.default_rng(sampling_seed)
    train_positive_set = set(map(tuple, train_positive.tolist()))
    evaluation_index_pairs = {
        (record["source_index"], record["target_index"]) for record in evaluation
    }
    evaluation_index_pairs |= {(target, source) for source, target in evaluation_index_pairs}
    negative_eligibles = [
        (source, target)
        for source in range(len(genes))
        for target in range(len(genes))
        if (source, target) not in train_positive_set
        and (source, target) not in evaluation_index_pairs
    ]
    num_negatives = int(len(train_positive) * negative_ratio)
    sampled = rng.choice(
        len(negative_eligibles),
        size=min(num_negatives, len(negative_eligibles)),
        replace=False,
    )
    train_unlabeled = np.asarray(
        [negative_eligibles[int(index)] for index in sampled], dtype=np.int64
    )

    return dict(genes=genes, train_positive_edge_ids=train_positive_edge_ids,
        train_unlabeled=train_unlabeled.tolist(), evaluation=evaluation), train_positive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--any-source",action="store_true",help="Rebuild the published manifest (unlabeled pairs from every gene pair)")
    args=parser.parse_args()
    root=configuration()["data_root"]
    genes=pd.read_csv(root/"raw/trrust/expression.csv",index_col=0).index.astype(str).tolist()
    trrust=pd.read_csv(root/"raw/trrust/relationships.tsv",sep="\t",header=None,names=["source","target","effect","pmid"],dtype=str)
    prior=pd.read_csv(root/"raw/trrust/prior.tsv",sep="\t",dtype=str)
    if args.any_source:
        record,positive=build_any_source(genes,trrust,prior)
    else:
        catalogue=pd.read_csv(root/"raw/transcription_factors.csv")
        record,positive=build(genes,trrust,prior,catalogue.loc[catalogue["Is TF?"]=="Yes","HGNC symbol"].dropna().astype(str))
    out=root/"prepared/trrust";out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"training.npz",positive=positive,unlabeled=np.array(record["train_unlabeled"],dtype=np.int64))
    (out/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
if __name__=="__main__":main()
