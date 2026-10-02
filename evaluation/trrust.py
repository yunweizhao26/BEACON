"""Same-regulator TRRUST endpoints with the original paired bootstrap order."""
import numpy as np
import pandas as pd
from evaluation.metrics import binary_metrics, paired

def records(bundle, indices):
    rows=[]
    for index in indices:
        value=bundle.json(index)
        if value["context"]!="flu":continue
        for row in value["records"]:
            rows.append(dict(method_seed=int(value["method_seed"]),source=row["source"],target=row["target"],analysis_role=row["analysis_role"],score=float(row["probability"])))
    return pd.DataFrame(rows)

def exclude_overlap(frames, original, min_supported=5):
    original_pairs=set(original[["source_genesymbol","target_genesymbol"]].itertuples(index=False,name=None))
    filtered={}
    for method,frame in frames.items():
        overlap=np.fromiter(((s,t) in original_pairs for s,t in frame[["source","target"]].itertuples(index=False,name=None)),dtype=bool,count=len(frame))
        drop=(frame.analysis_role=="supported").to_numpy() & overlap
        filtered[method]=frame.loc[~drop].copy()
    beacon=filtered["BEACON"]
    counts=beacon.loc[beacon.analysis_role=="supported",["source","target"]].drop_duplicates().groupby("source").target.nunique()
    eligible=sorted(counts[counts>=min_supported].index)
    return {method:frame[frame.source.isin(eligible)].copy() for method,frame in filtered.items()}

def bootstrap_difference(beacon, baseline, iterations=10000, seed=42):
    sources=sorted(set(beacon)&set(baseline))
    differences=np.asarray([beacon[s]-baseline[s] for s in sources])
    rng=np.random.default_rng(seed)
    boot=np.asarray([float(rng.choice(differences,len(differences),replace=True).mean()) for _ in range(iterations)])
    signs=rng.choice(np.asarray([-1.,1.]),size=(iterations,len(differences)))
    null=(signs*differences).mean(axis=1)
    p=(1+int(np.sum(null>=differences.mean())))/(iterations+1)
    return float(differences.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),float(p)

def evaluate(bundle,out,*,paired_monitor=True):
    manifest=bundle.manifest["trrust"]
    indices=[e["reference"]["record"] for e in bundle.manifest["experiments"] if e["suite"]=="trrust"]
    frames={"BEACON":records(bundle,indices)}
    for key,name in (("gnnlink","GNNLink"),("genelink","GENELink"),("genie3","GENIE3"),("grnboost2","GRNBoost2"),("inferelator","Inferelator")):
        selected=manifest["paired_grnboost2"] if key=="grnboost2" and paired_monitor else manifest["comparators"][key]
        frames[name]=records(bundle,selected)
    identity=["method_seed","source","target","analysis_role"]
    reference=frames["BEACON"].set_index(identity).index
    if reference.has_duplicates:raise ValueError("Duplicate TRRUST pairs")
    for name,frame in frames.items():
        assert set(frame.method_seed)=={42,14,100,38,47}
        assert not frame.set_index(identity).index.has_duplicates
        assert set(frame.set_index(identity).index)==set(reference),name
    prior=bundle.frame(manifest["prior"],dtype=str).set_index("edge_id").loc[manifest["train_positive_edge_ids"]]
    for name,degrees,column in (("Prior target in-degree",prior.target.value_counts(),"target"),("Prior regulator out-degree",prior.source.value_counts(),"source")):
        frame=frames["BEACON"].copy();frame["score"]=frame[column].map(degrees).fillna(0);frames[name]=frame
    original=bundle.frame(manifest["unmasked_prior"],sep="\t",dtype=str)
    rows,comparisons,pooled=[],[],[]
    for evaluation in ("all","overlap_removed"):
        current=frames if evaluation=="all" else exclude_overlap(frames,original)
        for name,frame in current.items():
            for seed,part in frame.groupby("method_seed"):
                pooled.append(dict(evaluation=evaluation,method=name,seed=seed,**binary_metrics((part.analysis_role=="supported").to_numpy(),part.score.to_numpy())))
            for (seed,tf),part in frame.groupby(["method_seed","source"]):
                rows.append(dict(evaluation=evaluation,method=name,seed=seed,tf=tf,**binary_metrics((part.analysis_role=="supported").to_numpy(),part.score.to_numpy())))
        section=pd.DataFrame(rows).query("evaluation == @evaluation")
        for metric in ("average_precision","auprc_trapezoid","auroc","top100_supported_fraction"):
            wide=section.groupby(["tf","method"])[metric].mean().unstack("method")
            for method in wide.columns:
                if method!="BEACON":comparisons.append(dict(evaluation=evaluation,comparator=method,metric=metric,**paired(wide.BEACON-wide[method])))
    out.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(rows);frame.to_csv(out/"per_tf_metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(out/"pooled_metrics.csv",index=False)
    frame.groupby(["evaluation","method"])[["average_precision","auprc_trapezoid","auroc","top100_supported_fraction"]].mean().reset_index().to_csv(out/"ranking.csv",index=False)
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv",index=False)
