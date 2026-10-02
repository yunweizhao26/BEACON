"""Reciprocal-edge and between-regulator ranking checks from saved scores."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from figures.build import CONTEXT

def all_tf_ap(edges, y, s, tfs):
    total=0.0
    for _,g in pd.DataFrame(dict(r=edges[:,0],y=y,s=s)).groupby("r"):
        if g.y.any():total+=ap(g.y,g.s)
    return total/len(tfs)

def median_between(P, rows_keep):
    X=np.asarray(P[rows_keep],dtype=float)
    R=np.apply_along_axis(rankdata,1,X);R=(R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
    C=R@R.T/R.shape[1];iu=np.triu_indices(len(C),1)
    return float(np.median(C[iu])),float(np.quantile(C[iu],.05)),len(C)

def run(bundle,out):
    rows=[]
    for ds,name in CONTEXT.items():
        for seed in (42,14,100):
            e,=[e for e in bundle.manifest["experiments"] if e["suite"]=="fixed_pools" and e["condition"]["dataset"]==ds and e["condition"]["split_seed"]==seed and e["condition"]["coverage"]==.8 and e["condition"]["ratio"]==5 and e["condition"]["corruption"]==0 and e["condition"]["control"]=="beacon" and not e["condition"]["snn_weight"]]
            c,=[c for c in bundle.manifest["comparators"] if c["suite"]=="fixed_pools" and c.get("dataset")==ds and c["split_seed"]==seed and c["coverage"]==.8 and c["method"]=="gnnlink"]
            z=bundle.arrays(e["reference"]["split"]);p=bundle.arrays(e["reference"]["predictions"]);g=bundle.arrays(c["artifacts"]["predictions"])
            key={tuple(e):i for i,e in enumerate(g["edges"])};idx=np.array([key[tuple(e)] for e in p["edges"]])
            edges=p["edges"];y=p["labels"].astype(bool);prior=z["supplied_positive_edges"];pset=set(map(tuple,prior))
            indeg=np.bincount(prior[:,1],minlength=len(z["genes"]));rev=np.array([(b,a) in pset for a,b in edges])
            topo=p["degree_logistic"];t01=(topo-topo.min())/(topo.max()-topo.min()+1e-12)
            scores={"BEACON":p["beacon"],"GNNLink":g["gnnlink"][idx],"Topology control":topo,"Prior target in-degree":indeg[edges[:,1]].astype(float),"Reverse-edge rule":rev+.5*t01}
            keep=~(y&rev);tfs=z["tf_indices"]
            for m,s in scores.items():
                rows.append(dict(context=name,split=seed,method=m,share_reverse=float(rev[y].mean()),ap_all=ap(y,s),ap_without=ap(y[keep],s[keep]),alltf_all=all_tf_ap(edges,y,s,tfs),alltf_without=all_tf_ap(edges[keep],y[keep],s[keep],tfs)))
    pd.DataFrame(rows).to_csv(out/"reciprocal_edges.csv",index=False)
    rows=[]
    for context in ("k562","tcell_resting","tcell_stimulated"):
        e,=[e for e in bundle.manifest["experiments"] if e["suite"]=="external" and e["condition"]["context"]==context and e["condition"]["control"]=="beacon"]
        c,=[c for c in bundle.manifest["comparators"] if c["suite"]=="external" and c.get("context")==context and c["method"]=="gnnlink"]
        for method,record in (("BEACON",e["reference"]["beacon"]),("GNNLink",c["artifacts"]["scores"])):
            P=bundle.array(record);med,lo,n=median_between(P,np.arange(P.shape[0]))
            rows.append(dict(context=context,method=method,regulators=n,median_spearman=med,p05=lo))
    pd.DataFrame(rows).to_csv(out/"regulator_similarity.csv",index=False)
