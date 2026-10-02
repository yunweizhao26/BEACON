"""Map the frozen source-restricted prior; sample five unlabeled pairs per positive."""
import argparse
import numpy as np
import pandas as pd
from beacon.data import configuration

def training(expression, genes, symbols, tf_indices, prior):
    n=len(genes); assert expression.shape[0]==n
    symbol_counts=pd.Series(symbols).value_counts()
    unique_map={name:i for i,name in enumerate(symbols) if symbol_counts[name]==1}
    prior=prior.copy()
    prior['source_index']=prior.source_genesymbol.map(unique_map)
    prior['target_index']=prior.target_genesymbol.map(unique_map)
    valid=prior.source_index.notna()&prior.target_index.notna()&prior.source_index.isin(tf_indices)&(prior.source_index!=prior.target_index)
    prior['retained_for_training']=valid
    positive=prior.loc[valid,['source_index','target_index']].astype(int).drop_duplicates().to_numpy()
    assert len(positive)>=20
    allowed=np.zeros((n,n),dtype=bool); allowed[tf_indices]=True; np.fill_diagonal(allowed,False)
    train=np.full((n,n),-1,dtype=np.int8); train[tuple(positive.T)]=1
    unknown=np.flatnonzero(allowed&(train!=1))
    chosen=np.random.default_rng(42).choice(unknown,size=5*len(positive),replace=False)
    train.flat[chosen]=0
    del unknown,allowed
    discovery=train[tf_indices]==-1
    discovery[np.arange(len(tf_indices)),tf_indices]=False
    return dict(train=train,genes=genes,symbols=symbols,tf_indices=tf_indices),discovery,prior

def main():
    p=argparse.ArgumentParser();p.add_argument("context",choices=("k562","rpe1","tcell_resting","tcell_stimulated"));a=p.parse_args()
    root=configuration()["data_root"];prepared=root/"prepared"/a.context
    with np.load(prepared/"control_expression.npz",allow_pickle=False) as z:data={k:z[k] for k in z.files}
    prior=pd.read_csv(root/"raw/prior/allowed_prior.tsv",sep="\t")
    train,mask,audit=training(data["expression"],data["genes"],data["symbols"],data["tf_indices"],prior)
    with (prepared/"training.npz").open("xb") as f:np.savez_compressed(f,**train)
    with (prepared/"discovery_mask.npz").open("xb") as f:np.savez_compressed(f,values=mask)
    audit.to_csv(prepared/"prior_mapping_audit.csv",index=False)
if __name__=="__main__":main()
