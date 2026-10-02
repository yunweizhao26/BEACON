"""Preserve the complete SERGIO truth and noisy count matrix."""
import argparse
import json
import numpy as np
import pandas as pd
from beacon.data import configuration, sha256
def main():
    p=argparse.ArgumentParser();p.add_argument("--density",choices=("sparse","dense"),required=True);args=p.parse_args()
    root=configuration()["data_root"];source=root/"raw/sergio"/args.density
    truth=pd.read_csv(source/"network.csv",header=None,names=["source","target"]).astype(int)
    metadata=json.loads((source/"manifest.json").read_text())
    assert len(truth)==int(metadata["stats"]["edges"]) and not truth.duplicated().any()
    values=np.load(source/"expression.npy",allow_pickle=False)
    assert values.shape==(300,2700) and np.all(values>=0)
    genes=np.array([f"Gene_{i}" for i in range(len(values))])
    out=root/"prepared/sergio"/args.density;out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(values,index=genes).to_csv(out/"expression.csv")
    pd.DataFrame({"Gene1":genes[truth.source],"Gene2":genes[truth.target]}).to_csv(out/"network.csv",index=False)
    pd.DataFrame({"TF":genes}).to_csv(out/"transcription_factors.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(generator=metadata,source_sha256={str(p):sha256(p) for p in source.iterdir() if p.is_file()}),indent=2)+"\n")
if __name__=="__main__":main()
