#!/usr/bin/env python3
"""Freeze control-only expression and TF eligibility for the K562 application."""
import json
import os
from pathlib import Path
import socket
import time
import h5py
import numpy as np
import pandas as pd
from beacon.data import configuration, sha256

ROOT=configuration()["data_root"]
BASE=ROOT


def column(group, name):
    ds=group[name]
    if 'categories' in ds.attrs:
        categories=group.file[ds.attrs['categories']].asstr()[:]
        return pd.Categorical.from_codes(ds[:], categories=categories)
    if h5py.check_string_dtype(ds.dtype):
        return ds.asstr()[:]
    return ds[:]


def main():
    started=time.perf_counter()
    output=BASE/'prepared/k562'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists(): raise FileExistsError(output)
    raw=BASE/'raw/k562/raw.h5ad'
    bulk=BASE/'raw/k562/bulk.h5ad'
    tfs=pd.read_csv(BASE/'raw/transcription_factors.csv')
    tf_ids=set(tfs.loc[tfs['Is TF?']=='Yes','Ensembl ID'])
    with h5py.File(bulk) as handle:
        group=handle['obs']
        bulk_ids=column(group,'gene_transcript')
        core_ids=set(bulk_ids[np.asarray(column(group,'core_control'),dtype=bool)])
        eligibility=pd.DataFrame({'perturbation_id':bulk_ids,
            'gene_id':[v.rsplit('_',1)[-1] for v in bulk_ids],
            'cells':column(group,'num_cells_filtered'),
            'on_target_fold_expression':column(group,'fold_expr')})
    with h5py.File(raw) as handle:
        obs,var=handle['obs'],handle['var']
        group_ids=column(obs,'gene_transcript')
        core_categories=np.array([v in core_ids for v in group_ids.categories])
        assert np.all(group_ids.codes>=0)
        core=core_categories[group_ids.codes]
        umi=column(obs,'UMI_count')
        quality=(column(obs,'core_adjusted_UMI_count')>=2000)&(column(obs,'mitopercent')<.25)
        controls=np.flatnonzero(core&quality)
        assert len(controls)>1000 and np.all(umi[controls]>0)
        gem=column(obs,'gem_group')
        rng=np.random.default_rng(42)
        selected=np.sort(np.concatenate([rng.choice(indices,min(50,len(indices)),replace=False)
            for g in np.unique(gem[controls]) for indices in [controls[gem[controls]==g]]]))
        genes=column(var,'gene_id'); symbols=np.asarray(column(var,'gene_name')).astype(str)
        assert len(set(genes))==len(genes)
        detection=np.zeros(len(genes),dtype=np.int64)
        sums=np.zeros(len(genes),dtype=np.float64)
        train=np.empty((len(selected),len(genes)),dtype=np.float32)
        # Contiguous reads avoid quadratic HDF5 fancy indexing on dispersed control cells.
        for start in range(0,len(core),8192):
            stop=min(start+8192,len(core))
            control_rows=(core&quality)[start:stop]
            if not control_rows.any(): continue
            block=handle['X'][start:stop]
            counts=block[control_rows]
            assert np.isfinite(counts).all() and counts.min()>=0
            detection+=(counts>0).sum(axis=0)
            sums+=counts.sum(axis=0,dtype=np.float64)
            lo,hi=np.searchsorted(selected,[start,stop])
            train[lo:hi]=block[selected[lo:hi]-start]
            if start%(8192*40)==0: print('Read cells',stop,'of',len(core),flush=True)
        keep=detection/len(controls)>=.01
        qc=pd.DataFrame({'gene_id':genes,'symbol':symbols,'control_detected_cells':detection,
                         'control_detection_fraction':detection/len(controls),'control_mean_counts':sums/len(controls),
                         'retained':keep,'lambert_tf':np.isin(genes,list(tf_ids))})
        qc.to_csv(output/'gene_qc.csv',index=False)
        expression=np.log1p(1e4*train[:,keep]/umi[selected,None]).T.astype(np.float32)
        assert np.isfinite(expression).all() and np.all(expression>=0)
        np.savez_compressed(output/'control_expression.npz',expression=expression,
                            genes=np.asarray(genes[keep],dtype=str),symbols=symbols[keep],
                            tf_indices=np.flatnonzero(np.isin(genes[keep],list(tf_ids))),
                            training_cell_indices=selected,all_control_cell_indices=controls)
        barcodes=column(obs,'cell_barcode')
        pd.DataFrame({'cell_index':selected,'barcode':barcodes[selected],'gem_group':gem[selected],
                      'umi_count':umi[selected],'perturbation_id':np.asarray(group_ids)[selected]}).to_csv(output/'training_controls.csv.gz',index=False)
        pd.DataFrame({'gem_group':gem[controls]}).value_counts().rename('passing_core_controls').reset_index().to_csv(output/'control_counts_by_gem.csv',index=False)
        retained=set(genes[keep])
        eligibility['in_lambert_tf_list']=eligibility.gene_id.isin(tf_ids)
        eligibility['expressed_in_controls']=eligibility.gene_id.isin(retained)
        eligibility['eligible']=(eligibility.in_lambert_tf_list & eligibility.expressed_in_controls &
                                 (eligibility.cells>=50)&(eligibility.on_target_fold_expression<=.5))
        eligibility['primary_construct']=False
        primary=eligibility.loc[eligibility.eligible].sort_values(['gene_id','cells','perturbation_id'],ascending=[True,False,True]).drop_duplicates('gene_id').index
        eligibility.loc[primary,'primary_construct']=True
        eligibility.to_csv(output/'perturbation_eligibility.csv',index=False)
        manifest={'host':socket.gethostname(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),
                  'source_cells':len(core),'measured_genes':len(genes),'passing_core_controls':len(controls),
                  'core_control_constructs':len(core_ids),'training_controls':len(selected),'gem_groups':int(len(np.unique(gem[controls]))),
                  'retained_genes':int(keep.sum()),'expressed_tfs':int(qc.loc[keep,'lambert_tf'].sum()),
                  'eligible_tf_constructs':int(eligibility.eligible.sum()),'eligible_tfs':len(primary),
                  'source_metadata':str(BASE/'raw/k562/manifest.json'),
                  'script_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(BASE/'raw/k562/protocol.md'),
                  'tf_catalogue_sha256':sha256(BASE/'raw/transcription_factors.csv'),
                  'control_expression_sha256':sha256(output/'control_expression.npz'),
                  'eligibility_sha256':sha256(output/'perturbation_eligibility.csv'),
                  'seconds':time.perf_counter()-started,
                  'normalization':'log1p(10000 * raw counts / total cell UMI_count); genes selected by >=1% detection among all quality-passing core controls',
                  'construct_rule':'For each eligible TF choose the eligible construct with the most retained cells; ties by identifier. Other eligible constructs remain in the manifest. No downstream response or model score used.',
                  'replication_note':'GEM groups are technical captures; cell counts and groups are not donor-level biological replication.'}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
