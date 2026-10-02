#!/usr/bin/env python3
"""Prepare donor-resolved control expression and pseudobulks for primary T cells."""
import json
import os
from pathlib import Path
import re
import socket
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from beacon.data import configuration, sha256

ROOT=configuration()["data_root"];BASE=ROOT;SOURCE=BASE/'raw/tcells'

def column(group,name):
    obj=group[name]
    if isinstance(obj,h5py.Group):
        levels=obj['levels'].asstr()[:];values=obj['values'][:]
        assert np.all((values>=1)&(values<=len(levels))),name
        return levels[values-1]  # h5Seurat stores R factor codes, which start at one.
    return obj.asstr()[:] if h5py.check_string_dtype(obj.dtype) else obj[:]

def main():
    started=time.perf_counter();protocol=BASE/'raw/tcells/protocol.json';assert protocol.exists()
    donor_map=pd.read_csv(SOURCE/'souporcell_match_vcf_res/donor_calls.txt',sep='\t').set_index('Well_ID')
    genotype=[];genotype_paths=[]
    for path in sorted((SOURCE/'souporcell').glob('*/clusters.tsv')):
        name=path.parent.name;well=int(re.search(r'well([1-4])',name).group(1))+(0 if name.startswith('nostim') else 4)
        frame=pd.read_csv(path,sep='\t',dtype={'assignment':str});frame['barcode']=frame.barcode.str.replace(r'-\d+$',f'-{well}',regex=True)
        mapping={str(k):str(donor_map.loc[well,f'Souporcell_call{k}_DonorA_or_B']).strip() for k in [0,1]}
        frame['resolved_donor']=frame.assignment.map(mapping).map({'A':'Donor2','B':'Donor1'});genotype.append(frame[['barcode','status','resolved_donor']]);genotype_paths.append(path)
    genotype=pd.concat(genotype).set_index('barcode');assert genotype.index.is_unique
    guides=pd.read_csv(SOURCE/'cellranger-guidecalls-aggregated-unfiltered.txt',sep='\t').set_index('cell_barcode');assert guides.index.is_unique
    feature_source=BASE/'raw/tcells/features.tsv.gz';features=pd.read_csv(feature_source,sep='\t',header=None,names=['gene_id','symbol','feature_type']);features=features[features.feature_type=='Gene Expression']
    features['gene_id']=features.gene_id.str.split('.').str[0]
    unambiguous=features[~features.symbol.duplicated(False)&~features.gene_id.duplicated(False)].set_index('symbol').gene_id.to_dict()
    tf_path=BASE/'raw/transcription_factors.csv';tf=pd.read_csv(tf_path);tf_ids=set(tf.loc[tf['Is TF?']=='Yes','Ensembl ID'])
    for condition,filename in [('resting','HuTcellsCRISPRaPerturbSeq_Resting.h5Seurat'),('stimulated','HuTcellsCRISPRaPerturbSeq_Re-stimulated.h5Seurat')]:
        out=BASE/f'prepared/tcell_{condition}';out.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():raise FileExistsError(out)
        path=SOURCE/filename
        with h5py.File(path,'r') as h:
            symbols=np.asarray(h['assays/RNA/features'].asstr()[:],dtype=str);barcodes=h['cell.names'].asstr()[:]
            meta=pd.DataFrame({k:column(h['meta.data'],k) for k in ['gene','guide_id','crispr','donor','condition','nFeature_RNA','nCount_RNA','percent.mt','CD4.or.CD8']},index=barcodes)
            counts=h['assays/RNA/counts'];x=sparse.csc_matrix((counts['data'][:],counts['indices'][:],counts['indptr'][:]),shape=tuple(counts.attrs['dims']))
        assert x.shape==(len(symbols),len(meta)) and np.isfinite(x.data).all() and np.all(x.data>0) and np.all(x.data==np.floor(x.data))
        assert len(np.unique(symbols))==len(symbols)
        umi=np.asarray(x.sum(axis=0)).ravel();detected=np.diff(x.indptr);mito=np.asarray(x[np.char.startswith(symbols,'MT-')].sum(axis=0)).ravel()/umi
        assert np.allclose(umi,meta.nCount_RNA) and np.array_equal(detected,meta.nFeature_RNA)
        gt=genotype.reindex(barcodes);calls=guides.reindex(barcodes);assert gt.status.notna().all() and calls.num_features.notna().all()
        assert np.array_equal(calls.feature_call.to_numpy(),meta.guide_id.to_numpy())
        assert np.array_equal(meta.guide_id.str.rsplit('-',n=1).str[0].to_numpy(),meta.gene.to_numpy())
        known=(gt.status=='singlet')&meta.donor.isin(['Donor1','Donor2']);assert np.array_equal(gt.loc[known,'resolved_donor'],meta.loc[known,'donor'])
        guide_umi=pd.to_numeric(calls.num_umis,errors='coerce')
        passed=known.to_numpy()&(detected>400)&(detected<6000)&(mito<.25)&(calls.num_features.to_numpy()==1)&(guide_umi.to_numpy()>=5)
        meta['souporcell_status']=gt.status.to_numpy();meta['guide_umis']=guide_umi.to_numpy();meta['passing']=passed;meta['technical_well']=meta.index.str.rsplit('-',n=1).str[-1]
        meta.to_csv(out/'cell_qc.csv.gz',index_label='barcode');all_cells=len(meta)
        meta=meta.loc[passed].copy();x=x[:,passed].tocsr();umi=umi[passed];nt=meta.crispr.to_numpy()=='NT'
        assert np.all(meta.loc[nt,'gene']=='NO-TARGET') and all((meta.loc[nt,'donor']==donor).sum()>=50 for donor in ['Donor1','Donor2'])
        detected_control=np.asarray((x[:,nt]>0).sum(axis=1)).ravel();mapped=np.array([unambiguous.get(g,'') for g in symbols]);keep=(detected_control/nt.sum()>=.01)&(mapped!='')
        assert len(np.unique(mapped[keep]))==int(keep.sum())
        qc=pd.DataFrame({'symbol':symbols,'gene_id':mapped,'control_detected_cells':detected_control,'control_detection_fraction':detected_control/nt.sum(),'unambiguous_mapping':mapped!='','retained':keep,'lambert_tf':np.isin(mapped,list(tf_ids))});qc.to_csv(out/'gene_qc.csv',index=False)
        expression=np.log1p(1e4*x[keep][:,nt].toarray()/umi[nt]).astype(np.float32);genes=mapped[keep];retained_symbols=symbols[keep]
        tf_indices=np.flatnonzero(np.isin(genes,list(tf_ids)))
        np.savez_compressed(out/'control_expression.npz',expression=expression,genes=genes,symbols=retained_symbols,tf_indices=tf_indices,training_barcodes=meta.index.to_numpy(dtype=str)[nt])
        meta.loc[nt].to_csv(out/'training_controls.csv.gz',index_label='barcode')
        meta.groupby(['donor','gene','guide_id','technical_well'],observed=True).size().rename('cells').reset_index().to_csv(out/'guide_donor_cell_counts.csv',index=False)
        groups=meta[['donor','gene']].drop_duplicates().sort_values(['donor','gene']).reset_index(drop=True);lookup={(r.donor,r.gene):i for i,r in groups.iterrows()};codes=np.array([lookup[v] for v in meta[['donor','gene']].itertuples(index=False,name=None)])
        aggregate=sparse.csr_matrix((np.ones(len(meta)),(np.arange(len(meta)),codes)),shape=(len(meta),len(groups)))
        bulk=(x@aggregate).toarray();sizes=np.bincount(codes,minlength=len(groups));groups['cells']=sizes
        means=(x.multiply(1e4/umi)@aggregate).toarray()/sizes
        by_symbol={g:i for i,g in enumerate(symbols)};rows=[]
        for target in sorted(set(meta.gene)-{'NO-TARGET'}):
            index=by_symbol.get(target);gid=unambiguous.get(target,'');record={'symbol':target,'gene_id':gid,'catalogue_tf':gid in tf_ids,'expressed_in_controls':gid in set(genes)}
            for donor in ['Donor1','Donor2']:
                a=lookup.get((donor,target));c=lookup[(donor,'NO-TARGET')];fold=None
                if a is not None and index is not None and means[index,c]>0:fold=float(means[index,a]/means[index,c])
                record[donor+'_cells']=int(sizes[a]) if a is not None else 0;record[donor+'_on_target_fold']=fold
            record['eligible']=record['catalogue_tf'] and record['expressed_in_controls'] and all(record[d+'_cells']>=50 and record[d+'_on_target_fold'] is not None and record[d+'_on_target_fold']>=1.5 for d in ['Donor1','Donor2'])
            rows.append(record)
        eligibility=pd.DataFrame(rows);eligibility.to_csv(out/'perturbation_eligibility.csv',index=False);targets=eligibility.loc[eligibility.eligible,'symbol'].tolist()
        selected=groups.gene.isin(['NO-TARGET',*targets]).to_numpy();pseudobulks=bulk[keep][:,selected].T.astype(np.int64);sample_meta=groups.loc[selected].reset_index(drop=True)
        sample_meta['sample']=sample_meta.donor+'__'+sample_meta.gene;sample_meta.to_csv(out/'pseudobulk_samples.csv',index=False)
        # Full-RNA library sums support descriptive donor effect directions; DESeq2 estimates its own size factors.
        np.savez_compressed(out/'pseudobulk_counts.npz',counts=pseudobulks,genes=genes,symbols=retained_symbols,samples=sample_meta['sample'].to_numpy(dtype=str),full_rna_library_counts=bulk[:,selected].sum(axis=0))
        source_paths=[path,feature_source,tf_path,protocol,SOURCE/'cellranger-guidecalls-aggregated-unfiltered.txt',SOURCE/'souporcell_match_vcf_res/donor_calls.txt',*genotype_paths,Path(__file__)]
        manifest={'condition':condition,'author_archived_cells':all_cells,'passing_cells':len(meta),'training_control_cells':int(nt.sum()),'control_cells_by_donor':meta.loc[nt,'donor'].value_counts().to_dict(),'measured_genes':len(symbols),'retained_genes':len(genes),'expressed_tfs':len(tf_indices),'perturbed_targets_in_archive':len(rows),'eligible_tfs':len(targets),'eligible_tf_symbols':targets,'donors':['Donor1','Donor2'],'pseudobulk_samples':len(sample_meta),'source_qc_note':'Author code additionally required guide singlets with >=5 guide UMIs, >=3 expressing cells per RNA gene, and removed targets having <100 cells in either condition before archiving. We additionally require genotype singlets and the frozen per-donor eligibility.','control_expression_sha256':sha256(out/'control_expression.npz'),'pseudobulk_sha256':sha256(out/'pseudobulk_counts.npz'),'inputs':{str(p):sha256(p) for p in source_paths},'host':socket.gethostname(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'elapsed_seconds_from_both_condition_preparation_start':time.perf_counter()-started}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'},indent=2),flush=True)

if __name__=='__main__':main()
