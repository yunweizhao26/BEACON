#!/usr/bin/env python3
"""Freeze source-only ENCODE eligibility and derive promoter-overlap labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import requests
from beacon.data import configuration, sha256

BASE=configuration()["data_root"]

def overlaps(peaks, starts, ends):
    """Any one-base overlap between half-open query and peak intervals."""
    if not len(peaks): return np.zeros(len(starts),dtype=bool)
    peaks=np.asarray(peaks); peaks=peaks[np.argsort(peaks[:,0],kind='stable')]
    ix=np.searchsorted(peaks[:,0],ends,side='left')-1
    return (ix>=0)&(np.maximum.accumulate(peaks[:,1])[np.maximum(ix,0)]>starts)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        assert overlaps([[10,20],[5,8]],[8,9,19,20,4],[10,11,20,21,6]).tolist()==[False,True,True,False,True]
        print('Half-open promoter overlap checks passed');return
    out=BASE/'prepared/binding';out.mkdir(parents=True,exist_ok=False)
    protocol={'assembly':'GRCh38','annotation':'GENCODE v44 basic, published GTF and MD5SUMS',
      'transcript_rule':'MANE_Select when present; otherwise longest summed exon length in basic annotation; transcript ID lexicographic tie-break.',
      'promoter_rule':'Zero-based transcription boundary is GTF start-1 on + and GTF end on -. Primary + interval [boundary-1000,boundary+100), - interval [boundary-100,boundary+1000); sensitivity boundary +/-2000. Any >=1 bp overlap.',
      'experiment_rule':'Released TF ChIP-seq K562; expressed Lambert TF; untreated and no genetic modifications; at least two biological replicates in conservative IDR peak file; matched controls; no ERROR or NOT_COMPLIANT audits and no identical-biosample biological-replicate audit. Warnings retained.',
      'selection_rule':'One experiment per TF, latest release date then accession; one conservative GRCh38 IDR file, latest creation date then accession. No peak counts, overlaps, or model scores used in selection.',
      'outcome_note':'Promoter binding support, not direct regulation or effect direction. Missing annotation or excluded assay is missing evidence, not a negative. All training pairs and self-pairs remain excluded at evaluation.'}
    protocol_path=out/'protocol_before_overlap.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==protocol
    else: protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    train=np.load(BASE/'prepared/k562/training.npz');genes=train['genes'].astype(str);tfs=train['tf_indices'];tf_ids=set(genes[tfs])
    rows=json.loads((BASE/'raw/binding/experiments.json').read_text())['@graph'];eligible=[];audit=[]
    for r in rows:
        target=r.get('target') or {}; ids=sorted({s.split(':',1)[1] for g in target.get('genes',[]) for s in g.get('dbxrefs',[]) if s.startswith('ENSEMBL:')} & tf_ids)
        bios=[v['library']['biosample'] for v in r.get('replicates',[]) if isinstance(v.get('library'),dict) and isinstance(v['library'].get('biosample'),dict)]
        cats={k:sorted({v['category'] for v in vals}) for k,vals in r.get('audit',{}).items()}
        reasons=[]
        if len(ids)!=1: reasons.append('not_one_expressed_catalogue_TF')
        if not bios: reasons.append('missing_biosample_metadata')
        if any(b.get('treatments') for b in bios): reasons.append('treated_biosample')
        if any(b.get('genetic_modifications') or b.get('applied_modifications') for b in bios): reasons.append('modified_biosample')
        if not r.get('possible_controls'): reasons.append('no_matched_control')
        if cats.get('ERROR') or cats.get('NOT_COMPLIANT'): reasons.append('error_or_noncompliant_audit')
        if 'biological replicates with identical biosample' in cats.get('INTERNAL_ACTION',[]): reasons.append('identical_biosample_replicates')
        files=[f for f in r.get('files',[]) if f.get('status')=='released' and f.get('assembly')=='GRCh38' and f.get('file_format')=='bed' and f.get('output_type')=='conservative IDR thresholded peaks' and len(set(f.get('biological_replicates',[])))>=2]
        if not files: reasons.append('no_replicated_conservative_GRCh38_IDR')
        record={'experiment':r['accession'],'tf_id':ids[0] if len(ids)==1 else None,'release':r.get('date_released',''),'reasons':reasons,'audit':cats}
        if not reasons:
            f=max(files,key=lambda f:(f.get('date_created',''),f['accession']))
            record['file']={k:f.get(k) for k in ['accession','href','md5sum','file_size','biological_replicates','date_created','output_type']}
            eligible.append(record)
        audit.append(record)
    chosen={}
    for r in sorted(eligible,key=lambda r:(r['release'],r['experiment'])):chosen[r['tf_id']]=r
    selected=sorted(chosen.values(),key=lambda r:r['tf_id'])
    (out/'experiment_selection.json').write_text(json.dumps({'protocol_sha256':sha256(protocol_path),'source_sha256':sha256(BASE/'raw/binding/experiments.json'),'selected':selected,'all_experiments':audit},indent=2)+'\n')
    print('Selected',len(selected),'TF assays from',len(eligible),'eligible experiments',flush=True)
    peaks_dir=out/'peaks';peaks_dir.mkdir(exist_ok=True)
    def fetch(r):
        f=r['file'];dest=peaks_dir/(f['accession']+'.bed.gz')
        if not dest.exists():
            url='https://www.encodeproject.org'+f['href'];response=requests.get(url,timeout=(30,120));response.raise_for_status()
            assert len(response.content)==f['file_size'] and hashlib.md5(response.content).hexdigest()==f['md5sum'],f['accession']
            tmp=dest.with_suffix('.partial');tmp.write_bytes(response.content);tmp.rename(dest)
        assert hashlib.md5(dest.read_bytes()).hexdigest()==f['md5sum']
        return r['experiment']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(fetch,selected),1):
            if i%25==0:print('Verified peak files',i,flush=True)
    annotation=BASE/'raw/binding/annotation.gtf.gz'; expected=next(line.split()[0] for line in (BASE/'raw/binding/checksums.txt').read_text().splitlines() if line.split()[-1]=='annotation.gtf.gz')
    assert hashlib.md5(annotation.read_bytes()).hexdigest()==expected
    transcripts={};lengths=defaultdict(int); gene_set=set(genes)
    with gzip.open(annotation,'rt') as handle:
        for line in handle:
            if line.startswith('#'):continue
            f=line.rstrip('\n').split('\t')
            if f[2] not in ['transcript','exon']:continue
            attrs=defaultdict(list)
            for key,val in re.findall(r'(\w+) "([^"]+)"',f[8]):attrs[key].append(val)
            gene=attrs['gene_id'][0].split('.')[0]
            if gene not in gene_set:continue
            tx=attrs['transcript_id'][0]
            if f[2]=='exon':lengths[tx]+=int(f[4])-int(f[3])+1
            else:transcripts[tx]={'gene':gene,'transcript':tx,'chrom':f[0],'strand':f[6],'boundary':int(f[3])-1 if f[6]=='+' else int(f[4]),'mane':'MANE_Select' in attrs.get('tag',[])}
    selected_tx={}
    for tx,r in sorted(transcripts.items(),key=lambda item:(-int(item[1]['mane']),-lengths[item[0]],item[0])):
        if r['gene'] not in selected_tx:selected_tx[r['gene']]={**r,'exon_length':lengths[tx]}
    pd.DataFrame(selected_tx.values()).to_csv(out/'gene_transcription_boundaries.csv',index=False)
    annotated=np.array([g in selected_tx for g in genes]);labels=np.zeros((2,len(tfs),len(genes)),dtype=bool);assayed=np.zeros(len(tfs),dtype=bool);counts=[]
    tf_row={genes[v]:i for i,v in enumerate(tfs)}
    for r in selected:
        by_chr=defaultdict(list);n=0
        with gzip.open(peaks_dir/(r['file']['accession']+'.bed.gz'),'rt') as handle:
            for line in handle:
                if line.startswith(('#','track','browser')):continue
                f=line.split('\t');by_chr[f[0]].append((int(f[1]),int(f[2])));n+=1
        i=tf_row[r['tf_id']];assayed[i]=True
        for chrom in {v['chrom'] for v in selected_tx.values()}:
            idx=np.array([j for j,g in enumerate(genes) if g in selected_tx and selected_tx[g]['chrom']==chrom],dtype=int)
            tss=np.array([selected_tx[genes[j]]['boundary'] for j in idx]);plus=np.array([selected_tx[genes[j]]['strand']=='+' for j in idx])
            start=np.maximum(0,tss-np.where(plus,1000,100));end=tss+np.where(plus,100,1000)
            labels[0,i,idx]=overlaps(by_chr[chrom],start,end)
            labels[1,i,idx]=overlaps(by_chr[chrom],np.maximum(0,tss-2000),tss+2000)
        counts.append({'tf_id':r['tf_id'],'experiment':r['experiment'],'file':r['file']['accession'],'peaks':n,'primary_supported_genes':int(labels[0,i].sum()),'wide_supported_genes':int(labels[1,i].sum())})
    np.savez_compressed(out/'binding_labels.npz',genes=genes,tf_indices=tfs,annotated=annotated,assayed=assayed,primary=labels[0],wide=labels[1])
    pd.DataFrame(counts).to_csv(out/'assay_counts.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'selected_tfs':len(selected),'source_experiments':len(rows),'eligible_experiments':len(eligible),'annotated_genes':int(annotated.sum()),'all_genes':len(genes),'protocol_sha256':sha256(protocol_path),'annotation_sha256':sha256(annotation),'labels_sha256':sha256(out/'binding_labels.npz'),'script_sha256':sha256(Path(__file__))},indent=2)+'\n')
    print('Binding labels prepared',len(selected),'TFs',int(annotated.sum()),'annotated genes',flush=True)

if __name__=='__main__':main()
