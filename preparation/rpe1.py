#!/usr/bin/env python3
"""Prepare protocol-frozen RPE1 inputs; never decode perturbation expression/effects."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from beacon.data import configuration
import hashlib
import json
import math
import os
import resource
import socket
import time
import urllib.request

BASE = configuration()["data_root"]
BEACON = BASE
QUARANTINE = BASE / "raw/rpe1"
OUTPUT = BASE / "prepared/rpe1"
PROTOCOL = BASE / "raw/rpe1/protocol.md"
CATALOG = BASE / "raw/transcription_factors.csv"
SOURCES = json.loads(Path(__file__).with_name("manifest.json").read_text())["rpe1_sources"]

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_cpu_seconds=r.ru_utime, system_cpu_seconds=r.ru_stime,
                process_max_rss_kib=r.ru_maxrss, host=socket.gethostname(),
                slurm_job_id=os.getenv('SLURM_JOB_ID'), gpu_used=False)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')

def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest())

def download(source, workers):
    path = QUARANTINE / source["name"]
    observed = hashes(path)
    assert path.stat().st_size == source["size"] and observed["md5"] == source["md5"]
    return path, dict(name=source["name"], verified_hashes=observed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.workers <= 32
    assert PROTOCOL.exists() and CATALOG.exists()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    access = []
    def read(group, name, rows=None):
        import h5py
        import numpy as np
        ds = group[name]
        access.append(dict(file=Path(group.file.filename).name, path=ds.name,
                           rows='all metadata rows' if rows is None else len(rows),
                           purpose='control identity, control QC or gene identity'))
        if 'categories' in ds.attrs:
            cat = group.file[ds.attrs['categories']]
            access.append(dict(file=Path(group.file.filename).name, path=cat.name,
                               purpose='decode permitted categorical metadata'))
            labels = cat.asstr()[:] if h5py.check_string_dtype(cat.dtype) else cat[:]
            codes = ds[:] if rows is None else ds[rows]
            return np.array([labels[c] if c >= 0 else '' for c in codes])
        if h5py.check_string_dtype(ds.dtype):
            return ds.asstr()[:] if rows is None else ds.asstr()[rows]
        return ds[:] if rows is None else ds[rows]
    raw, raw_log = download(SOURCES['raw'], args.workers)
    bulk, bulk_log = download(SOURCES['bulk'], args.workers)
    import h5py
    import numpy as np
    import pandas as pd
    prepare_started = time.perf_counter()
    with h5py.File(bulk, 'r') as f:
        identities = read(f['obs'], 'gene_transcript')
        core_flags = read(f['obs'], 'core_control').astype(bool)
        assert len(identities) == 2679 and len(set(identities)) == len(identities)
        core_ids = set(identities[core_flags])
        assert core_ids and all(v.endswith('_non-targeting') for v in core_ids)
        pd.DataFrame({'gene_transcript': sorted(core_ids)}).to_csv(OUTPUT/'author_core_control_constructs.csv', index=False)
    catalog = pd.read_csv(CATALOG)
    tf_rows = catalog.loc[catalog['Is TF?'] == 'Yes']
    tf_ids = set(tf_rows['Ensembl ID'])
    with h5py.File(raw, 'r') as f:
        obs, var = f['obs'], f['var']
        assert f['X'].shape == (247914, 8749)
        group_ids = read(obs, 'gene_transcript')
        all_core_rows = np.flatnonzero(np.isin(group_ids, list(core_ids)))
        assert len(all_core_rows) > 1000
        core_gene_ids = read(obs, 'gene_id', all_core_rows)
        assert np.all(core_gene_ids == 'non-targeting')
        umi_core = read(obs, 'UMI_count', all_core_rows)
        adjusted_core = read(obs, 'core_adjusted_UMI_count', all_core_rows)
        mito_core = read(obs, 'mitopercent', all_core_rows)
        gems_core = read(obs, 'gem_group', all_core_rows)
        assert np.isfinite(umi_core).all() and np.isfinite(adjusted_core).all() and np.isfinite(mito_core).all()
        quality = (adjusted_core >= 3000) & (mito_core < 0.11) & (umi_core > 0)
        controls = all_core_rows[quality]
        umi = umi_core[quality]
        gems = gems_core[quality]
        assert len(controls) > 1000 and (umi > 0).all()
        rng = np.random.default_rng(42)
        selected = np.sort(np.concatenate([rng.choice(controls[gems == g], min(50, np.sum(gems == g)), replace=False)
                                           for g in np.unique(gems)]))
        selected_positions = np.searchsorted(controls, selected)
        assert np.array_equal(controls[selected_positions], selected)
        genes = read(var, 'gene_id').astype(str)
        symbols = read(var, 'gene_name').astype(str)
        assert len(genes) == 8749 and len(set(genes)) == len(genes)
        stored = pd.read_csv(QUARANTINE/'metadata/gene_identity_metadata.tsv', sep='\t')
        assert np.array_equal(genes, stored['gene_id'].astype(str).to_numpy())
        detection = np.zeros(len(genes), dtype=np.int64)
        sums = np.zeros(len(genes), dtype=np.float64)
        training_raw = np.empty((len(selected), len(genes)), dtype=np.float32)
        expression_reads = []
        for lo in range(0, len(controls), 256):
            hi = min(lo + 256, len(controls))
            indices = controls[lo:hi]
            # Only selected, quality-passing non-targeting control rows enter NumPy.
            counts = f['X'][indices, :]
            assert np.isfinite(counts).all(), 'Nonfinite control count'
            assert (counts >= 0).all(), 'Negative control count'
            assert np.equal(counts, np.floor(counts)).all(), 'Noninteger control count'
            detection += (counts > 0).sum(axis=0)
            sums += counts.sum(axis=0, dtype=np.float64)
            tlo, thi = np.searchsorted(selected, [indices[0], indices[-1]+1])
            if thi > tlo:
                positions = np.searchsorted(indices, selected[tlo:thi])
                assert np.array_equal(indices[positions], selected[tlo:thi])
                training_raw[tlo:thi] = counts[positions]
            expression_reads.append(dict(start_position_in_passing_controls=lo, stop_position=hi,
                                         row_indices=indices.tolist(), gene_columns='all 8749 released genes'))
            if lo % 2048 == 0:
                print(json.dumps({'event':'control_expression_progress','passing_controls_read':hi,
                                  'total_passing_controls':len(controls)}), flush=True)
        access.append(dict(file=raw.name,path='/X',purpose='control-only gene detection and predictor preparation',
                           permitted_rows_file='all_passing_controls.csv.gz', batches=expression_reads,
                           perturbation_rows_read=0))
        keep = detection / len(controls) >= 0.01
        tf_mask = np.isin(genes, list(tf_ids))
        qc = pd.DataFrame(dict(gene_id=genes, symbol=symbols, control_detected_cells=detection,
                               control_detection_fraction=detection/len(controls),
                               control_mean_counts=sums/len(controls), retained=keep, lambert_tf=tf_mask))
        qc.to_csv(OUTPUT/'gene_qc.csv', index=False)
        selected_umi = umi[selected_positions]
        expression = np.log1p(10000 * training_raw[:, keep] / selected_umi[:, None]).T.astype(np.float32)
        assert np.isfinite(expression).all() and (expression >= 0).all()
        np.savez_compressed(OUTPUT/'control_expression.npz', expression=expression,
                            genes=genes[keep], symbols=symbols[keep],
                            tf_indices=np.flatnonzero(tf_mask[keep]),
                            training_cell_indices=selected, all_control_cell_indices=controls)
        barcodes = read(obs, 'cell_barcode', controls)
        control_table = pd.DataFrame(dict(cell_index=controls, barcode=barcodes, gem_group=gems,
                                         umi_count=umi, core_adjusted_UMI_count=adjusted_core[quality],
                                         mitochondrial_fraction=mito_core[quality],
                                         perturbation_id=group_ids[controls]))
        control_table.to_csv(OUTPUT/'all_passing_controls.csv.gz', index=False)
        training_table = control_table.iloc[selected_positions].copy()
        training_table.to_csv(OUTPUT/'training_controls.csv.gz', index=False)
        counts_by_gem = []
        for g in np.unique(gems_core):
            counts_by_gem.append(dict(gem_group=int(g), assigned_author_core_controls=int(np.sum(gems_core==g)),
                                     passing_core_controls=int(np.sum(gems==g)),
                                     sampled_controls=int(np.sum(training_table.gem_group.to_numpy()==g))))
        pd.DataFrame(counts_by_gem).to_csv(OUTPUT/'control_counts_by_gem.csv', index=False)
        expressed = qc.loc[keep & tf_mask].copy()
        lookup = dict(zip(tf_rows['Ensembl ID'], tf_rows['HGNC symbol']))
        expressed['lambert_symbol'] = expressed.gene_id.map(lookup)
        expressed['retained_gene_index'] = [int(np.flatnonzero(genes[keep] == g)[0]) for g in expressed.gene_id]
        expressed.to_csv(OUTPUT/'expressed_tf_universe.csv', index=False)
        # Check persisted alignment and deterministic sampling without touching new source rows.
        saved = np.load(OUTPUT/'control_expression.npz')
        assert saved['expression'].shape == (int(keep.sum()), len(selected))
        assert np.array_equal(saved['genes'], genes[keep])
        assert np.array_equal(saved['training_cell_indices'], training_table.cell_index.to_numpy())
        assert np.array_equal(saved['tf_indices'], expressed.retained_gene_index.to_numpy())
        source_counts = dict(source_cells=len(group_ids), source_measured_genes=len(genes),
                             author_core_control_constructs=len(core_ids), assigned_author_core_control_cells=len(all_core_rows),
                             passing_core_control_cells=len(controls), training_control_cells=len(selected),
                             retained_genes=int(keep.sum()), expressed_catalog_TFs=len(expressed),
                             passing_GEM_groups=len(np.unique(gems)), sampled_GEM_groups=training_table.gem_group.nunique())
    source_hashes = {raw.name:raw_log['verified_hashes'], bulk.name:bulk_log['verified_hashes']}
    dump(OUTPUT/'ACCESS_LOG.json', dict(completed_UTC=utc(), permitted_datasets=access,
         bulk_X_read=False, bulk_on_target_values_read=False, perturbation_expression_rows_read=0,
         test_or_effect_summaries_read=False, models_run=False, protocol_sha256=hashes(PROTOCOL)['sha256']))
    retained_files = sorted(p for p in OUTPUT.iterdir() if p.is_file())
    manifest = dict(completed_UTC=utc(), **source_counts, sources=source_hashes,
                    source_urls={s['name']:f"https://ndownloader.figshare.com/files/{s['id']}" for s in SOURCES.values()},
                    code_sha256=hashes(Path(__file__))['sha256'], protocol_sha256=hashes(PROTOCOL)['sha256'],
                    TF_catalog_sha256=hashes(CATALOG)['sha256'],
                    files={p.name:dict(bytes=p.stat().st_size, **hashes(p)) for p in retained_files},
                    normalization='log1p(10000 * raw count / total-cell UMI_count)',
                    cell_QC='Author non-targeting core controls; core_adjusted_UMI_count >= 3000; mitopercent < 0.11; UMI_count > 0',
                    gene_QC='>=1% detection among all quality-passing author core control cells, within released gene universe',
                    sampling='NumPy default_rng(42); groups sorted; up to 50 controls per GEM group; row indices sorted',
                    raw_control_counts_finite_nonnegative_integer=True,
                    replication_note='GEM groups are technical captures, not donor-level replication.',
                    response_eligibility_assessed=False, models_run=False,
                    wall_seconds=time.perf_counter()-started,
                    preparation_seconds=time.perf_counter()-prepare_started, resource=usage())
    dump(OUTPUT/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
