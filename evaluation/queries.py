import numpy as np
import pandas as pd
CONTEXT = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
ORDER = ["hESC", "mDC", "mHSC-E", "mESC"]
TABLES = {}
rows = []
def table(name):
    return TABLES[name].copy()
def add(section, item, context, metric, value):
    rows.append(dict(section=section, item=item, context=context, metric=metric,
                     value=None if value is None or pd.isna(value) else float(value)))
def primary():
    frame = table('summary/completion_runs.csv')
    frame['label'] = frame.variant + ':' + frame.method
    keep = frame.query('coverage == .8')
    for metric in ['auprc_trapezoid', 'average_precision', 'all_tf_ap', 'auroc']:
        means = keep.groupby(['label', 'context'])[metric].mean()
        for key in sorted(means.index):
            add('80% prior (3-split mean)', key[0], key[1], metric, means.get(key))
        for label in sorted(set(keep.label)):
            add('80% prior (3-split mean)', label, 'mean of 4', metric, means.xs(label, level=0).mean())
    ranges = keep.query('label == "beacon:beacon"').groupby('context').auprc_trapezoid.agg(['min', 'max'])
    for context in ORDER:
        add('80% prior (3-split mean)', 'beacon:beacon split min', context, 'auprc_trapezoid', ranges.loc[context, 'min'])
        add('80% prior (3-split mean)', 'beacon:beacon split max', context, 'auprc_trapezoid', ranges.loc[context, 'max'])
    return keep

def coverage():
    frame = table('summary/completion_runs.csv')
    sel = lambda m: frame.query('variant == "beacon" and method == @m').groupby(['context', 'coverage']).auprc_trapezoid.mean()
    beacon, degree, gm = sel('beacon'), sel('degree_logistic'), sel('gnnlink')
    for key in beacon.index:
        add('Coverage (pooled AUPRC)', 'BEACON', key[0], f'coverage {key[1]:g}', beacon[key])
        add('Coverage (pooled AUPRC)', 'BEACON - topology control', key[0], f'coverage {key[1]:g}', beacon[key] - degree[key])
        if key in gm.index:
            add('Coverage (pooled AUPRC)', 'BEACON - GNNLink', key[0], f'coverage {key[1]:g}', beacon[key] - gm[key])
    for context in ORDER:
        beats = [c for c in [.05, .1, .2, .4, .8] if beacon[(context, c)] > degree[(context, c)]]
        add('Coverage (pooled AUPRC)', 'first coverage where BEACON > topology control', context, 'coverage', min(beats) if beats else np.nan)
        add('Coverage (pooled AUPRC)', 'gain 5% to 80%', context, 'auprc_trapezoid', beacon[(context, .8)] - beacon[(context, .05)])

def concordance():
    frame = table('completion_summary/hard_challenge_metrics.csv')
    frame['context'] = frame.dataset.map(CONTEXT)
    means = frame.groupby(['method', 'context']).pair_concordance.mean()
    for key in means.index:
        add('Same-regulator concordance', key[0], key[1], 'pair_concordance', means[key])
    for context in ORDER:
        for other in ['gnnlink', 'degree_logistic']:
            add('Same-regulator concordance', f'BEACON - {other}', context, 'pair_concordance',
                means[('beacon', context)] - means[(other, context)])

def calibration():
    frame = table('probability_summary/probability_metrics.csv')
    brier = frame.query('suite == "completion" and coverage == .8').assign(
        context=lambda x: x.dataset_id.map(CONTEXT)).groupby(['method', 'form', 'context']).brier.mean()
    for key in brier.index:
        add('Calibration at 80% (Brier)', f'{key[0]} ({key[1]})', key[2], 'brier', brier[key])
    for context in ORDER:
        add('Calibration at 80% (Brier)', 'BEACON Platt - topology control Platt', context, 'brier',
            brier[('beacon', 'platt', context)] - brier[('degree_logistic', 'platt', context)])
        add('Calibration at 80% (Brier)', 'BEACON Platt - prevalence constant', context, 'brier',
            brier[('beacon', 'platt', context)] - brier[('validation_prevalence_constant', 'constant', context)])
        add('Calibration at 80% (Brier)', 'BEACON Platt - BEACON raw', context, 'brier',
            brier[('beacon', 'platt', context)] - brier[('beacon', 'raw', context)])
    frame = table('probability_summary/selective_metrics.csv')
    retention = frame.query('suite == "completion" and coverage == .8 and method == "beacon"').pipe(
        lambda x: x[np.isclose(x.retained_coverage, .5)]).assign(context=lambda x: x.dataset_id.map(CONTEXT)).groupby(['criterion', 'context']).positive_retention.mean()
    for key in retention.index:
        add('Selective retention at 50% coverage', key[0], key[1], 'positive_retention', retention[key])

def sergio():
    frame = table('summary/sergio_runs.csv')
    means = frame.groupby(['dataset', 'coverage'])[['ap_over_prevalence', 'auroc']].mean()
    for key in means.index:
        for metric in ['ap_over_prevalence', 'auroc']:
            add('SERGIO', key[0], f'coverage {key[1]:g}', metric, means.loc[key, metric])
    add('SERGIO', 'max hidden fraction of unlabeled', 'all', 'fraction', frame.hidden_fraction_of_unlabeled.max())

def stability():
    frame = table('summary/optimization_seeds.csv')
    means = frame.groupby(['seed', 'context']).auprc_trapezoid.mean()
    for key in means.index:
        add('Optimization seeds', f'seed {key[0]}', key[1], 'auprc_trapezoid', means[key])
    for seed in [14, 42, 100]:
        for metric in ['auprc_trapezoid', 'all_tf_ap']:
            add('Optimization seeds', f'seed {seed}', 'mean of 4', metric, frame.query('seed == @seed')[metric].mean())
    spread = frame.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).groupby('context').max()
    for context in ORDER:
        add('Optimization seeds', 'largest seed range (per split)', context, 'auprc_trapezoid', spread[context])
    frame = table('summary/inducing_points.csv')
    if len(frame):
        for metric in ['test_auprc_trapezoid', 'test_average_precision', 'validation_average_precision', 'training_seconds', 'peak_cuda_allocated_mb']:
            means = frame.groupby('inducing_points')[metric].mean()
            for m in means.index:
                add('Inducing points', f'M = {m}', 'mean of 12', metric, means[m])

def expression_controls():
    frame = table("expression")

    def pick(readout, cov, seed42=True):
        mask = (frame.readout == readout) & np.isclose(frame.coverage, cov)
        return frame[mask & (frame.opt_seed == 42)] if seed42 else frame[mask]

    for readout in ['beacon_gp', 'beacon_decoder']:
        for cov in [.8, .2]:
            f = pick(readout, cov)
            wide = f.pivot_table(index=['context', 'split_seed'], columns='control', values='auprc_trapezoid')
            add('Expression controls', f'{readout}: real beats all three controls (of 12)', f'coverage {cov:g}', 'auprc_trapezoid',
                int((wide.real > wide[['cell_shuffled', 'gene_permuted', 'random']].max(axis=1)).sum()))
            means = f.pivot_table(index='context', columns='control', values='auprc_trapezoid')
            for control in ['cell_shuffled', 'gene_permuted', 'random']:
                for context in ORDER:
                    add('Expression controls', f'{readout}: real - {control}', f'{context}, coverage {cov:g}', 'auprc_trapezoid',
                        means.loc[context, 'real'] - means.loc[context, control])
            for context in ORDER:
                add('Expression controls', f'{readout}: real', f'{context}, coverage {cov:g}', 'auprc_trapezoid', means.loc[context, 'real'])
        f = pick(readout, .8, seed42=False)
        f = f[f.control == 'real']
        add('Expression controls', f'{readout}: mean optimization range (seeds 14/42/100)', 'coverage 0.8', 'auprc_trapezoid',
            f.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).mean())
    seeds = table('summary/comparator_seeds.csv')
    for method, f in seeds.groupby('method'):
        add('Expression controls', f'{method}: mean optimization range (seeds 14/42/100)', 'coverage 0.8', 'auprc_trapezoid',
            f.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).mean())
        means = f.groupby(['context', 'seed']).auprc_trapezoid.mean()
        for context in ORDER:
            add('Expression controls', f'{method}: seed mean, minimum over seeds', context, 'auprc_trapezoid', means.loc[context].min())
            add('Expression controls', f'{method}: seed mean, maximum over seeds', context, 'auprc_trapezoid', means.loc[context].max())

def resources():
    frame = table('summary/resources.csv')
    for group in sorted(set(frame.group)):
        for metric in ['total_seconds', 'encoder_seconds', 'gp_seconds', 'peak_rss_mb', 'peak_cuda_allocated_mb']:
            add('Resources (median)', group, 'all', metric, frame.query('group == @group')[metric].median())
