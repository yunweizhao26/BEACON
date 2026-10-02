"""Paper queries retain the original aggregation and subtraction order."""
import numpy as np
import pandas as pd
CONTEXT = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}
ORDER = ["hESC", "mDC", "mHSC-E", "mESC"]
TABLES = {}
rows = []
def both(name):
    value = TABLES[name]
    return value.copy(), value.copy()
def add(section, item, context, metric, reference, value):
    rows.append(dict(section=section, item=item, context=context, metric=metric,
                     value=None if value is None or pd.isna(value) else float(value)))
def primary():
    a, b = both('summary/completion_runs.csv')
    for frame in (a, b):
        frame['label'] = frame.variant + ':' + frame.method
    keep = lambda f: f.query('coverage == .8 and ratio == 5 and corruption == 0')
    for metric in ['auprc_trapezoid', 'average_precision', 'all_tf_ap', 'auroc']:
        ta = keep(a).groupby(['label', 'context'])[metric].mean()
        tb = keep(b).groupby(['label', 'context'])[metric].mean()
        for key in sorted(set(ta.index) | set(tb.index)):
            add('80% prior (3-split mean)', key[0], key[1], metric, ta.get(key), tb.get(key))
        for label in sorted(set(keep(a).label) | set(keep(b).label)):
            add('80% prior (3-split mean)', label, 'mean of 4', metric,
                ta.xs(label, level=0).mean() if label in ta.index.get_level_values(0) else None,
                tb.xs(label, level=0).mean() if label in tb.index.get_level_values(0) else None)
    ranges = []
    for frame in (a, b):
        f = keep(frame).query('label == "beacon:beacon"')
        ranges.append(f.groupby('context').auprc_trapezoid.agg(['min', 'max']))
    for context in ORDER:
        add('80% prior (3-split mean)', 'beacon:beacon split min', context, 'auprc_trapezoid', ranges[0].loc[context, 'min'], ranges[1].loc[context, 'min'])
        add('80% prior (3-split mean)', 'beacon:beacon split max', context, 'auprc_trapezoid', ranges[0].loc[context, 'max'], ranges[1].loc[context, 'max'])
    return keep(a), keep(b)

def coverage():
    a, b = both('summary/completion_runs.csv')
    gnn = TABLES["coverage_comparators"].copy()
    gnn['context'] = gnn.dataset.map(CONTEXT)
    gm = gnn.groupby(['context', 'coverage']).auprc_trapezoid.mean()
    sel = lambda f, m: f.query('variant == "beacon" and ratio == 5 and corruption == 0 and method == @m').groupby(['context', 'coverage']).auprc_trapezoid.mean()
    ba, bb, da = sel(a, 'beacon'), sel(b, 'beacon'), sel(a, 'degree_logistic')
    for key in ba.index:
        add('Coverage (pooled AUPRC)', 'BEACON', key[0], f'coverage {key[1]:g}', ba[key], bb.get(key))
        add('Coverage (pooled AUPRC)', 'BEACON - topology control', key[0], f'coverage {key[1]:g}', ba[key] - da[key], bb.get(key) - da[key])
        if key in gm.index:
            add('Coverage (pooled AUPRC)', 'BEACON - GNNLink', key[0], f'coverage {key[1]:g}', ba[key] - gm[key], bb.get(key) - gm[key])
    for context in ORDER:
        first = []
        for frame in (ba, bb):
            beats = [c for c in [.05, .1, .2, .4, .8] if frame[(context, c)] > da[(context, c)]]
            first.append(min(beats) if beats else np.nan)
        add('Coverage (pooled AUPRC)', 'first coverage where BEACON > topology control', context, 'coverage', *first)
        add('Coverage (pooled AUPRC)', 'gain 5% to 80%', context, 'auprc_trapezoid', ba[(context, .8)] - ba[(context, .05)], bb[(context, .8)] - bb[(context, .05)])

def ratio_corruption():
    a, b = both('summary/completion_runs.csv')
    sel = lambda f: f.query('variant == "beacon" and coverage == .2 and method == "beacon"').groupby(['ratio', 'corruption', 'context']).average_precision.mean()
    ta, tb = sel(a), sel(b)
    for key in ta.index:
        add('Ratio and corruption at 20% (pooled AP)', f'ratio {key[0]:g}, corruption {key[1]:g}', key[2], 'average_precision', ta[key], tb.get(key))
    for context in ORDER:
        add('Ratio and corruption at 20% (pooled AP)', 'ratio 20 - ratio 1', context, 'average_precision',
            ta[(20, 0, context)] - ta[(1, 0, context)], tb[(20, 0, context)] - tb[(1, 0, context)])
        add('Ratio and corruption at 20% (pooled AP)', 'corruption 20% - none', context, 'average_precision',
            ta[(5, .2, context)] - ta[(5, 0, context)], tb[(5, .2, context)] - tb[(5, 0, context)])

def concordance():
    a, b = both('completion_summary/hard_challenge_metrics.csv')
    if b is None:
        return
    for frame in (a, b):
        frame['context'] = frame.dataset.map(CONTEXT)
    ta, tb = (f.groupby(['method', 'context']).pair_concordance.mean() for f in (a, b))
    for key in ta.index:
        add('Same-regulator concordance', key[0], key[1], 'pair_concordance', ta[key], tb.get(key))
    for context in ORDER:
        for other in ['gnnlink', 'degree_logistic']:
            add('Same-regulator concordance', f'BEACON - {other}', context, 'pair_concordance',
                ta[('beacon', context)] - ta[(other, context)], tb[('beacon', context)] - tb[(other, context)])

def calibration():
    a, b = both('probability_summary/probability_metrics.csv')
    if b is None:
        return
    sel = lambda f: f.query('suite == "completion" and coverage == .8 and ratio == 5 and corruption == 0').assign(
        context=lambda x: x.dataset_id.map(CONTEXT)).groupby(['method', 'form', 'context']).brier.mean()
    ta, tb = sel(a), sel(b)
    for key in ta.index:
        add('Calibration at 80% (Brier)', f'{key[0]} ({key[1]})', key[2], 'brier', ta[key], tb.get(key))
    for context in ORDER:
        add('Calibration at 80% (Brier)', 'BEACON Platt - topology control Platt', context, 'brier',
            ta[('beacon', 'platt', context)] - ta[('degree_logistic', 'platt', context)],
            tb[('beacon', 'platt', context)] - tb[('degree_logistic', 'platt', context)])
        add('Calibration at 80% (Brier)', 'BEACON Platt - prevalence constant', context, 'brier',
            ta[('beacon', 'platt', context)] - ta[('validation_prevalence_constant', 'constant', context)],
            tb[('beacon', 'platt', context)] - tb[('validation_prevalence_constant', 'constant', context)])
        add('Calibration at 80% (Brier)', 'BEACON Platt - BEACON raw', context, 'brier',
            ta[('beacon', 'platt', context)] - ta[('beacon', 'raw', context)], tb[('beacon', 'platt', context)] - tb[('beacon', 'raw', context)])
    a, b = both('probability_summary/selective_metrics.csv')
    sel = lambda f: f.query('suite == "completion" and coverage == .8 and ratio == 5 and corruption == 0 and method == "beacon"').pipe(
        lambda x: x[np.isclose(x.retained_coverage, .5)]).assign(context=lambda x: x.dataset_id.map(CONTEXT)).groupby(['criterion', 'context']).positive_retention.mean()
    ta, tb = sel(a), sel(b)
    for key in ta.index:
        add('Selective retention at 50% coverage', key[0], key[1], 'positive_retention', ta[key], tb.get(key))

def sergio():
    a, b = both('summary/sergio_runs.csv')
    ta, tb = (f.groupby(['dataset', 'coverage'])[['ap_over_prevalence', 'auroc']].mean() for f in (a, b))
    for key in ta.index:
        for metric in ['ap_over_prevalence', 'auroc']:
            add('SERGIO', key[0], f'coverage {key[1]:g}', metric, ta.loc[key, metric], tb.loc[key, metric] if key in tb.index else None)
    add('SERGIO', 'max hidden fraction of unlabeled', 'all', 'fraction', a.hidden_fraction_of_unlabeled.max(), b.hidden_fraction_of_unlabeled.max())

def stability():
    a, b = both('summary/optimization_seeds.csv')
    ta, tb = (f.groupby(['seed', 'context']).auprc_trapezoid.mean() for f in (a, b))
    for key in ta.index:
        add('Optimization seeds', f'seed {key[0]}', key[1], 'auprc_trapezoid', ta[key], tb.get(key))
    for seed in [14, 42, 100]:
        for metric in ['auprc_trapezoid', 'all_tf_ap']:
            add('Optimization seeds', f'seed {seed}', 'mean of 4', metric,
                a.query('seed == @seed')[metric].mean(), b.query('seed == @seed')[metric].mean())
    ra = a.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).groupby('context').max()
    rb = b.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).groupby('context').max()
    for context in ORDER:
        add('Optimization seeds', 'largest seed range (per split)', context, 'auprc_trapezoid', ra[context], rb[context])
    a, b = both('summary/inducing_points.csv')
    if b is not None and len(b):
        for metric in ['test_auprc_trapezoid', 'test_average_precision', 'validation_average_precision', 'training_seconds', 'peak_cuda_allocated_mb']:
            ta, tb = a.groupby('inducing_points')[metric].mean(), b.groupby('inducing_points')[metric].mean()
            for m in ta.index:
                add('Inducing points', f'M = {m}', 'mean of 12', metric, ta[m], tb.get(m))

def expression_controls():
    a, b = both("expression")

    def pick(frame, readout, cov, seed42=True):
        mask = (frame.readout == readout) & np.isclose(frame.coverage, cov)
        return frame[mask & (frame.opt_seed == 42)] if seed42 else frame[mask]

    for readout in ['beacon_gp', 'beacon_decoder']:
        for cov in [.8, .2]:
            counts, tables = [], []
            for frame in (a, b):
                f = pick(frame, readout, cov)
                wide = f.pivot_table(index=['context', 'split_seed'], columns='control', values='auprc_trapezoid')
                counts.append(int((wide.real > wide[['cell_shuffled', 'gene_permuted', 'random']].max(axis=1)).sum()))
                tables.append(f.pivot_table(index='context', columns='control', values='auprc_trapezoid'))
            add('Expression controls', f'{readout}: real beats all three controls (of 12)', f'coverage {cov:g}', 'auprc_trapezoid', *counts)
            ta, tb = tables
            for control in ['cell_shuffled', 'gene_permuted', 'random']:
                for context in ORDER:
                    add('Expression controls', f'{readout}: real - {control}', f'{context}, coverage {cov:g}', 'auprc_trapezoid',
                        ta.loc[context, 'real'] - ta.loc[context, control], tb.loc[context, 'real'] - tb.loc[context, control])
            for context in ORDER:
                add('Expression controls', f'{readout}: real', f'{context}, coverage {cov:g}', 'auprc_trapezoid',
                    ta.loc[context, 'real'], tb.loc[context, 'real'])
        noise = []
        for frame in (a, b):
            f = pick(frame, readout, .8, seed42=False)
            f = f[f.control == 'real']
            noise.append(f.groupby(['context', 'split_seed']).auprc_trapezoid.agg(lambda x: x.max() - x.min()).mean())
        add('Expression controls', f'{readout}: mean optimization range (seeds 14/42/100)', 'coverage 0.8', 'auprc_trapezoid', *noise)

def resources():
    a, b = both('summary/resources.csv')
    for group in sorted(set(a.group) | set(b.group)):
        for metric in ['total_seconds', 'encoder_seconds', 'gp_seconds', 'peak_rss_mb', 'peak_cuda_allocated_mb']:
            add('Resources (median)', group, 'all', metric, a.query('group == @group')[metric].median(), b.query('group == @group')[metric].median())
