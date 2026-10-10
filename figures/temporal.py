"""Temporal completion: recovery of regulatory edges first reported after the prior's cutoff year.

Reads the per-setting tables written by the temporal analysis (``*_metrics.csv``, ``*_paired.csv`` and
``*_concentration.csv``) and writes the main-text figure and two supplementary tables.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from figures.drawing import plt, Line2D, Patch, STYLE, relabel, row_major

PUBLISHED = ['BEACON', 'GNNLink', 'GCLink', 'scRegulate', 'RegGAIN', 'GENELink', 'scRegNet', 'GRNBoost2', 'GENIE3', 'Inferelator']
BASELINES = ['Expression correlation', 'Degree baseline', 'Target popularity']
STYLE_KEY = {'Degree baseline': 'Topology control', 'Target popularity': 'Prior target in-degree', 'Regulator popularity': 'Prior regulator out-degree'}
CONTEXTS = {'k562': 'K562', 'tcell_resting': 'Resting T', 'tcell_stimulated': 'Re-stimulated T', 'rpe1': 'RPE1'}
CUTOFFS = (2000, 2005, 2010)
PRIMARY = 'all later edges'


def style(method):
    return STYLE[STYLE_KEY.get(method, method)]


def load(source):
    read = lambda kind: pd.concat(pd.read_csv(f) for f in sorted(Path(source).glob(f'*_{kind}.csv')))
    return tuple(frame[frame.cutoff.isin(CUTOFFS)] for frame in (read('metrics'), read('paired'), read('concentration')))


def outcome(paired, variant, comparators):
    q = paired[paired.variant.eq(variant) & paired.quantity.eq('ap') & paired.comparator.isin(comparators)]
    q = q.assign(result=np.where(q.ci_low > 0, 'higher', np.where(q.ci_high < 0, 'lower', 'tie')))
    return q.groupby('comparator').result.value_counts().unstack(fill_value=0).reindex(index=comparators, columns=['higher', 'tie', 'lower'], fill_value=0)


def draw(metrics, paired, concentration):
    fig = plt.figure(figsize=(7.15, 5.6), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :])
    frame = metrics[metrics.variant.eq(PRIMARY)].assign(fold=lambda d: d.pooled_ap / d.prevalence)
    methods = PUBLISHED + BASELINES
    width = .86 / len(methods)
    for j, method in enumerate(methods):
        g = frame[frame.method.eq(method)].groupby('cutoff').fold.agg(['mean', 'min', 'max']).reindex(CUTOFFS)
        x = np.arange(len(CUTOFFS)) + (j - (len(methods) - 1) / 2) * width
        color, _, _, hatch = style(method)
        ax.bar(x, g['mean'], width, color=color, hatch=hatch, edgecolor='#333333', linewidth=.3, label=method)
        ax.errorbar(x, g['mean'], yerr=[g['mean'] - g['min'], g['max'] - g['mean']], fmt='none', ecolor='#222222', elinewidth=.5, capsize=1)
    ax.axhline(1, color='#555555', lw=.7, ls='--')
    ax.set_xticks(range(len(CUTOFFS)), [f'Prior: edges reported by {c}' for c in CUTOFFS])
    ax.set_ylabel('Pooled AP / chance'); ax.set_title('Recovery of edges reported after the cutoff', loc='left', fontsize=8.2)
    ax.text(-.06, 1.02, 'A', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(row_major(handles, 7), row_major(labels, 7), ncol=7, frameon=False, fontsize=6.2, loc='upper center',
              bbox_to_anchor=(.5, -.1), handlelength=1.4, columnspacing=.9)

    ax = fig.add_subplot(grid[1, 0])
    counts = outcome(paired, PRIMARY, PUBLISHED[1:])
    colors = {'higher': '#0B6E4F', 'tie': '#C9C9C9', 'lower': '#C0480A'}
    names = {'higher': 'BEACON higher', 'tie': 'No difference detected', 'lower': 'BEACON lower'}
    left = np.zeros(len(counts))
    for result in ('higher', 'tie', 'lower'):
        ax.barh(range(len(counts)), counts[result], left=left, color=colors[result], edgecolor='white', linewidth=.4, label=names[result])
        left += counts[result].to_numpy()
    ax.set_yticks(range(len(counts)), counts.index); ax.invert_yaxis()
    ax.set_xlabel('Settings (of 12)'); ax.set_xlim(0, 12)
    ax.set_title('Per-regulator AP, BEACON vs. each method', loc='left', fontsize=8.2)
    ax.text(-.38, 1.02, 'B', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.42, -.17), ncol=3, handlelength=1, columnspacing=.8)

    ax = fig.add_subplot(grid[1, 1])
    shades = dict(zip(CONTEXTS, ['#3B3B3B', '#6F6F6F', '#A0A0A0', '#CFCFCF']))
    width = .8 / len(CONTEXTS)
    for k, (context, name) in enumerate(CONTEXTS.items()):
        g = concentration[concentration.context.eq(context)].set_index('cutoff').reindex(CUTOFFS)
        ax.bar(np.arange(len(CUTOFFS)) + (k - 1.5) * width, g.top10pct_regulator_share, width, color=shades[context],
               edgecolor='#333333', linewidth=.3, label=name)
    ax.axhline(.1, color='#555555', lw=.7, ls='--')
    ax.set_xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS]); ax.set_xlabel('Cutoff year'); ax.set_ylim(0, 1)
    ax.set_ylabel('Share of later edges')
    ax.set_title('Later edges on the 10% of regulators\nwith the most prior edges', loc='left', fontsize=8.2)
    ax.text(-.2, 1.02, 'C', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=4, handlelength=1, columnspacing=.8)
    return fig


def tables(metrics, paired):
    frame = metrics[metrics.variant.eq(PRIMARY)]
    columns = [(c, x) for c in CUTOFFS for x in CONTEXTS]
    def block(value, digits):
        wide = frame.pivot_table(index='method', columns=['cutoff', 'context'], values=value).reindex(columns=pd.MultiIndex.from_tuples(columns))
        best = wide.loc[PUBLISHED].max()
        lines = []
        for method in PUBLISHED + BASELINES:
            cells = [(f'\\textbf{{{v:.{digits}f}}}' if method in PUBLISHED and round(v, digits) == round(best[col], digits) else f'{v:.{digits}f}')
                     for col, v in wide.loc[method].items()]
            lines.append(f'{method} & ' + ' & '.join(cells) + ' \\\\')
        return lines
    head = ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{Edges reported by {c}}}' for c in CUTOFFS) + ' \\\\'
    rule = ''.join(f'\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(3))
    sub = 'Method & ' + ' & '.join(['K562', 'Rest.', 'Stim.', 'RPE1'] * 3) + ' \\\\'
    later = frame[frame.method.eq('BEACON')].set_index(['cutoff', 'context']).positives.reindex(columns)
    counts = 'Later edges & ' + ' & '.join(f'{int(v):,}' for v in later) + ' \\\\'
    main = '\n'.join(['\\toprule', head, rule, sub, '\\midrule', counts, '\\midrule', '\\multicolumn{13}{@{}l}{\\textit{Pooled AP}} \\\\',
                      *block('pooled_ap', 4), '\\midrule', '\\multicolumn{13}{@{}l}{\\textit{Mean per-regulator AP}} \\\\', *block('mean_ap', 4), '\\bottomrule'])
    rows = []
    for variant, name in ((PRIMARY, 'All later edges'), ('supported by a paper citing <= 50 edges', 'Without large-paper-only edges'),
                          ('two or more later papers', 'Two or more later papers')):
        wide = metrics[metrics.variant.eq(variant)].pivot_table(index=['cutoff', 'context'], columns='method', values='pooled_ap')
        rank = wide[PUBLISHED].rank(axis=1, ascending=False).BEACON
        counts = outcome(paired, variant, PUBLISHED[1:])
        cell = lambda m: '/'.join(str(int(counts.loc[m, r])) for r in ('higher', 'tie', 'lower'))
        rows.append(f'{name} & {int((rank == 1).sum())} & {int((rank == 2).sum())} & {int((rank >= 3).sum())} & '
                    f'{cell("scRegNet")} & {cell("GCLink")} & {cell("GNNLink")} & '
                    f'{min(int(counts.loc[m, "higher"]) for m in PUBLISHED[1:] if m not in ("scRegNet", "GCLink", "GNNLink"))} \\\\')
    robustness = '\n'.join(['\\toprule', ' & \\multicolumn{3}{c}{BEACON rank by pooled AP} & \\multicolumn{4}{c}{Per-regulator AP: higher/tie/lower} \\\\',
                            '\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}', 'Positives & 1st & 2nd & 3rd or lower & scRegNet & GCLink & GNNLink & Other six, higher (min.) \\\\',
                            '\\midrule', *rows, '\\bottomrule'])
    return main, robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metrics, paired, concentration = load(args.source)
    fig = draw(metrics, paired, concentration); relabel(fig)
    for extension in ('png', 'pdf'):
        fig.savefig(args.out / f'temporal_validation.{extension}', dpi=300, bbox_inches='tight')
    main_table, robustness = tables(metrics, paired)
    (args.out / 'temporal_table.tex').write_text(main_table + '\n')
    (args.out / 'temporal_robustness.tex').write_text(robustness + '\n')


if __name__ == '__main__':
    main()
