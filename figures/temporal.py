"""Temporal completion: recovery of regulatory edges first reported after the prior's cutoff year.

Reads the tables written by ``python -m evaluation.temporal`` (``metrics.csv``, ``paired.csv`` and
``concentration.csv``) and writes the two main-text figures and two supplementary tables.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from figures.drawing import plt, Line2D, Patch, STYLE, relabel, row_major
from matplotlib.colors import LinearSegmentedColormap

PUBLISHED = ['BEACON', 'GNNLink', 'GCLink', 'scRegulate', 'RegGAIN', 'GENELink', 'scRegNet', 'GRNBoost2', 'GENIE3', 'Inferelator']
BASELINES = ['Expression correlation', 'Degree baseline', 'Target popularity']
STYLE_KEY = {'Degree baseline': 'Topology control', 'Target popularity': 'Prior target in-degree', 'Regulator popularity': 'Prior regulator out-degree'}
CONTEXTS = {'k562': 'K562', 'tcell_resting': 'Resting T', 'tcell_stimulated': 'Re-stimulated T', 'rpe1': 'RPE1'}
CUTOFFS = (2000, 2005, 2010)
PRIMARY = 'all later edges'


def style(method):
    return STYLE[STYLE_KEY.get(method, method)]


def load(source):
    read = lambda kind: pd.read_csv(Path(source) / f'{kind}.csv')
    return tuple(frame[frame.cutoff.isin(CUTOFFS)] for frame in (read('metrics'), read('paired'), read('concentration')))


def outcome(paired, variant, comparators):
    q = paired[paired.variant.eq(variant) & paired.quantity.eq('ap') & paired.comparator.isin(comparators)]
    q = q.assign(result=np.where(q.ci_low > 0, 'higher', np.where(q.ci_high < 0, 'lower', 'tie')))
    return q.groupby('comparator').result.value_counts().unstack(fill_value=0).reindex(index=comparators, columns=['higher', 'tie', 'lower'], fill_value=0)


def draw(metrics, paired):
    fig = plt.figure(figsize=(7.15, 5.9), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, .018])
    ax = fig.add_subplot(grid[0, :])
    frame = metrics[metrics.variant.eq(PRIMARY)].assign(fold=lambda d: d.pooled_ap / d.prevalence)
    methods = PUBLISHED + BASELINES
    width = .86 / len(methods)
    for j, method in enumerate(methods):
        g = frame[frame.method.eq(method)].groupby('cutoff').fold.agg(['mean', 'min', 'max']).reindex(CUTOFFS)
        x = np.arange(len(CUTOFFS)) + (j - (len(methods) - 1) / 2) * width
        color, _, _, hatch = style(method)
        ax.bar(x, g['mean'], width, color=color, hatch=hatch, edgecolor='#555555', linewidth=.3, label=method)
        ax.errorbar(x, g['mean'], yerr=[g['mean'] - g['min'], g['max'] - g['mean']], fmt='none', ecolor='#555555', elinewidth=.5, capsize=1)
    ax.axhline(1, color='#555555', lw=.7, ls='--')
    ax.set_xticks(range(len(CUTOFFS)), [f'Prior: edges reported by {c}' for c in CUTOFFS])
    ax.set_ylabel('Pooled AP / chance'); ax.set_title('Recovery of edges reported after the cutoff', loc='left', fontsize=8.2)
    ax.text(-.06, 1.02, 'A', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(row_major(handles, 7), row_major(labels, 7), ncol=7, frameon=False, fontsize=6.2, loc='upper center',
              bbox_to_anchor=(.5, -.1), handlelength=1.4, columnspacing=.9)

    # B: per-setting difference in mean per-regulator AP with its paired regulator-bootstrap interval.
    ax = fig.add_subplot(grid[1, 0])
    comparators = PUBLISHED[1:] + BASELINES
    settings = [(c, x) for c in CUTOFFS for x in CONTEXTS]
    q = paired[paired.variant.eq(PRIMARY) & paired.quantity.eq('ap')].set_index(['comparator', 'cutoff', 'context'])
    cell = lambda column: np.array([[q.loc[(m, c, x), column] for c, x in settings] for m in comparators])
    values, low, high = cell('beacon_minus'), cell('ci_low'), cell('ci_high')
    cmap = LinearSegmentedColormap.from_list('beacon_difference', ['#E8803A', '#FFFFFF', '#1B9E77'])
    image = ax.imshow(values, cmap=cmap, vmin=-.05, vmax=.05, aspect='auto')
    rows, columns = np.nonzero((low > 0) | (high < 0))
    ax.scatter(columns, rows, s=5, color='#333333', linewidths=0)
    for x in (3.5, 7.5):
        ax.axvline(x, color='white', lw=2.5)
    ax.axhline(len(PUBLISHED) - 1.5, color='white', lw=2.5)
    ax.set_xticks(range(len(settings)), [CONTEXTS[x].replace('Re-stimulated', 'Re-stim.') for _, x in settings], fontsize=6, rotation=30, ha='right')
    ax.set_yticks(range(len(comparators)), comparators)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for k, cutoff in enumerate(CUTOFFS):
        ax.text(1.5 + 4 * k, -.75, f'Prior: edges reported by {cutoff}', ha='center', va='bottom', fontsize=6.6)
    ax.set_title('Mean per-regulator AP, BEACON minus each method', loc='left', fontsize=8.2, pad=14)
    ax.text(-.063, 1.095, 'B', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    ax.legend([Line2D([], [], marker='o', ls='', color='#333333', ms=2.2)], ['95% interval excludes zero'], frameon=False,
              fontsize=6.2, loc='upper right', bbox_to_anchor=(1, 1.19), handletextpad=.3)
    fig.colorbar(image, cax=fig.add_subplot(grid[1, 1]), label='Difference in AP')
    return fig


def concentration_figure(concentration):
    """Share of later-reported edges whose regulator was among the 10% with the most prior edges."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7), layout='constrained')
    shades = dict(zip(CONTEXTS, ['#707070', '#969696', '#B8B8B8', '#DEDEDE']))
    width = .8 / len(CONTEXTS)
    for k, (context, name) in enumerate(CONTEXTS.items()):
        g = concentration[concentration.context.eq(context)].set_index('cutoff').reindex(CUTOFFS)
        ax.bar(np.arange(len(CUTOFFS)) + (k - 1.5) * width, g.top10pct_regulator_share, width, color=shades[context],
               edgecolor='#555555', linewidth=.3, label=name)
    ax.axhline(.1, color='#555555', lw=.7, ls='--')
    ax.set_xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS]); ax.set_xlabel('Cutoff year'); ax.set_ylim(0, 1)
    ax.set_ylabel('Share of later edges')
    ax.set_title('Later edges on the 10% of regulators\nwith the most prior edges', loc='left', fontsize=8.2)
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=4, handlelength=1, columnspacing=.8)
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
    for name, fig in (('temporal_validation', draw(metrics, paired)), ('temporal_concentration', concentration_figure(concentration))):
        relabel(fig)
        for extension in ('png', 'pdf'):
            fig.savefig(args.out / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    main_table, robustness = tables(metrics, paired)
    (args.out / 'temporal_table.tex').write_text(main_table + '\n')
    (args.out / 'temporal_robustness.tex').write_text(robustness + '\n')


if __name__ == '__main__':
    main()
