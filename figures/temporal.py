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
SEEDS = (42, 14, 100)
SEEDED = ('BEACON', 'scRegNet')
PERMUTED = 'scRegNet permuted embeddings'


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
    ax.set_xticks(range(len(CUTOFFS)), [f'Prior: edges discovered by {c}' for c in CUTOFFS])
    ax.set_ylabel('Pooled AP / chance'); ax.set_title('Recovery of edges discovered after the cutoff', loc='left', fontsize=8.2)
    ax.text(-.06, 1.02, 'A', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(row_major(handles, 7), row_major(labels, 7), ncol=7, frameon=False, fontsize=6.2, loc='upper center',
              bbox_to_anchor=(.5, -.1), handlelength=1.4, columnspacing=.9)

    ax = fig.add_subplot(grid[1, 0])
    comparators = PUBLISHED[1:] + BASELINES
    settings = [(c, x) for c in CUTOFFS for x in CONTEXTS]
    q = paired[paired.variant.eq(PRIMARY) & paired.quantity.eq('ap')].set_index(['comparator', 'cutoff', 'context'])
    cell = lambda column: np.array([[q.loc[(m, c, x), column] for c, x in settings] for m in comparators])
    values, low, high = cell('beacon_minus'), cell('ci_low'), cell('ci_high')
    cmap = LinearSegmentedColormap.from_list('beacon_difference', ['#E8803A', '#FFFFFF', '#1B9E77'])
    image = ax.imshow(values, cmap=cmap, vmin=-.05, vmax=.05, aspect='auto')
    for (row, column), value in np.ndenumerate(values):
        excludes = low[row, column] > 0 or high[row, column] < 0
        text = f'{value:.3f}' if round(value, 3) else '0.000'
        ax.text(column, row, text.replace('-', '−'), ha='center', va='center', fontsize=6,
                color='#111111' if excludes else '#A3A3A3')
    for x in (3.5, 7.5):
        ax.axvline(x, color='white', lw=2.5)
    ax.axhline(len(PUBLISHED) - 1.5, color='white', lw=2.5)
    ax.set_xticks(range(len(settings)), [CONTEXTS[x].replace('Re-stimulated', 'Re-stim.') for _, x in settings], fontsize=6, rotation=30, ha='right')
    ax.set_yticks(range(len(comparators)), comparators)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for k, cutoff in enumerate(CUTOFFS):
        ax.text(1.5 + 4 * k, -.75, f'Prior: edges discovered by {cutoff}', ha='center', va='bottom', fontsize=6.6)
    ax.set_title('Mean per-regulator AP, BEACON minus each method', loc='left', fontsize=8.2, pad=14)
    ax.text(-.063, 1.095, 'B', transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')
    ax.text(1, 1.095, 'Grey numbers: 95% interval includes zero', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.2)
    fig.colorbar(image, cax=fig.add_subplot(grid[1, 1]), label='Difference in AP')
    return fig


def concentration_figure(concentration):
    fig, ax = plt.subplots(figsize=(3.4, 2.7), layout='constrained')
    shades = dict(zip(CONTEXTS, ['#707070', '#969696', '#B8B8B8', '#DEDEDE']))
    width = .8 / len(CONTEXTS)
    for k, (context, name) in enumerate(CONTEXTS.items()):
        g = concentration[concentration.context.eq(context)].set_index('cutoff').reindex(CUTOFFS)
        ax.bar(np.arange(len(CUTOFFS)) + (k - 1.5) * width, g.top10pct_regulator_share, width, color=shades[context],
               edgecolor='#555555', linewidth=.3, label=name)
    ax.axhline(.1, color='#555555', lw=.7, ls='--')
    ax.set_xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS]); ax.set_xlabel('Cutoff year'); ax.set_ylim(0, 1)
    ax.set_ylabel('Share of held-out edges')
    ax.set_title('Held-out edges on the 10% of regulators\nwith the most known edges', loc='left', fontsize=8.2)
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=4, handlelength=1, columnspacing=.8)
    return fig


def seed_blocks(runs):
    columns = [(c, x) for c in CUTOFFS for x in CONTEXTS]
    values = ['pooled_ap', 'mean_ap', 'pooled_top1000']
    seeded = runs[runs.method.isin(SEEDED)].set_index(['method', 'seed', 'cutoff', 'context'])
    wanted = [(m, s, c, x) for m in SEEDED for s in SEEDS for c, x in columns]
    if seeded.index.has_duplicates or not all(key in seeded.index for key in wanted):
        raise ValueError('Seed rows need seeds 42, 14 and 100 for BEACON and scRegNet in every setting')
    permuted = runs[runs.method.eq(PERMUTED) & runs.seed.eq(42)].set_index(['cutoff', 'context'])
    if permuted.index.has_duplicates or not all(key in permuted.index for key in columns):
        raise ValueError('The permuted-embedding block needs one seed-42 scRegNet fit per setting')
    formats = dict(pooled_ap='{:.4f}', mean_ap='{:.4f}', pooled_top1000='{:,.0f}')
    title = lambda text: f'\\multicolumn{{13}}{{@{{}}l}}{{\\textit{{{text}}}}} \\\\'
    lines = []
    for value, text in (('pooled_ap', 'Pooled AP of each fit with optimization seed 42, 14 or 100'),
                        ('mean_ap', 'Mean per-regulator AP of the same fits'),
                        ('pooled_top1000', 'Held-out edges among the 1,000 top-ranked pairs of the same fits')):
        lines.append(title(text))
        for method in SEEDED:
            for seed in SEEDS:
                cells = [formats[value].format(seeded.loc[(method, seed, c, x), value]) for c, x in columns]
                lines.append(f'{method}, seed {seed} & ' + ' & '.join(cells) + ' \\\\')
    lines.append(title('scRegNet with Geneformer embeddings permuted among genes, seed 42'))
    for value, label in (('pooled_ap', 'Pooled AP'), ('mean_ap', 'Per-regulator AP'), ('pooled_top1000', 'Top 1,000')):
        lines.append(f'{label} & ' + ' & '.join(formats[value].format(permuted.loc[key, value]) for key in columns) + ' \\\\')
    return lines


def tables(metrics, paired, runs=None):
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
    head = ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{Cutoff year {c}}}' for c in CUTOFFS) + ' \\\\'
    rule = ''.join(f'\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(3))
    sub = 'Method & ' + ' & '.join(['K562', 'Rest.', 'Stim.', 'RPE1'] * 3) + ' \\\\'
    later = frame[frame.method.eq('BEACON')].set_index(['cutoff', 'context']).positives.reindex(columns)
    counts = 'Held-out edges & ' + ' & '.join(f'{int(v):,}' for v in later) + ' \\\\'
    seeds = [] if runs is None else ['\\midrule', *seed_blocks(runs)]
    main = '\n'.join(['\\toprule', head, rule, sub, '\\midrule', counts, '\\midrule', '\\multicolumn{13}{@{}l}{\\textit{Pooled AP}} \\\\',
                      *block('pooled_ap', 4), '\\midrule', '\\multicolumn{13}{@{}l}{\\textit{Mean per-regulator AP}} \\\\', *block('mean_ap', 4),
                      *seeds, '\\bottomrule'])
    rows = []
    for variant, name in ((PRIMARY, 'All held-out edges'), ('supported by a paper citing <= 50 edges', 'Without edges supported only by large papers'),
                          ('two or more later papers', 'Two or more later papers')):
        wide = metrics[metrics.variant.eq(variant)].pivot_table(index=['cutoff', 'context'], columns='method', values='pooled_ap')
        rank = wide[PUBLISHED].rank(axis=1, ascending=False).BEACON
        counts = outcome(paired, variant, PUBLISHED[1:])
        cell = lambda m: '/'.join(str(int(counts.loc[m, r])) for r in ('higher', 'tie', 'lower'))
        rows.append(f'{name} & {int((rank == 1).sum())} & {int((rank == 2).sum())} & {int((rank >= 3).sum())} & '
                    f'{cell("scRegNet")} & {cell("GCLink")} & {cell("GNNLink")} & '
                    f'{min(int(counts.loc[m, "higher"]) for m in PUBLISHED[1:] if m not in ("scRegNet", "GCLink", "GNNLink"))} \\\\')
    robustness = '\n'.join(['\\toprule', ' & \\multicolumn{3}{c}{BEACON rank by pooled AP} & \\multicolumn{4}{c}{Per-regulator AP: higher/tie/lower} \\\\',
                            '\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}', 'Held-out edges & 1st & 2nd & 3rd or lower & scRegNet & GCLink & GNNLink & Other six, higher (min.) \\\\',
                            '\\midrule', *rows, '\\bottomrule'])
    return main, robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metrics, paired, concentration = load(args.source)
    runs = pd.read_csv(Path(args.source) / 'seeds.csv')
    for name, fig in (('temporal_validation', draw(metrics, paired)), ('temporal_concentration', concentration_figure(concentration))):
        relabel(fig)
        for extension in ('png', 'pdf'):
            fig.savefig(args.out / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    main_table, robustness = tables(metrics, paired, runs)
    (args.out / 'temporal_table.tex').write_text(main_table + '\n')
    (args.out / 'temporal_robustness.tex').write_text(robustness + '\n')


if __name__ == '__main__':
    main()
