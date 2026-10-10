"""Extracted manuscript artists and plotting helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.3,
    'axes.titlesize': 8.2, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.7, 'ytick.labelsize': 6.7,
    'legend.fontsize': 6.4, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .65,
    'ytick.major.width': .65, 'xtick.major.size': 3,
    'ytick.major.size': 3, 'lines.linewidth': 1.15, 'hatch.linewidth': .45,
})

DS = [1501, 1605, 1709, 1801]

SHORT = ['hESC', 'mDC', 'mHSC-E', 'mESC']

CONTEXT = ['hESC\nSTRING reference', 'mDC\nnon-specific ChIP-seq',
           'mHSC-E\nspecific ChIP-seq', 'mESC\nperturbation reference']

TF_COUNT = {1501: 410, 1605: 321, 1709: 205, 1801: 620}

# One order and one style per method in every figure: color, marker, line style, hatch.
ORDER = ['BEACON', 'GNNLink', 'GCLink', 'scRegulate', 'RegGAIN', 'GENELink', 'scRegNet', 'GRNBoost2', 'GENIE3', 'Inferelator',
         'Topology control', 'Prior target in-degree', 'Prior regulator out-degree',
         'Target expression', 'Expression correlation', 'Target detection']
STYLE = {
    'BEACON': ('#0B6E4F', 'o', '-', ''), 'GNNLink': ('#C0480A', 's', '--', ''), 'GCLink': ('#4D7F17', '^', '-.', ''),
    'scRegulate': ('#A67C00', 'D', ':', ''), 'RegGAIN': ('#5A4E9B', 'v', (0, (4, 2)), ''), 'GENELink': ('#B5155E', 'P', '-', ''),
    'scRegNet': ('#159AA8', '8', (0, (3, 1, 1, 1)), ''), 'GRNBoost2': ('#7F4F24', 'h', '-', ''), 'GENIE3': ('#1F5F99', '*', '-', ''), 'Inferelator': ('#B22222', 'd', '-', ''),
    'Topology control': ('#2B2B2B', 'X', (0, (1.5, 1.2)), ''), 'Prior target in-degree': ('#555555', '>', '-.', '///'),
    'Prior regulator out-degree': ('#7A7A7A', '<', ':', '\\\\'), 'Target expression': ('#8F8F8F', 'p', '-', 'xx'),
    'Expression correlation': ('#A8A8A8', 'H', '-', '...'), 'Target detection': ('#6E6E6E', 'p', '-', '++')}

# Published names of the baselines. Data keys keep the internal names, and relabel() renames figure text at save time.
DISPLAY = {'Topology control': 'Degree baseline', 'Degree logistic': 'Degree baseline',
           'Prior target in-degree': 'Target popularity', 'Prior target\nin-degree': 'Target\npopularity',
           'Prior regulator out-degree': 'Regulator popularity', 'Prior regulator\nout-degree': 'Regulator\npopularity'}

def display(text):
    for internal, published in DISPLAY.items():
        text = text.replace(internal, published)
    return text

def relabel(fig):
    """Rename baselines in every text artist and in fixed or categorical tick labels, which are redrawn at save time."""
    from matplotlib.category import StrCategoryFormatter
    from matplotlib.ticker import FixedFormatter, FuncFormatter
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_text():
            artist.set_text(display(artist.get_text()))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for formatter in (axis.get_major_formatter(), axis.get_minor_formatter()):
                if isinstance(formatter, FixedFormatter):
                    formatter.seq = [display(str(label)) for label in formatter.seq]
                elif isinstance(formatter, StrCategoryFormatter):
                    formatter._units = {display(key): value for key, value in formatter._units.items()}
                elif isinstance(formatter, FuncFormatter):
                    formatter.func = lambda x, pos=None, original=formatter.func: display(str(original(x, pos)))

def ordered(methods):
    return sorted(methods, key=ORDER.index)

def row_major(items, ncol):
    """Matplotlib fills legend columns first; reorder so the legend reads left to right in plot order."""
    rows = -(-len(items) // ncol)
    grid = [items[r * ncol:(r + 1) * ncol] for r in range(rows)]
    return [grid[r][c] for c in range(ncol) for r in range(rows) if c < len(grid[r])]

METHOD_STYLE = {m: STYLE[m][:3] for m in ORDER}

TOPOLOGY = 'Topology control'

FIG2_STYLE = {m: STYLE[m][:3] for m in ['BEACON', 'GNNLink', 'GCLink', 'scRegulate', 'RegGAIN', TOPOLOGY,
                                        'Prior target in-degree', 'Prior regulator out-degree']}

def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=6)
    ax.text(-.12, 1.04, letter.upper(), transform=ax.transAxes, weight='bold',
            fontsize=10, va='bottom')

def light_grid(ax, axis='y'):
    ax.grid(axis=axis, color='#DDDDDD', lw=.45)
    ax.set_axisbelow(True)

def label_panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.16,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=11)


def prior_completion(coverage, primary, save):
    # Figure 2: coverage and raw revised benchmark metrics.
    fig = plt.figure(figsize=(7.15, 4.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, .95], left=.07, right=.99,
                          bottom=.15, top=.96, hspace=.45)
    top_gs = gs[0].subgridspec(1, 4, wspace=.24)
    bottom_gs = gs[1].subgridspec(1, 2, wspace=.2)
    for j, (dataset, title) in enumerate(zip(DS, CONTEXT)):
        ax = fig.add_subplot(top_gs[j])
        for label in ['BEACON',TOPOLOGY,'GNNLink','RegGAIN']:
            part = coverage[(coverage.dataset_id == dataset) & (coverage.label == label)]
            stats = part.groupby('coverage').auprc_trapezoid.agg(['mean','min','max'])
            color, marker, ls = FIG2_STYLE[label]
            ax.plot(stats.index*100, stats['mean'], marker=marker, ms=3.8,
                    color=color, ls=ls, label=label)
            ax.fill_between(stats.index*100, stats['min'], stats['max'], color=color, alpha=.10)
        ax.set_title(title, pad=3); ax.set_xticks([5,20,40,80]); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(4)); light_grid(ax)
        if j == 0: ax.set_ylabel('AUPRC')
        ax.set_xlabel('Prior revealed (%)')
        ax.text(.02,.97,chr(ord('A')+j),transform=ax.transAxes,weight='bold',va='top',fontsize=9)
    axes = [fig.add_subplot(bottom_gs[0]), fig.add_subplot(bottom_gs[1])]
    methods = list(FIG2_STYLE)
    for ax, metric, title, letter, ylabel in zip(axes, ['auprc_trapezoid', 'all_tf_ap'],
                                                 ['80% prior: pooled AUPRC', '80% prior: all-TF mean AP'], ['e', 'f'],
                                                 ['Pooled AUPRC', 'All-TF mean AP']):
        series, errors = {}, {}
        for m in methods:
            means, spans = [], []
            for dataset in DS:
                vals = primary[(primary.dataset == dataset) & (primary.label == m)][metric]
                assert len(vals) == 3
                means.append(vals.mean()); spans.append((vals.mean() - vals.min(), vals.max() - vals.mean()))
            series[m], errors[m] = means, np.array(spans).T
        grouped_bars(ax, SHORT, series, errors=errors)
        ax.set_ylabel(ylabel); panel(ax, letter, title)
    fig2_handles = row_major([Line2D([0],[0],color=FIG2_STYLE[m][0],marker=FIG2_STYLE[m][1],
                           ls=FIG2_STYLE[m][2],label=m,ms=4) for m in FIG2_STYLE], 4)
    fig.legend(handles=fig2_handles, frameon=False, ncol=6, loc='lower center',
               bbox_to_anchor=(.5,-.035))
    save(fig, 'prior_completion')


    return fig


def component_evidence(components, ablation, comparisons, save):
    """Ablation heatmaps shaded relative to BEACON, and paired K562 differences for the expression controls."""
    fig = plt.figure(figsize=(7.15, 4.0), layout='constrained')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.03, wspace=.03, hspace=.05)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 1], width_ratios=[1, 1, .045])
    cmap = sns.color_palette('rocket', as_cmap=True)
    ablations = sum(not label.startswith('Prior') for _, _, label in components)
    for j, (metric, title) in enumerate([('auprc_trapezoid', 'Pooled AUPRC at 80% prior'), ('all_tf_ap', 'All-TF mean AP at 80% prior')]):
        ax = fig.add_subplot(gs[0, j])
        values = []
        for variant, method, _ in components:
            part = ablation[(ablation.variant == variant) & (ablation.method == method)]
            assert len(part) == 12 and part.groupby('dataset_id').size().eq(3).all()
            values.append([part[part.dataset_id == d][metric].mean() for d in DS])
        values = np.array(values)
        relative = values / values[0]
        image = ax.imshow(relative, cmap=cmap, aspect='auto', vmin=0, vmax=1.1)
        ax.set_xticks(range(4), SHORT)
        ax.set_yticks(range(len(components)), [label for _, _, label in components] if j == 0 else [])
        ax.tick_params(length=0)
        for r in range(len(components)):
            for c in range(4):
                ax.text(c, r, f'{values[r, c]:.3f}', ha='center', va='center', fontsize=6.6,
                        color='black' if relative[r, c] > .6 else 'white')
        ax.axhline(ablations - .5, color='white', lw=3)
        for side in ['left', 'bottom']: ax.spines[side].set_visible(False)
        panel(ax, chr(ord('a') + j), title)
    fig.colorbar(image, cax=fig.add_subplot(gs[0, 2]), label='Value relative to BEACON')
    ax = fig.add_subplot(gs[1, :2])
    rows = [(key, label) for key, label in (('graph_only', 'w/o expression (random features)'),
                                            ('permuted', 'w/o expression (genes permuted)'),
                                            ('Prior target in-degree', 'Prior target in-degree'))]
    for y, (key, label) in zip(range(len(rows))[::-1], rows):
        row = comparisons.loc[key]
        mean, low, high = row.beacon_minus_comparator_macro_ap, row.ci_low, row.ci_high
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt='o', color=METHOD_STYLE['BEACON'][0],
                    ms=4.5, capsize=2.5, lw=1)
    ax.axvline(0, color='#777777', ls='--', lw=.8)
    ax.set_yticks(range(len(rows))[::-1], [label for _, label in rows])
    ax.set_ylim(-.6, len(rows) - .4)
    span = max(abs(v) for key, _ in rows for v in comparisons.loc[key, ['ci_low', 'ci_high']]) * 1.25
    ax.set_xlim(-span, span)
    ax.set_xlabel('Mean per-regulator response AP, BEACON minus comparator')
    ax.text(.99, .04, 'favors BEACON →', transform=ax.transAxes, ha='right', fontsize=6.3, color='#555555')
    ax.text(.01, .04, '← favors comparator', transform=ax.transAxes, ha='left', fontsize=6.3, color='#555555')
    light_grid(ax, 'x')
    panel(ax, 'c', 'K562 perturbation response, 478 regulators')
    save(fig, 'component_evidence')
    return fig


EVAL_STYLE = {m: (STYLE[m][0], STYLE[m][3]) for m in ORDER}


def grouped_bars(ax, groups, series, errors=None, chance=None, labels=False, width=.84):
    """One bar per method within each group; dashed segments mark the chance level of each group."""
    n = len(series)
    for j, (method, values) in enumerate(series.items()):
        color, hatch = EVAL_STYLE[method]
        x = np.arange(len(groups)) + (j - (n - 1) / 2) * width / n
        bars = ax.bar(x, values, width / n * .9, color=color, hatch=hatch, edgecolor='#222222', linewidth=.35, label=method)
        if errors is not None and method in errors:
            ax.errorbar(x, values, yerr=errors[method], fmt='none', ecolor='#222222', elinewidth=.55, capsize=1.1)
        if labels:
            ax.bar_label(bars, fmt='%g', fontsize=4.8, padding=1)
    ax.set_xticks(range(len(groups)), groups)
    if chance is not None:
        for x, level in enumerate(np.broadcast_to(chance, (len(groups),))):
            ax.plot([x - width / 2, x + width / 2], [level, level], color='#222222', ls='--', lw=.8)
    ax.tick_params(axis='x', length=0)
    light_grid(ax)


def legend_below(fig, methods, ncol, y):
    handles = row_major([Patch(facecolor=EVAL_STYLE[m][0], hatch=EVAL_STYLE[m][1], edgecolor='#222222', linewidth=.35, label=m) for m in ordered(methods)], ncol)
    fig.legend(handles=handles, ncol=ncol, frameon=False, loc='lower center', bbox_to_anchor=(.5, y))


def k562_evaluation(fractions, ratios, ratio_errors, relative_ap, dnmt1, expected, chance, ks, methods):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.9), layout='constrained')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.03, wspace=.04, hspace=.06)
    groups = [f'Top {k}' for k in ks]
    grouped_bars(axes[0, 0], groups, {m: fractions[m] for m in methods}, chance=chance)
    axes[0, 0].set_ylabel('Response-supported fraction')
    panel(axes[0, 0], 'a', 'Responses among top-ranked targets')
    grouped_bars(axes[0, 1], groups, {m: ratios[m] for m in methods}, errors=ratio_errors, chance=1)
    axes[0, 1].set_ylabel('Observed / expression-matched')
    panel(axes[0, 1], 'b', 'Enrichment beyond target expression')
    grouped_bars(axes[1, 0], ['Response', 'Promoter binding', 'Both'], {m: relative_ap[m] for m in methods}, chance=1)
    axes[1, 0].set_ylabel('Mean AP / chance')
    panel(axes[1, 0], 'c', 'Average precision over full target lists')
    grouped_bars(axes[1, 1], ['Response', 'Promoter binding', 'Both'], {m: dnmt1[m] for m in methods}, chance=expected, labels=True)
    axes[1, 1].set_ylabel('Supported targets in top 100')
    axes[1, 1].set_ylim(0, max(max(v) for v in dnmt1.values()) * 1.15)
    panel(axes[1, 1], 'd', 'DNMT1 top 100 targets')
    legend_below(fig, methods, ncol=6, y=-.055)
    return fig


def cross_context(tcells, matched, matched_errors, rpe1, trrust, trrust_chance, legend):
    fig = plt.figure(figsize=(7.15, 7.0), layout='constrained')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.03, wspace=.04, hspace=.06)
    gs = fig.add_gridspec(3, 2)
    for j, (condition, title) in enumerate((('resting', 'Primary resting T cells'), ('stimulated', 'Primary re-stimulated T cells'))):
        ax = fig.add_subplot(gs[0, j])
        groups, series, chance = tcells[condition]
        grouped_bars(ax, groups, series, chance=chance)
        ax.set_ylabel('Response AP')
        panel(ax, 'ab'[j], title)
    ax = fig.add_subplot(gs[1, 0])
    groups, series = matched
    grouped_bars(ax, groups, series, errors=matched_errors, chance=1)
    ax.set_ylabel('Observed / matched, top 100')
    panel(ax, 'c', 'Enrichment beyond target expression')
    ax = fig.add_subplot(gs[1, 1])
    means, errors = rpe1
    for x, method in enumerate(means):
        color, hatch = EVAL_STYLE[method]
        ax.bar(x, means[method], .7, color=color, hatch=hatch, edgecolor='#222222', linewidth=.35)
        ax.errorbar(x, means[method], yerr=np.array(errors[method])[:, None], fmt='none', ecolor='#222222', elinewidth=.55, capsize=1.5)
    ax.set_xticks(range(len(means)), list(means), rotation=30, ha='right', rotation_mode='anchor', fontsize=6.2)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Mean top-100 response (control SD)')
    light_grid(ax)
    panel(ax, 'd', 'RPE1 response magnitude')
    for j, (metric, title, label, chance) in enumerate((('auroc', 'TRRUST same-regulator AUROC', 'AUROC', .5),
                                                         ('average_precision', 'TRRUST regulator-weighted AP', 'Average precision', trrust_chance))):
        ax = fig.add_subplot(gs[2, j])
        grouped_bars(ax, ['All supported pairs', 'DoRothEA overlap removed'], trrust[metric], chance=chance)
        if metric == 'auroc':
            ax.set_ylim(.4, None)
        ax.set_ylabel(label)
        panel(ax, 'ef'[j], title)
    legend_below(fig, legend, ncol=6, y=-.05)
    return fig


def runtime_scaling(scaling, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), layout='constrained')
    for ax, xcol, ycol, title, letter, xlabel, ylabel in [
        (axes[0], 'sampled_training_pairs', 'model_fit_seconds', 'Model fitting', 'a',
         'Sampled training pairs', 'Encoder + GP fit time (s)'),
        (axes[1], 'test_pairs', 'other_seconds', 'Loading, factor analysis and scoring', 'b',
         'Scored test pairs', 'Time outside model fitting (s)')]:
        x = scaling[xcol].to_numpy(float)
        y = scaling[ycol].to_numpy(float)
        ax.scatter(x, y, s=16, facecolors='none', edgecolors='#222222', linewidths=.55)
        coef = np.polyfit(np.log10(x), np.log10(y), 1)
        guide_x = np.geomspace(x.min(), x.max(), 100)
        ax.plot(guide_x, 10 ** np.polyval(coef, np.log10(guide_x)), color='#777777',
                linestyle='--', linewidth=.9, label=f'log-log slope {coef[0]:.2f}')
        ax.legend(frameon=False, loc='lower right')
        # The time outside fitting varies within 10-20 s, so panel b uses a linear axis from zero.
        ax.set(xscale='log', yscale='log' if letter == 'a' else 'linear', xlabel=xlabel, ylabel=ylabel)
        if letter == 'b': ax.set_ylim(0, 1.15 * y.max())
        light_grid(ax, 'both')
        panel(ax, letter, title)
        ax.text(.03, .95, f'{len(scaling)} sampled-pair runs', transform=ax.transAxes, va='top', fontsize=6.4)
    fig.text(.5, 1.01, 'Empirical resource use of the BEACON model on one NVIDIA L40S GPU',
             ha='center', va='bottom', fontsize=7.1)
    save(fig, 'runtime_scaling')


def completion_sensitivity(h, match, grid, COLORS, NAMES, contextcolors, panel, save):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    ax=axes[0,0]
    for j,method in enumerate(['beacon','degree_logistic','gnnlink','reggain']):
     p=h[h.method==method].groupby('dataset').pair_concordance.mean().reindex(DS);ax.plot(np.arange(4)+(j-1.5)*.1,p,'o',color=COLORS[NAMES[method]],label=NAMES[method],ms=4)
    ax.axhline(.5,color='black',ls=':',lw=.8);ax.set_xticks(range(4),SHORT,rotation=25,ha='right');ax.set_ylim(.45,.85);ax.set_ylabel('Within-pair concordance');ax.legend(frameon=False,fontsize=6,ncol=2);panel(ax,'a','Same-TF matched targets')
    ax=axes[0,1];m=match.groupby('dataset').fraction_matched.agg(['mean','min','max']).reindex(DS);ax.bar(range(4),m['mean'],color=contextcolors);ax.errorbar(range(4),m['mean'],yerr=np.vstack([m['mean']-m['min'],m['max']-m['mean']]),fmt='none',color='black',lw=.7,capsize=2);ax.set_xticks(range(4),SHORT,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Fraction of positives matched');panel(ax,'b','Coverage of the challenge')
    for ax,variable,title,letter in [(axes[1,0],'corruption','Corrupted prior at 20% coverage','c'),(axes[1,1],'ratio','Training unlabeled ratio at 20% coverage','d')]:
     for d,label,color in zip(DS,SHORT,contextcolors):
      p=grid[(grid.dataset_id==d)&(grid.coverage==.2)&(grid.control=='beacon')&(grid.method=='beacon')]
      p=p[p.ratio==5] if variable=='corruption' else p[p.corruption==0]
      g=p.groupby(variable).average_precision.agg(['mean','min','max']);x=g.index.to_numpy()*(100 if variable=='corruption' else 1);ax.plot(x,g['mean'],color=color,marker='o',ms=3,label=label);ax.fill_between(x,g['min'],g['max'],color=color,alpha=.1)
     ax.set_xlabel('Prior replacement (%)' if variable=='corruption' else 'Unlabeled pairs per positive');ax.set_ylabel('Pooled AP');panel(ax,letter,title)
    axes[1,0].legend(frameon=False,fontsize=6,ncol=2);save(fig,'completion_sensitivity')


def shortcut(comparison, audit):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.8))
    colors={'STRING':'#0072B2','Non-Specific':'#D55E00','Specific':'#009E73','Lofgof':'#CC79A7'}
    for ax,metric,title in zip(axes[:2],['auprc_trapezoid','macro_auroc'],['a  Sampled-pool AUPRC','b  Within-regulator AUROC']):
        for family,color in colors.items():
            mask=comparison.family.eq(family)
            ax.scatter(comparison.loc[mask,f'beacon_{metric}'],comparison.loc[mask,f'degree_{metric}'],label=family,s=27,color=color,alpha=.85)
        ax.plot([0,1],[0,1],color='.5',lw=.9,ls='--')
        ax.set(xlim=(.4,1.01),ylim=(.4,1.01),xlabel='BEACON',ylabel='Training-degree logistic',title=title)
        ax.set_aspect('equal')
    axes[0].legend(fontsize=8,loc='lower right',frameon=False)
    fraction=audit.test_unlabeled_source_absent_from_train/audit.test_unlabeled
    for family,color in colors.items():
        ids=comparison.loc[comparison.family.eq(family),'dataset_id']
        mask=audit.dataset_id.isin(ids)
        axes[2].scatter(fraction[mask],comparison.set_index('dataset_id').loc[audit.loc[mask,'dataset_id'],'degree_auprc_trapezoid'],s=27,color=color,alpha=.85)
    axes[2].set(xlim=(0,1),ylim=(.4,1.01),xlabel='Unlabeled test pairs with a source\nabsent from the training prior',ylabel='Training-degree logistic AUPRC',title='c  Source-role imbalance')
    fig.tight_layout()

    return fig, axes


def specificity_regimes(fixed, external):
    """Regulator-specific gain (top) and within-regulator AP (bottom) against revealed edges per regulator in each fixed pool.
    Lines join the coverage levels of methods fitted at every coverage; markers show methods run at 80% coverage. The band
    marks the perturbation-assay priors."""
    curves = ['RegGAIN', 'Topology control', 'GNNLink', 'BEACON']
    points = ['GCLink', 'scRegulate', 'GENELink', 'scRegNet', 'GRNBoost2', 'GENIE3', 'Inferelator']
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.3), layout='constrained', sharex='col', sharey='row')
    low, high = external.revealed_per_regulator.min(), external.revealed_per_regulator.max()
    for row, (metric, label) in enumerate((('gain', 'Within-regulator AP /\ntarget-only AP'), ('within_ap', 'Within-regulator AP'))):
        for ax, context in zip(axes[row], SHORT):
            part = fixed[fixed.context.eq(context)]
            ax.axvspan(low / 1.12, high * 1.12, color='#E3E3E3', lw=0, zorder=0)
            if metric == 'gain':
                ax.axhline(1, color='#777777', lw=.6, ls='--', zorder=1)
            for method in curves:
                g = (part[part.method.eq(method)].groupby('coverage')
                     .agg(x=('revealed_per_regulator', 'mean'), y=(metric, 'mean'), low=(metric, 'min'), high=(metric, 'max')))
                color, marker, style = STYLE[method][:3]
                ax.errorbar(g.x, g.y, yerr=[g.y - g.low, g.high - g.y], color=color, ls=style, marker=marker, ms=3,
                            lw=1 if method == 'BEACON' else .8, elinewidth=.5, capsize=1.2, zorder=4 if method == 'BEACON' else 3)
            top = part[part.coverage.eq(.8)]
            x80 = top.revealed_per_regulator.mean()
            present = [m for m in points if m in set(top.method)]
            for k, method in enumerate(present):
                g = top[top.method.eq(method)][metric]
                x = x80 * 1.12 ** (k - (len(present) - 1) / 2 + .5)
                color, marker = STYLE[method][:2]
                ax.errorbar([x], [g.mean()], yerr=[[g.mean() - g.min()], [g.max() - g.mean()]], color=color, marker=marker, ms=3.2, ls='',
                            elinewidth=.5, capsize=1.2, mfc='white', zorder=2)
            ax.set_xscale('log'); ax.set_xticks([.3, 1, 3, 10, 30], ['0.3', '1', '3', '10', '30']); ax.minorticks_off()
            if row == 0:
                ax.set_title(context, fontsize=8)
            else:
                ax.set_xlabel('Revealed edges per regulator', fontsize=6.8)
        axes[row, 0].set_ylabel(label)
        axes[row, 0].text(-.42, 1.04, 'AB'[row], transform=axes[row, 0].transAxes, fontsize=10, weight='bold', va='bottom')
    axes[0, 0].text((low * high) ** .5, axes[0, 0].get_ylim()[1], 'Perturbation-\nassay priors', ha='center', va='top', fontsize=5.6, color='#444444')
    order = ['BEACON', 'GNNLink', 'RegGAIN', 'Topology control'] + points
    handles = [Line2D([], [], color=STYLE[m][0], ls=STYLE[m][2] if m in curves else '', marker=STYLE[m][1], ms=3.2,
                      mfc=STYLE[m][0] if m in curves else 'white', label=m) for m in order]
    fig.legend(handles=row_major(handles, 6), loc='outside lower center', ncol=6, frameon=False, fontsize=6.2, handlelength=1.8, columnspacing=1.0)
    return fig
