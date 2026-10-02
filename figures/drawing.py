"""Extracted manuscript artists and plotting helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    'ytick.major.size': 3, 'lines.linewidth': 1.15,
})

DS = [1501, 1605, 1709, 1801]

SHORT = ['hESC', 'mDC', 'mHSC-E', 'mESC']

CONTEXT = ['hESC\nSTRING reference', 'mDC\nnon-specific ChIP-seq',
           'mHSC-E\nspecific ChIP-seq', 'mESC\nperturbation reference']

TF_COUNT = {1501: 410, 1605: 321, 1709: 205, 1801: 620}

METHOD_STYLE = {
    'BEACON': ('#1B9E77', 'o', '-'),
    'GNNLink': ('#D95F02', 's', '--'),
    'GCLink': ('#66A61E', '^', '-.'),
    'scRegulate': ('#E6AB02', 'D', ':'),
    'RegGAIN': ('#7570B3', 'v', (0, (4, 2))),
}

TOPOLOGY = 'Topology control'

FIG2_STYLE = {'BEACON': METHOD_STYLE['BEACON'], TOPOLOGY: ('#555555', 'X', (0, (1.5, 1.2))),
              **{m: METHOD_STYLE[m] for m in ['GNNLink', 'GCLink', 'scRegulate', 'RegGAIN']}}

def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=6)
    ax.text(-.12, 1.04, letter.upper(), transform=ax.transAxes, weight='bold',
            fontsize=10, va='bottom')

def light_grid(ax, axis='y'):
    ax.grid(axis=axis, color='#DDDDDD', lw=.45)
    ax.set_axisbelow(True)

def paired_scatter(ax, frame, xmethod, ymethod, metric, title, xlabel, ylabel,
                   note=None, sqrt_axes=False):
    wide = frame.pivot(index='tf', columns='method', values=metric).dropna()
    x, y = wide[xmethod].to_numpy(), wide[ymethod].to_numpy()
    plot_x, plot_y = (np.sqrt(x), np.sqrt(y)) if sqrt_axes else (x, y)
    high = max(float(plot_x.max()), float(plot_y.max())) * 1.04
    ax.scatter(plot_x, plot_y, s=11, facecolors='none', edgecolors='#333333',
               linewidths=.45, alpha=.65)
    ax.plot([0, high], [0, high], color='#888888', ls='--', lw=.8)
    ax.set(xlim=(0, high), ylim=(0, high), xlabel=xlabel, ylabel=ylabel)
    if sqrt_axes:
        ticks = np.array([0, .04, .25, 1.0])
        ticks = ticks[np.sqrt(ticks) <= high]
        ax.set_xticks(np.sqrt(ticks), [f'{v:g}' for v in ticks])
        ax.set_yticks(np.sqrt(ticks), [f'{v:g}' for v in ticks])
    ax.set_aspect('equal', adjustable='box')
    if note is None:
        note = f'n={len(wide)}, BEACON higher: {(y>x).sum()}'
    ax.text(.03, .88, note, transform=ax.transAxes, va='top', fontsize=5.8,
            bbox=dict(facecolor='white', edgecolor='none', pad=.4, alpha=.85))
    panel(ax, title[0], title[2:])

def paired_difference_strip(ax, frame, control, comparison, title):
    wide = frame[frame.method.isin(['BEACON', control])].pivot(
        index='tf', columns='method', values='average_precision_all_tf').dropna()
    delta = (wide.BEACON - wide[control]).to_numpy()
    limit = .04
    shown = np.clip(delta, -limit, limit)
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-.14, .14, len(delta))
    inside = np.abs(delta) <= limit
    ax.scatter(shown[inside], jitter[inside], s=9, facecolors='none',
               edgecolors=METHOD_STYLE['BEACON'][0], linewidths=.45, alpha=.65)
    for side, marker in [(delta > limit, '>'), (delta < -limit, '<')]:
        if side.any():
            ax.scatter(shown[side], jitter[side], s=16, marker=marker,
                       color=METHOD_STYLE['BEACON'][0], linewidths=.3)
    mean = comparison.beacon_minus_comparator_macro_ap
    low, high = comparison.ci_low, comparison.ci_high
    ax.errorbar(mean, .27, xerr=[[mean-low], [high-mean]], fmt='o',
                color='#222222', capsize=2, ms=3.5, lw=.8)
    positive, tied = int((delta > 0).sum()), int((delta == 0).sum())
    ax.text(.03, .97, f'n={len(delta)}, positive {positive}, tied {tied}',
            transform=ax.transAxes, va='top', fontsize=6.2)
    off, below = int((delta > limit).sum()), int((delta < -limit).sum())
    print(f'{control}: n={len(delta)}, clipped above {limit}: {off}, below -{limit}: {below}, '
          f'max {delta.max():.4f}, min {delta.min():.4f}, mean {delta.mean():.5f} [{low:.5f}, {high:.5f}]', flush=True)
    clipped = []
    if off: clipped.append(f'{off} above 0.04 (max {delta.max():.3f})')
    if below: clipped.append(f'{below} below −0.04 (min {delta.min():.3f})'.replace('-', '−'))
    if clipped:
        ax.text(.03, .88, 'Clipped: ' + '\n'.join(clipped), transform=ax.transAxes, va='top', fontsize=5.9)
    ax.axvline(0, color='#777777', ls='--', lw=.75)
    ax.set(xlim=(-.042, .042), ylim=(-.19, .52), yticks=[],
           xlabel='BEACON − control AP')
    light_grid(ax, 'x')
    panel(ax, title[0], title[2:])

def trrust_dumbbell(ax, all_values, novel_values, title, xlabel):
    y=np.arange(len(methods))[::-1]
    for yy,m in zip(y,methods):
        ax.plot([all_values[m],novel_values[m]],[yy,yy],color='#BBBBBB',lw=.8)
        ax.scatter(all_values[m],yy,color=method_colors[m],marker='o',s=19,zorder=3)
        ax.scatter(novel_values[m],yy,facecolors='white',edgecolors=method_colors[m],marker='D',s=18,zorder=3)
    ax.set_yticks(y,methods); ax.set_ylim(-.6, len(methods)-.4); ax.set_xlabel(xlabel); light_grid(ax,'x'); panel(ax,title[0],title[2:])

def label_panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.16,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=11)


def prior_completion(coverage, primary, save):
    # Figure 2: coverage and raw revised benchmark metrics.
    fig = plt.figure(figsize=(7.15, 3.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, .70], left=.07, right=.99,
                          bottom=.18, top=.96, hspace=.52)
    top_gs = gs[0].subgridspec(1, 4, wspace=.24)
    bottom_gs = gs[1].subgridspec(1, 2, wspace=.36)
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
    for ax, metric, title, letter in zip(axes, ['auprc_trapezoid','all_tf_ap'],
                                         ['80% prior: pooled AUPRC','80% prior: all-TF mean AP'], ['e','f']):
        y = np.arange(4)[::-1]
        for offset, label in zip(np.linspace(.27,-.27,len(FIG2_STYLE)), FIG2_STYLE):
            means=[]; lows=[]; highs=[]
            for dataset in DS:
                vals = primary[(primary.dataset==dataset)&(primary.label==label)][metric]
                assert len(vals) == 3
                mean=vals.mean(); means.append(mean); lows.append(mean-vals.min()); highs.append(vals.max()-mean)
            color, marker, _ = FIG2_STYLE[label]
            ax.errorbar(means, y+offset, xerr=np.array([lows,highs]), fmt=marker,
                        color=color, markeredgecolor='#222222', markeredgewidth=.35,
                        ms=4.5, capsize=1.3, lw=.65, label=label)
        ax.set_yticks(y, SHORT); ax.set_xlim(left=0); ax.xaxis.set_major_locator(MaxNLocator(4))
        right = ax.get_xlim()[1]
        ax.set_xlim(-.035 * right, right)
        light_grid(ax,'x'); panel(ax,letter,title)
    fig2_handles = [Line2D([0],[0],color=FIG2_STYLE[m][0],marker=FIG2_STYLE[m][1],
                           ls=FIG2_STYLE[m][2],label=m,ms=4) for m in FIG2_STYLE]
    fig.legend(handles=fig2_handles, frameon=False, ncol=6, loc='lower center',
               bbox_to_anchor=(.5,-.035))
    save(fig, 'prior_completion')


    return fig


def component_evidence(components, ablation, resp_controls, control_comparisons, save):
    fig=plt.figure(figsize=(7.15,4.3),layout='constrained')
    fig.get_layout_engine().set(w_pad=.01, h_pad=.03, wspace=.02, hspace=.04)
    gs=fig.add_gridspec(2,2,height_ratios=[1.55,1])
    beeline_cmap=sns.color_palette('rocket',as_cmap=True)
    for j,(metric,title) in enumerate([('auprc_trapezoid','Pooled AUPRC at 80% prior'),('all_tf_ap','All-TF mean AP at 80% prior')]):
        ax=fig.add_subplot(gs[0,j])
        mat=[]
        for variant,method,_ in components:
            part=ablation[(ablation.variant==variant)&(ablation.method==method)]
            assert len(part)==12 and part.groupby('dataset_id').size().eq(3).all()
            mat.append([part[part.dataset_id==d][metric].mean() for d in DS])
        mat=np.array(mat)
        ax.imshow(mat,cmap=beeline_cmap,aspect='auto',vmin=0,vmax=max(.01,mat.max()))
        ax.set_xticks(range(4),SHORT)
        ax.set_yticks(range(len(components)),[c[2] for c in components])
        ax.tick_params(length=0)
        for r in range(len(components)):
            for c in range(4):
                ax.text(c,r,f'{mat[r,c]:.3f}',ha='center',va='center',fontsize=6.8,
                        color='black' if mat[r,c]>.58*mat.max() else 'white')
        for side in ['left','bottom']: ax.spines[side].set_visible(False)
        panel(ax,chr(ord('a')+j),title)
    for j,(control,title) in enumerate([('graph_only','K562 response: random node features'),('permuted','K562 response: permuted features')]):
        ax=fig.add_subplot(gs[1,j])
        paired_difference_strip(ax, resp_controls, control,
                                control_comparisons.loc[control],
                                f'{chr(ord("c")+j)} {title}')
    save(fig,'component_evidence')


    return fig


def experimental_validation(response_agg, response_tf, binding_agg, binding_tf, response_curves, binding_curves, pub, ks):
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.6),layout='constrained')
    fig.get_layout_engine().set(w_pad=.015, h_pad=.02, wspace=.015, hspace=.035)
    resp_summary=response_agg.set_index('method')
    resp_note=(f'Mean AP\nBEACON {resp_summary.loc["BEACON","macro_ap_all_tf"]:.5f}'
               f'\nGNNLink {resp_summary.loc["GNNLink","macro_ap_all_tf"]:.5f}')
    paired_scatter(axes[0,0],response_tf[response_tf.method.isin(['BEACON','GNNLink'])],
                   'GNNLink','BEACON','average_precision_all_tf','a K562 perturbation response',
                   'GNNLink AP','BEACON AP',resp_note, sqrt_axes=True)
    binding_summary=binding_agg.query('assay == "binding"').set_index('method')
    binding_note=(f'Mean AP\nBEACON {binding_summary.loc["BEACON","macro_ap"]:.3f}'
                  f'\nGNNLink {binding_summary.loc["GNNLink","macro_ap"]:.3f}')
    paired_scatter(axes[0,1],binding_tf.query('assay == "binding" and method in @pub'),
                   'GNNLink','BEACON','average_precision','b K562 promoter binding',
                   'GNNLink AP','BEACON AP',binding_note)
    joint_summary=binding_agg.query('assay == "binding_and_response"').set_index('method')
    joint_note=(f'Mean AP\nBEACON {joint_summary.loc["BEACON","macro_ap"]:.5f}'
                f'\nGNNLink {joint_summary.loc["GNNLink","macro_ap"]:.5f}')
    paired_scatter(axes[0,2],binding_tf.query('assay == "binding_and_response" and method in @pub'),
                   'GNNLink','BEACON','average_precision','c K562 joint assay support',
                   'GNNLink AP','BEACON AP',joint_note)
    for ax,curves,title,letter in [(axes[1,0],response_curves,'K562 response at top K','d'),
                                   (axes[1,1],binding_curves,'K562 binding at top K','e')]:
        for method in pub:
            color,marker,ls=METHOD_STYLE[method]
            ax.plot(ks,[curves[method][k] for k in ks],color=color,marker=marker,ls=ls,ms=3.6,label=method)
        ax.set_xscale('log'); ax.set_xticks(ks,[str(k) for k in ks]); ax.set_xlabel('Top K targets')
        ax.set_ylabel('Mean supported fraction')
        if letter == 'e':
            shown = np.array([[curves[method][k] for k in ks] for method in pub])
            pad = .12 * (shown.max() - shown.min())
            ax.set_ylim(shown.min() - pad, shown.max() + pad)
        else:
            ax.set_ylim(bottom=0)
        light_grid(ax); panel(ax,letter,title)

    return fig, axes


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
