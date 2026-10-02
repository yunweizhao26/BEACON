"""Manuscript figure artists, reading only the configured bundle."""
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from beacon.data import RELEASE
from figures import drawing
from figures.build import Build, POP, CONTEXT
FIGURES = RELEASE / "results" / "figures"
ROOT = Path(".")
def namespace():
    env = drawing.__dict__
    env["METHOD_STYLE"].update(STYLES)
    env["FIG2_STYLE"].update(STYLES)
    return env
def save(build, fig, name, metric, notes, provisional=False):
    build.save(fig, name, metric, notes)
def tcell_evaluations(build):
    return build.tcells()
STYLES = {POP[0]: ("#0072B2", "^", "-."), POP[1]: ("#CC79A7", "v", ":")}
PUB = ["BEACON", "GNNLink", "RegGAIN", POP[0]]
KS = [10, 25, 50, 100, 200]
COMPONENTS = [('beacon', 'beacon', 'BEACON'), ('beacon', 'decoder', 'Pair decoder'),
              ('beacon', 'learned_logistic', 'Logistic on learned\nrepresentations'),
              ('beacon_snn', 'beacon', 'BEACON + SNN objective'),
              ('graph_only', 'graph_only', 'Random node features'), ('permuted', 'permuted', 'Permuted features'),
              ('fa_gp', 'fa_gp', 'GP without encoder'), *[('beacon', m, m) for m in POP]]
RPE_NAMES = {"beacon_model": "BEACON", "gnnlink": "GNNLink", "reggain": "RegGAIN",
             "inferelator_tfa_bbsr_5_bootstraps": "Inferelator", "degree_logistic": "Topology control",
             "expression_abs_correlation": "Expression correlation", "target_control_detection": "Target detection", POP[0]: POP[0]}
TIED_AUPRC_NOTE = ("Tied constant scores inflate trapezoid AUPRC for prior controls "
                   "(up to 0.105 for target in-degree, DS1514); this area is not enrichment.")

def architecture(build):
    from figures import architecture as arch
    # Capture its existing artists before rendering: preserve boxes, colors and arrows.
    arch.OUT = build.directory
    arch.GP_PARAMS = 500 * 32 + 500 + 500 * 501 // 2 + 7
    with patch.object(Figure, "savefig", lambda *args, **kwargs: None):
        arch.main()
    fig = plt.gcf()
    found = set()
    for text in fig.findobj(matplotlib.text.Text):
        value = text.get_text()
        if "Minibatch: 32 observed" in value:
            text.set_text("Minibatch: 32 observed + 160 sampled unlabeled pairs · AdamW · 100 encoder epochs\n"
                          "Shared MLP and pair decoder: class-weighted BCE only; no contrastive / SNN loss")
            found.add("encoder")
        elif value.startswith("kernel $"):
            text.set_text(r"$k = k_{reg} + k_{tgt} + k_{pair}$" + "\nM = 500 inducing inputs\n" + r"$q(f_M) = \mathcal{N}(m,S)$")
            found.add("kernel")
        elif value.startswith("Fitted on the same labeled pairs"):
            text.set_text("Internal 10% split selects GP training length by early stopping\n"
                          "Refit encoder and GP on all training pairs; Bernoulli likelihood with probit link\n"
                          "25-epoch GP fallback when the internal split has too few positives (see configuration audit)")
            found.add("refit")
    if found != {"encoder", "kernel", "refit"}:
        raise ValueError(f"Architecture artist anchors changed: {found}")
    save(build, fig, "beacon_overview", "schematic (no empirical metric)",
         "Reuses architecture boxes and arrows. Shared MLP, pair-decoder class-weighted BCE, additive GP, "
         "500 inducing points, internal 10% stopping split and refit replace the old joint-kernel/50-epoch description.")

def assays(build):
    root = build.assays
    response = build.csv(root / "external/k562_response_evaluation/per_tf_metrics.csv").query('definition == "configured"').copy()
    response["average_precision_all_tf"] = response.average_precision
    comparisons = build.csv(root / "external/k562_response_evaluation/paired_comparisons.csv").query('definition == "configured"').set_index("comparator")
    binding = build.csv(root / "external/k562_binding_evaluation/per_tf_metrics.csv").query('window == "primary"')
    ra = build.csv(root / "external/k562_response_evaluation/aggregate.csv").query('definition == "configured"')
    ba = build.csv(root / "external/k562_binding_evaluation/aggregate.csv").query('window == "primary"').copy()
    ba["macro_ap"] = ba.macro_ap_all_tf
    wide = response.pivot(index="tf", columns="method", values="average_precision")
    if wide.isna().any().any():
        raise ValueError("Incomplete K562 paired regulator cohort")
    for method, row in comparisons.iterrows():
        if not np.isclose((wide.BEACON - wide[method]).mean(), row.beacon_minus_comparator_macro_ap):
            raise ValueError(f"K562 paired summary differs from per-TF AP: {method}")
    if not np.allclose(wide.mean().loc[ra.method], ra.macro_ap_all_tf):
        raise ValueError("K562 aggregate differs from per-TF AP")
    for assay, part in binding.groupby("assay"):
        wide = part.pivot(index="tf", columns="method", values="average_precision")
        aggregate = ba[ba.assay.eq(assay)].set_index("method")
        if wide.isna().any().any() or not np.allclose(wide.mean().loc[aggregate.index], aggregate.macro_ap):
            raise ValueError(f"K562 {assay} aggregate differs from per-TF AP")
    for frame, aggregate, keys in ((response, ra, ["method"]), (binding, ba, ["assay", "method"])):
        for k in KS:
            means = frame.groupby(keys)[f"top{k}_supported_fraction"].mean()
            values = aggregate.set_index(keys)[f"top{k}_response_fraction"]
            if not np.allclose(means.loc[values.index], values):
                raise ValueError(f"K562 top-{k} aggregate differs from per-TF values")
    return response, comparisons, binding, ra, ba

def dnmt1_counts(build):
    counts = build.csv(build.assays / "external/dnmt1_joint_top100.csv").set_index("method")
    columns = ["response", "bound", "both"]
    if counts.index.has_duplicates or not counts.loc[PUB, "targets"].eq(100).all():
        raise ValueError("DNMT1 needs one stable top-100 row per plotted method")
    expected = 100 * counts.loc["population", columns].to_numpy(float) / counts.loc["population", "targets"]
    return counts, expected

def rpe1_values(build):
    frame = build.csv(build.assays / "rpe1/per_tf_metrics.csv")
    report = build.js(build.assays / "rpe1/summary.json")
    wide = frame[frame.method.isin(RPE_NAMES)].pivot(index="tf", columns="method", values="top100_mean_abs_effect").sort_index()
    if wide.isna().any().any() or set(wide.columns) != set(RPE_NAMES):
        raise ValueError("Incomplete RPE1 cohort")
    old = build.npz("response_bootstrap")
    indices = old["indices"] if np.array_equal(old["tf_genes"], wide.index.to_numpy()) else np.random.default_rng(9071).integers(len(wide), size=(10000, len(wide)))
    return wide, indices, report

def main_figures(build, completion):
    architecture(build)
    env = namespace()
    labels = {"beacon": "BEACON", "degree_logistic": env["TOPOLOGY"], "gnnlink": "GNNLink", "reggain": "RegGAIN",
              "gclink": "GCLink", "scregulate": "scRegulate", **{m: m for m in POP}}
    primary = completion.query('variant == "beacon" and ratio == 5 and corruption == 0').copy()
    primary["label"] = primary.method.map(labels)
    primary["dataset"] = primary.dataset_id
    env.update(coverage=primary, primary=primary[primary.coverage.eq(.8)])

    def save_prior(fig, name):
        # The upper panels have a fixed method list; append the two controls.
        for ax, dataset in zip(fig.axes[:4], env["DS"]):
            for method in POP:
                p = primary[primary.dataset_id.eq(dataset) & primary.method.eq(method)]
                g = p.groupby("coverage").auprc_trapezoid.agg(["mean", "min", "max"])
                if len(g) != 5 or not p.groupby("coverage").size().eq(3).all():
                    raise ValueError(f"Incomplete coverage control: {dataset}, {method}")
                color, marker, ls = STYLES[method]
                ax.plot(g.index * 100, g["mean"], color=color, marker=marker, ls=ls, ms=3)
                ax.fill_between(g.index * 100, g["min"], g["max"], color=color, alpha=.08)
        for legend in list(fig.legends):
            legend.remove()
        handles = [Line2D([0], [0], color=s[0], marker=s[1], ls=s[2], label=m, ms=4) for m, s in env["FIG2_STYLE"].items()]
        fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.07))
        save(build, fig, name, "A–E: pooled trapezoid AUPRC; F: all-TF mean AP",
             "Same four coverage curves and two 80%-prior comparison panels. Adds both prior popularity controls; saved BEACON fit replaces BEACON and topology values.")
    env["save"] = save_prior
    drawing.prior_completion(env['coverage'], env['primary'], env['save'])

    response, comparisons, binding, ra, ba = assays(build)
    env.update(components=COMPONENTS, ablation=completion.query('coverage == .8 and ratio == 5 and corruption == 0'),
               resp_controls=response, control_comparisons=comparisons)
    def save_components(fig, name):
        prior = comparisons.loc[POP[0]]
        for ax in fig.axes[2:]:
            mean, low, high = prior.beacon_minus_comparator_macro_ap, prior.ci_low, prior.ci_high
            ax.errorbar(mean, -.12, xerr=[[mean-low], [high-mean]], fmt="^", color=STYLES[POP[0]][0], ms=4, capsize=2)
            # Fourth line under the n/clipped block, above the mean-difference marker.
            ax.text(.03, .745, "Blue triangle: BEACON − prior target in-degree", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.9, color=STYLES[POP[0]][0])
            left, right = ax.get_xlim()
            ax.set_xlim(min(left, low * 1.1), max(right, high * 1.1))
        save(build, fig, name, "A: pooled trapezoid AUPRC; B: all-TF mean AP; C–D: response AP differences",
             "Same two ablation heatmaps and two K562 control strips. Adds prior-only controls to heatmaps and "
             "prior-target paired AP intervals to strips; SNN is an ablation, not the published objective.", True)
    env["save"] = save_components
    drawing.component_evidence(env['components'], env['ablation'], env['resp_controls'], env['control_comparisons'], env['save'])

    pub, ks = PUB, KS
    def curves(frame):
        frame = frame.set_index("method")
        return {m: {k: float(frame.loc[m, f"top{k}_response_fraction"]) for k in ks} for m in pub}
    env.update(response_tf=response, binding_tf=binding, response_agg=ra, binding_agg=ba, pub=pub, ks=ks,
               response_curves=curves(ra), binding_curves=curves(ba.query('assay == "binding"')))
    env["fig"], env["axes"] = drawing.experimental_validation(env['response_agg'], env['response_tf'], env['binding_agg'], env['binding_tf'], env['response_curves'], env['binding_curves'], env['pub'], env['ks'])
    fig, axes = env["fig"], env["axes"]
    for ax, data, column in [(axes[0, 0], response, "average_precision_all_tf"),
                             (axes[0, 1], binding.query('assay == "binding"'), "average_precision"),
                             (axes[0, 2], binding.query('assay == "binding_and_response"'), "average_precision")]:
        wide = data.pivot(index="tf", columns="method", values=column)
        x, y = wide[POP[0]].to_numpy(), wide.BEACON.to_numpy()
        if ax is axes[0, 0]:
            x, y = np.sqrt(x), np.sqrt(y)
        ax.scatter(x, y, s=9, marker="^", edgecolors=STYLES[POP[0]][0], facecolors="none", linewidths=.4, alpha=.5)
        high = max(ax.get_xlim()[1], x.max() * 1.04, y.max() * 1.04)
        ax.set(xlim=(0, high), ylim=(0, high), xlabel="Comparator AP")
        ax.legend(handles=[Line2D([], [], marker="o", color="#333333", ls="", label="GNNLink"),
                           Line2D([], [], marker="^", color=STYLES[POP[0]][0], ls="", label="Prior target in-degree")],
                  fontsize=4.4, loc="lower right", frameon=False)
    counts, expected = dnmt1_counts(build)
    ax = axes[1, 2]
    columns = ["response", "bound", "both"]
    width = .8 / len(pub)
    highest = float(expected.max())
    for j, method in enumerate(pub):
        values = counts.loc[method, columns].to_numpy(float)
        highest = max(highest, values.max())
        bars = ax.bar(np.arange(3) + (j - (len(pub) - 1) / 2) * width, values, width * .93,
                      color=env["METHOD_STYLE"][method][0], edgecolor="#222222", linewidth=.35)
        ax.bar_label(bars, fontsize=5, padding=1)
    for x, value in enumerate(expected):
        ax.plot([x - .43, x + .43], [value, value], color="#222222", ls="--", lw=.8)
    ax.set(xticks=range(3), xticklabels=["Response", "Binding", "Both"], ylim=(0, highest * 1.55), ylabel="Supported targets in top 100")
    env["panel"](ax, "f", "DNMT1 top-100 assay support")
    env["light_grid"](ax)
    fig.legend(handles=[Line2D([], [], color=env["METHOD_STYLE"][m][0], marker=env["METHOD_STYLE"][m][1], label=m) for m in pub],
               ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.07))
    save(build, fig, "experimental_validation", "A–C: equal-regulator AP; D–E: fractional-tie top-K support; F: supported-target counts",
         "Preserves six panels; adds prior target in-degree. Reads configured evaluation outputs only. "
         "DNMT1 uses exactly 100 targets with stable ordering to resolve cutoff ties; dashed lines show population expectations. "
         "Prior regulator out-degree cannot rank targets within a regulator and is omitted.", True)
    cross_context(build, env)

def cross_context(build, env):
    tcells = tcell_evaluations(build)["per_tf_metrics"]
    trrust = build.csv("trrust_rankings")  # paired-monitor GRNBoost2
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.3), layout="constrained", gridspec_kw={"height_ratios": [1, 1.4]})
    for ax, condition, letter in zip(axes[0, :2], ("resting", "stimulated"), ("a", "b")):
        methods = ["BEACON", "GNNLink", POP[0]]
        data = tcells[tcells.condition.eq(condition) & tcells.method.isin(methods)]
        wide = data.pivot(index="tf", columns="method", values="average_precision").sort_values("BEACON")
        training = build.npz(("training", condition))
        names = dict(zip(training["genes"].astype(str), training["symbols"].astype(str)))
        for i, row in enumerate(wide.to_numpy()):
            ax.plot([row.min(), row.max()], [i, i], color="#BBBBBB", lw=.8)
        for m in methods:
            color, marker, _ = env["METHOD_STYLE"][m]
            # Hollow comparator markers and BEACON on top keep tied values visible.
            ax.scatter(wide[m], np.arange(len(wide)), facecolors=color if m == "BEACON" else "none",
                       edgecolors=color, marker=marker, s=16, label=m, zorder=3 if m == "BEACON" else 2)
        ax.set(yticks=range(len(wide)), yticklabels=[names.get(str(t), str(t)) for t in wide.index], xlabel="Per-regulator AP", xlim=(0, None))
        env["panel"](ax, letter, "Primary " + ("resting" if condition == "resting" else "re-stimulated") + " T cells")
        env["light_grid"](ax, "x")
    methods = ["BEACON", "GNNLink", "GENELink", "GRNBoost2", "GENIE3", "Inferelator", POP[0]]
    colors = {"BEACON": "#1B9E77", "GNNLink": "#D95F02", "GENELink": "#0072B2", "GRNBoost2": "#CC79A7",
              "GENIE3": "#E69F00", "Inferelator": "#56B4E9", POP[0]: "#0072B2", "RegGAIN": "#7570B3",
              "Topology control": "#555555", "Expression correlation": "#999999", "Target detection": "#222222"}
    env.update(methods=methods, method_colors=colors)
    for ax, metric, title in [(axes[0, 2], "auroc", "c TRRUST same-regulator AUROC"),
                               (axes[1, 0], "average_precision", "d TRRUST regulator-weighted AP")]:
        env["trrust_dumbbell"](ax, trrust.query('evaluation == "all"').set_index("method")[metric],
                              trrust.query('evaluation == "overlap_removed"').set_index("method")[metric], title,
                              "AUROC" if metric == "auroc" else "Average precision")
    names = RPE_NAMES
    wide, indices, report = rpe1_values(build)
    rng = np.random.default_rng(7)
    for paired, ax, letter in [(False, axes[1, 1], "e"), (True, axes[1, 2], "f")]:
        keys = list(names)[1:] if paired else list(names)
        for y, key in zip(np.arange(len(keys))[::-1], keys):
            values = (wide.beacon_model - wide[key] if paired else wide[key]).to_numpy()
            mean = values.mean()
            low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
            if paired:
                target = report["beacon_minus"][key]["top100_mean_abs_effect"]
                if not np.allclose([mean, low, high], [target["mean"], *target["ci95"]]):
                    raise ValueError("RPE1 bootstrap differs from saved evaluation")
                ax.scatter(values, y + rng.uniform(-.13, .13, len(values)), s=7, facecolors="none", edgecolors="#777777", linewidths=.4)
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", ms=4, capsize=2, color=colors.get(names[key], "#555555"))
        ax.set(yticks=np.arange(len(keys))[::-1], yticklabels=[names[k].replace(" ", "\n", 1) if len(names[k]) > 17 else names[k] for k in keys],
               xlabel="BEACON − comparator (control SD)" if paired else "Mean top-100 response (control SD)")
        if paired:
            ax.axvline(0, color="#777777", ls="--", lw=.75)
        else:
            ax.set_xlim(left=0)
        env["panel"](ax, letter, "RPE1 paired differences" if paired else "RPE1 mean response")
        env["light_grid"](ax, "x")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="#555555", ls="", label="TRRUST: all pairs"),
                Line2D([], [], marker="D", color="#555555", markerfacecolor="white", ls="", label="TRRUST: overlap removed")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.09))
    save(build, fig, "cross_context_validation", "A–B: response AP; C: within-regulator AUROC; D: regulator-weighted AP; E–F: top-100 mean absolute response (control SD)",
         "Preserves dumbbells and RPE1 mean/difference panels, expands space for prior target in-degree. "
         "T cells read primary rows directly from calibrated T-cell response tables. "
         "RPE1 continuous magnitude remains distinct from response AP. "
         "Constant within-regulator out-degree is omitted.", True)

def matched_target_panel(ax, challenge, datasets):
    labels = ["BEACON", "Topology control", "GNNLink", "RegGAIN"]
    lines = {line.get_label(): line for line in ax.lines if line.get_label() in labels}
    if set(lines) != set(labels):
        raise ValueError("Matched-target method artists changed")
    values = challenge[challenge.method.eq(POP[0])].groupby("dataset").pair_concordance.mean().reindex(datasets)
    if values.isna().any():
        raise ValueError("Missing target in-degree matched-pair results")
    lines[POP[0]], = ax.plot([], [], marker=STYLES[POP[0]][1], ls="", color=STYLES[POP[0]][0], label=POP[0], ms=4)
    lines[POP[0]].set_ydata(values)
    labels.append(POP[0])
    for j, label in enumerate(labels):
        lines[label].set_xdata(np.arange(len(datasets)) + (j - (len(labels) - 1) / 2) * .1)
    ax.set_ylim(.45, .95)
    ax.legend(frameon=False, fontsize=5, ncol=2)

def sampled_presentation(fig, path):
    """Override default label placement only; leave curves and scores untouched."""
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text({"mHSC E": "mHSC-E", "mHSC GM": "mHSC-GM", "mHSC L": "mHSC-L"}.get(text.get_text(), text.get_text()))
        if text.get_text().startswith("mean ") and text.get_rotation() == 90:
            text.set_ha("left")
            text.set_transform(offset_copy(text.get_transform(), fig=fig, x=4, y=0, units="points"))
    if path.stem == "score_distribution":
        ax = fig.axes[0]
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, frameon=False)
        fig.subplots_adjust(bottom=.34)
    elif path.stem == "score_distribution_by_evidence_type":
        fig.subplots_adjust(bottom=.22)

def heatmaps(build, source):
    methods = ["BEACON", "GNNLINK", "GENELINK", "Inferelator 3.0", "GRNBoost2", "GENIE3", *POP]
    cells = ["mESC", "hESC", "hHEP", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
    rows = [(family, cell) for family in ("STRING", "Non-Specific", "LOF/GOF", "Specific")
            for cell in (cells if family != "LOF/GOF" else ["mESC"])]
    for metric, name in [("auroc_pct", "auroc"), ("auprc_pct", "auprc")]:
        values = source.pivot(index=["reference_type", "cell_type"], columns=["tf_panel", "method"], values=metric)
        values = values.reindex(index=pd.MultiIndex.from_tuples(rows), columns=pd.MultiIndex.from_product([["TFs+500", "TFs+1000"], methods]))
        if values.isna().any().any():
            raise ValueError("Missing heatmap cell")
        fig, ax = plt.subplots(figsize=(12.5, 9.2))
        sns.heatmap(values, cmap="rocket", vmin=40, vmax=100, annot=True, fmt=".2f", annot_kws={"size": 6},
                    linewidths=.2, linecolor="#888888", cbar_kws={"label": "AUROC (%)" if name == "auroc" else "Trapezoid AUPRC (%)"}, ax=ax)
        names = [{"GNNLINK": "GNNLink", "GENELINK": "GENELink"}.get(m, m).replace("Prior target in-degree", "Prior target\nin-degree").replace("Prior regulator out-degree", "Prior regulator\nout-degree") for m in methods] * 2
        ax.set(xticklabels=names, yticklabels=[r[1] for r in rows], xlabel="", ylabel="")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelrotation=35, labelsize=6)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)
        ax.axvline(len(methods), color="white", lw=2.5)
        for y in (7, 14, 15):
            ax.axhline(y, color="white", lw=2.5)
        for y, title in [(3.5, "STRING"), (10.5, "Non-Specific"), (14.5, "LOF/GOF"), (18.5, "Specific")]:
            ax.text(-1.7, y, title, rotation=90, ha="center", va="center", fontsize=8)
        ax.text(.25, 1.12, "TFs+500", transform=ax.transAxes, ha="center")
        ax.text(.75, 1.12, "TFs+1000", transform=ax.transAxes, ha="center")
        save(build, fig, name, "AUROC" if name == "auroc" else "pooled trapezoid AUPRC",
             "Original heatmap generator is absent. Reconstructs its inspected rocket heatmap, reference-family blocks, "
             "cell order and two gene panels from the saved source CSV. Replaces BEACON and adds both prior popularity columns; "
             "five comparator columns contain benchmark values. " + TIED_AUPRC_NOTE)

def shortcut(build, sampled):
    comparison = build.csv("sampled_pair_summary/paired_comparison.csv")
    audit = build.csv("split_counts")
    fig, axes = drawing.shortcut(comparison, audit)
    controls = {POP[0]: ("#555555", "^"), POP[1]: ("#999999", "s")}
    for method, (color, marker) in controls.items():
        pop = sampled[sampled.method.eq(method)].set_index("dataset_id")
        axes[0].scatter(comparison.beacon_auprc_trapezoid, pop.loc[comparison.dataset_id, "auprc_trapezoid"],
                        marker=marker, s=15, facecolors="none", edgecolors=color, label=method)
        macro = []
        for d in comparison.dataset_id:
            z = build.npz(("sampled", d))
            values = []
            for tf in np.unique(z["edges"][:, 0]):
                mask = z["edges"][:, 0] == tf
                if len(np.unique(z["labels"][mask])) == 2:
                    values.append(roc_auc_score(z["labels"][mask], z[method][mask]))
            macro.append(np.mean(values) if values else np.nan)
        # Out-degree is constant within a source, so its within-source AUROC is 0.5 by construction.
        if method != POP[1]:
            axes[1].scatter(comparison.beacon_macro_auroc, macro, marker=marker, s=15,
                            facecolors="none", edgecolors=color)
        fraction = audit.set_index("dataset_id").loc[comparison.dataset_id]
        axes[2].scatter(fraction.test_unlabeled_source_absent_from_train / fraction.test_unlabeled,
                        pop.loc[comparison.dataset_id, "auprc_trapezoid"], marker=marker, s=15,
                        facecolors="none", edgecolors=color)
    # Every plotted comparator is built from training-graph degree; keep every point inside the axes.
    lowest = min(sampled[sampled.method.isin(POP)].query("dataset_id in @comparison.dataset_id").auprc_trapezoid.min(), .4)
    for ax, label in zip(axes, ["Training-degree control", "Training-degree control", "Training-degree control AUPRC"]):
        ax.set_ylabel(label)
    for ax in (axes[0], axes[2]):
        ax.set_ylim(np.floor(lowest * 10) / 10, ax.get_ylim()[1])
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["LOF/GOF" if label == "Lofgof" else label for label in labels]
    handles.append(Line2D([], [], marker="o", ls="", color="#555555", markerfacecolor="#555555"))
    labels.append("Topology control (filled circles)")
    axes[0].legend(handles, labels, fontsize=5, loc="upper left", frameon=False)
    save(build, fig, "benchmark_shortcut_audit", "A/C: pooled trapezoid AUPRC; B: within-source AUROC",
         "Reuses the three-panel shortcut renderer with paired comparison. Adds both training-degree controls, "
         "including the constant out-degree AUROC of 0.5 within each eligible source. " + TIED_AUPRC_NOTE)


def supplementary(build, metrics, completion):
    env = namespace()
    scaling = build.csv("runtime_scaling.csv").query('gpu_type == "NVIDIA L40S"').copy()
    scaling["other_seconds"] = scaling.total_seconds - scaling.model_fit_seconds
    if len(scaling) != 44 or not scaling.packing.eq(4).all() or (scaling[["model_fit_seconds", "other_seconds"]] <= 0).any().any():
        raise ValueError("Expected 44 positive L40S timing records with packing=4")
    scaling = scaling[["dataset_id", "sampled_training_pairs", "test_pairs", "model_fit_seconds", "other_seconds"]]
    env["scaling"] = scaling
    def save_runtime(fig, name):
        for text in fig.texts:
            text.set_text(text.get_text().replace("on one NVIDIA L40S GPU", "with four concurrent runs per NVIDIA L40S GPU"))
        save(build, fig, name, "seconds; descriptive log-log slopes",
             "Same two runtime panels, now 44 L40S runs with four concurrent runs per NVIDIA L40S GPU; "
             "times are not isolated throughput. Encoder timer includes internal GP selection and encoder refit; "
             "GP timer covers the refit.")
    env["save"] = save_runtime
    drawing.runtime_scaling(env['scaling'], env['save'])
    env.update(h=build.csv("completion_summary/hard_challenge_with_popularity.csv"),
               match=build.csv("completion_summary/hard_challenge_coverage.csv"),
               grid=completion.query('variant == "beacon"').assign(control="beacon"),
               COLORS={"BEACON": "#14678a", "Topology control": "#777777", "GNNLink": "#d37724", "RegGAIN": "#6b4c9a"},
               NAMES={"beacon": "BEACON", "degree_logistic": "Topology control", "gnnlink": "GNNLink", "reggain": "RegGAIN"},
               contextcolors=['#14678a', '#b96a2b', '#6b4c9a', '#56883c'], panel=env["label_panel"])
    def save_sensitivity(fig, name):
        matched_target_panel(fig.axes[0], env["h"], env["DS"])
        save(build, fig, name, "A: pair concordance; B: matched fraction; C–D: pooled AP (sensitivity-panel definition)",
             "Four panels. Adds prior target in-degree to matched-target comparison; "
             "regulator out-degree is omitted because it is constant within a regulator. "
             "Marker groups are centered on each context. Coverage fractions are unchanged protocol counts.")
    env["save"] = save_sensitivity
    drawing.completion_sensitivity(*[env[k] for k in ('h','match','grid','COLORS','NAMES','contextcolors','panel','save')])
    sampled_figures(build, metrics)

def sampled_figures(build, metrics):
    from figures import precision, complexity
    from evaluation.metrics import fractional_topk
    source = build.csv("comparators/sampled_pairs/beeline_topology_associations_source.csv")
    template = source[source.method.eq("BEACON")].set_index("dataset_id")
    sampled = metrics[metrics.suite.eq("sampled_pair")].copy()
    sampled["dataset_id"] = sampled.run.str.extract(r"DS(\d+)")[0].astype(int)
    predictions = []
    for e in build.bundle.manifest["experiments"]:
        if e["suite"] != "sampled_pairs": continue
        d = e["condition"]["dataset"]
        z = build.bundle.arrays(e["reference"]["predictions"])
        meta = template.loc[d]
        cell = meta.cell_type.replace("mHSC-", "mHSC ") + " " + str(meta.panel_size)
        network = {"Non-Specific":"Non-specific"}.get(meta.reference_type, meta.reference_type)
        predictions.append(pd.DataFrame(dict(dataset_id=d, network=network, cell_panel=cell,
            cell_display=precision.display_cell_label(cell), tf_panel=precision.panel_from_cell_label(cell),
            score=z["beacon"], label=z["labels"])))
    if len(predictions) != 44: raise ValueError("Expected 44 sampled-pair settings")
    predictions = pd.concat(predictions, ignore_index=True)
    values = []
    for d, part in predictions.groupby("dataset_id"):
        meta = part.iloc[0].drop(["score", "label"]).to_dict()
        for k in precision.K_VALUES:
            values.append(dict(meta, k=k, effective_k=min(k,len(part)), precision_percent=100*fractional_topk(part.label.to_numpy(),part.score.to_numpy(),k)))
    values = pd.DataFrame(values)
    for function, data, name in [(precision.plot_category_curves,values,"precision_by_evidence_type"),
        (precision.plot_network_curves,values,"precision_by_dataset"),
        (precision.plot_all_distribution,predictions,"score_distribution"),
        (precision.plot_category_distribution,predictions,"score_distribution_by_evidence_type")]:
        def save_precision(fig, path):
            sampled_presentation(fig,path)
            build.save(fig,name,"fractional-tie precision or score distribution","Current manuscript artists and saved scores.")
        with patch.object(precision,"save_figure",save_precision):
            function(data,build.directory/(name+".png"))
    adapted = source[source.method.ne("BEACON")].copy()
    for method, display in [("beacon","BEACON"),*[(m,m) for m in POP]]:
        rows=template.copy();current=sampled[sampled.method.eq(method)].set_index("dataset_id")
        if set(rows.index)!=set(current.index):raise ValueError("Incomplete sampled cohort")
        rows["method"]=display;rows["auroc_pct"]=current.auroc*100;rows["auprc_pct"]=current.auprc_trapezoid*100
        adapted=pd.concat([adapted,rows.reset_index()],ignore_index=True)
    heatmaps(build,adapted)
    complexity.METHOD_ORDER += list(POP)
    complexity.METHOD_LABELS.update({m:m for m in POP})
    control_styles = {POP[0]: ("#009E73", "^"), POP[1]: ("#CC79A7", "s")}
    complexity.COLORS.update({m: style[0] for m, style in control_styles.items()})
    original_panel = complexity.plot_panel
    def plot_complexity_panel(ax, *args):
        original_panel(ax, *args)
        for method, (color, marker) in control_styles.items():
            line = next(line for line in ax.lines if line.get_label() == method)
            line.set(marker=marker, markerfacecolor="none", markeredgecolor=color)
            points = ax.collections[2 * complexity.METHOD_ORDER.index(method)]
            style = MarkerStyle(marker)
            points.set_paths([style.get_path().transformed(style.get_transform())])
            points.set_facecolor("none")
            points.set_edgecolor(color)
    path=build.directory/"complexity_source.csv";adapted.to_csv(path,index=False)
    table=complexity.load_source(path);density,depth=complexity.summarize(table)
    complexity.metric_ylim=lambda metric:(.4 if metric=="AUROC" else 0,1.01)
    with patch.object(complexity,"plot_panel",plot_complexity_panel),patch.object(Figure,"savefig",lambda *a,**k:None),patch.object(plt,"close",lambda *a,**k:None):
        complexity.make_plot(table,density,depth,build.directory/"network_complexity.png",300)
    fig=plt.gcf();fig.subplots_adjust(bottom=.26)
    build.save(fig,"network_complexity","pooled AUPRC and AUROC","Saved graph descriptors and current manuscript trend renderer.")
    shortcut(build,sampled)
