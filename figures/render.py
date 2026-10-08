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
STYLES = {m: drawing.STYLE[m][:3] for m in POP}
PUB = ["BEACON", "GNNLink", "RegGAIN", POP[0]]
KS = [10, 25, 50, 100, 200]
COMPONENTS = [('beacon', 'beacon', 'BEACON (full model)'), ('fa_gp', 'fa_gp', 'w/o encoder'),
              ('beacon', 'decoder', 'w/o GP (decoder scores)'), ('beacon', 'learned_logistic', 'w/o GP (logistic head)'),
              ('graph_only', 'graph_only', 'w/o expression (random features)'),
              ('permuted', 'permuted', 'w/o expression (genes permuted)'), *[('beacon', m, m) for m in POP]]
RPE_NAMES = {"beacon_model": "BEACON", "gnnlink": "GNNLink", "reggain": "RegGAIN",
             "inferelator_tfa_bbsr_5_bootstraps": "Inferelator", "degree_logistic": "Topology control", POP[0]: POP[0],
             "expression_abs_correlation": "Expression correlation", "target_control_detection": "Target detection"}
TIED_AUPRC_NOTE = ("Tied constant scores inflate trapezoid AUPRC for prior controls "
                   "(up to 0.105 for target in-degree, DS1514); this area is not enrichment.")

def overview(build):
    from figures.overview import draw
    save(build, draw(), "beacon_overview", "schematic (no empirical metric)", "Method and evaluation schematic.")

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
    overview(build)
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
        # Open markers: pooled AUPRC after excluding held-out edges whose reverse was revealed.
        from evaluation.metrics import binary_metrics
        from evaluation.specificity import fixed_pool
        ax, labels = fig.axes[4], list(env["FIG2_STYLE"])
        offsets = dict(zip(labels, np.linspace(.27, -.27, len(labels))))
        for y, dataset in zip(np.arange(4)[::-1], env["DS"]):
            values = {}
            for seed in (42, 14, 100):
                edges, y_true, scores, reverse, _ = fixed_pool(build.bundle, dataset, seed, .8)
                keep = ~(y_true & reverse)
                for method, s in scores.items():
                    values.setdefault(method, []).append(binary_metrics(y_true[keep].astype(int), s[keep])["auprc_trapezoid"])
            for method, v in values.items():
                color = env["FIG2_STYLE"][method][0]
                ax.plot(np.mean(v), y + offsets[method], marker=env["FIG2_STYLE"][method][1], ms=4.5, ls="",
                        markerfacecolor="white", markeredgecolor=color, markeredgewidth=.9, zorder=4)
        for legend in list(fig.legends):
            legend.remove()
        handles = [Line2D([0], [0], color=s[0], marker=s[1], ls=s[2], label=m, ms=4) for m, s in env["FIG2_STYLE"].items()]
        handles.append(Line2D([0], [0], color="#555555", marker="o", ls="", markerfacecolor="white", ms=4,
                              label="Open: reverse-revealed held-out edges excluded"))
        fig.legend(handles=drawing.row_major(handles, 5), ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.09))
        save(build, fig, name, "A–E: pooled trapezoid AUPRC; F: all-TF mean AP",
             "Same four coverage curves and two 80%-prior comparison panels. Adds both prior popularity controls; saved BEACON fit replaces BEACON and topology values.")
    env["save"] = save_prior
    drawing.prior_completion(env['coverage'], env['primary'], env['save'])

    response, comparisons, binding, ra, ba = assays(build)
    env.update(components=COMPONENTS, ablation=completion.query('coverage == .8 and ratio == 5 and corruption == 0'))
    def save_components(fig, name):
        save(build, fig, name, "A: pooled trapezoid AUPRC; B: all-TF mean AP; C: paired K562 response AP differences",
             "Ablations of the final model with the two prior-only rankings; K562 expression controls with regulator-bootstrap intervals.")
    drawing.component_evidence(env['components'], env['ablation'], comparisons, save_components)

    summary, controls = matched_results(build)
    experimental_validation(build, ra, ba, summary, controls)
    cross_context(build, summary, controls)

NAMES = {"Degree logistic": "Topology control"}
K562_METHODS = ["BEACON", "GNNLink", "RegGAIN", "Topology control", "Prior target in-degree", "Target expression"]


def matched_results(build):
    """Expression-matched top-K enrichment and the target-expression control, computed from bundle predictions."""
    from evaluation.matched import CONTEXTS, enrichment, summarize
    frames, controls = zip(*(enrichment(build.bundle, context) for context in CONTEXTS))
    summary, _ = summarize(pd.concat(frames, ignore_index=True))
    summary["method"] = summary.method.replace(NAMES)
    controls = pd.concat(controls, ignore_index=True)
    return summary, controls


def ratio_series(summary, condition, k, methods):
    part = summary[summary.condition.eq(condition) & summary.k.eq(k)].set_index("method")
    return ({m: float(part.loc[m, "ratio"]) for m in methods},
            {m: (float(part.loc[m, "ratio"] - part.loc[m, "ratio_ci_low"]), float(part.loc[m, "ratio_ci_high"] - part.loc[m, "ratio"])) for m in methods})


def experimental_validation(build, ra, ba, summary, controls):
    methods, ks = K562_METHODS, KS
    k562 = summary[summary.condition.eq("k562")]
    fractions = {m: [float(k562[k562.method.eq(m) & k562.k.eq(k)].observed.iloc[0]) for k in ks] for m in methods}
    ratios = {m: [float(k562[k562.method.eq(m) & k562.k.eq(k)].ratio.iloc[0]) for k in ks] for m in methods}
    errors = {m: np.array([[float(r.ratio - r.ratio_ci_low), float(r.ratio_ci_high - r.ratio)]
                           for k in ks for r in [k562[k562.method.eq(m) & k562.k.eq(k)].iloc[0]]]).T for m in methods}
    target = controls[controls.condition.eq("k562")]
    relative = {}
    for frame, assay in ((ra.assign(assay="response"), "response"), (ba, "binding"), (ba, "binding_and_response")):
        part = frame[frame.assay.eq(assay)].assign(method=lambda f: f.method.replace(NAMES)).set_index("method")
        control = target[target.assay.eq(assay)]
        for m in methods:
            value = control.average_precision.mean() / control.prevalence.mean() if m == "Target expression" else part.loc[m, "macro_ap_all_tf"] / part.loc[m, "macro_prevalence"]
            relative.setdefault(m, []).append(float(value))
    counts, expected = dnmt1_counts(build)
    counts = counts.rename(index=NAMES)
    dnmt1 = {m: counts.loc[m, ["response", "bound", "both"]].astype(float).tolist() for m in methods if m != "Target expression"}
    dnmt1["Target expression"] = target[target.assay.eq("dnmt1_top100")][["response", "bound", "both"]].iloc[0].astype(float).tolist()
    chance = float(ra.set_index("method").loc["BEACON", "macro_prevalence"])
    fig = drawing.k562_evaluation(fractions, ratios, errors, relative, dnmt1, expected, chance, ks, methods)
    save(build, fig, "experimental_validation", "A: fractional-tie top-K response fraction; B: observed over expression-matched fraction; "
         "C: mean AP over chance; D: supported-target counts",
         "K562 response, promoter-binding and joint endpoints with the target-expression control and 95% regulator-bootstrap intervals in B.")


def cross_context(build, summary, controls):
    per_tf = tcell_evaluations(build)["per_tf_metrics"].assign(method=lambda f: f.method.replace(NAMES))
    shown = K562_METHODS
    tcells = {}
    for condition in ("resting", "stimulated"):
        data = per_tf[per_tf.condition.eq(condition)]
        wide = data.pivot(index="tf", columns="method", values="average_precision")
        control = controls[controls.condition.eq("tcell_" + condition) & controls.assay.eq("response")].set_index("tf")
        wide["Target expression"] = control.average_precision.reindex(wide.index)
        if wide[shown].isna().any().any():
            raise ValueError("Incomplete T-cell regulator cohort")
        wide = wide.sort_values("BEACON", ascending=False)
        prevalence = data[data.method.eq("BEACON")].set_index("tf").prevalence.reindex(wide.index)
        training = build.npz(("training", condition))
        names = dict(zip(training["genes"].astype(str), training["symbols"].astype(str)))
        groups = [names.get(str(t), str(t)) for t in wide.index] + ["Mean"]
        tcells[condition] = (groups, {m: [*wide[m].tolist(), float(wide[m].mean())] for m in shown}, [*prevalence.tolist(), float(prevalence.mean())])
    matched_methods = K562_METHODS
    series, errors = {m: [] for m in matched_methods}, {m: [] for m in matched_methods}
    for condition in ("tcell_resting", "tcell_stimulated", "rpe1"):
        values, spans = ratio_series(summary, condition, 100, matched_methods)
        for m in matched_methods:
            series[m].append(values[m]); errors[m].append(spans[m])
    errors = {m: np.array(v).T for m, v in errors.items()}
    wide, indices, _ = rpe1_values(build)
    means, spans = {}, {}
    for key, name in sorted(RPE_NAMES.items(), key=lambda item: drawing.ORDER.index(item[1])):
        values = wide[key].to_numpy()
        low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
        means[name], spans[name] = float(values.mean()), (float(values.mean() - low), float(high - values.mean()))
    trrust = build.csv("trrust_rankings")  # paired-monitor GRNBoost2
    trrust_methods = ["BEACON", "GNNLink", "GENELink", "GRNBoost2", "GENIE3", "Inferelator", POP[0]]
    wide = {e: trrust[trrust.evaluation.eq(e)].set_index("method") for e in ("all", "overlap_removed")}
    table = {metric: {m: [float(wide[e].loc[m, metric]) for e in wide] for m in trrust_methods} for metric in ("auroc", "average_precision")}
    chance = [float(wide[e].loc[POP[1], "average_precision"]) for e in wide]
    legend = drawing.ordered(set(K562_METHODS) | set(trrust_methods) | {"Inferelator", "Expression correlation", "Target detection"})
    fig = drawing.cross_context(tcells, (["Resting T cells", "Re-stimulated T cells", "RPE1"], series), errors,
                                (means, spans), table, chance, legend)
    save(build, fig, "cross_context_validation", "A–B: response AP; C: top-100 observed over expression-matched fraction; "
         "D: top-100 mean absolute response; E: within-regulator AUROC; F: regulator-weighted AP",
         "Primary T-cell, RPE1 and TRRUST endpoints with the target-expression control; 95% regulator-bootstrap intervals in C and D.")


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
               COLORS={m: drawing.STYLE[m][0] for m in ("BEACON", "Topology control", "GNNLink", "RegGAIN")},
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
    sampled["dataset_id"] = sampled.dataset_id.astype(int)
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
