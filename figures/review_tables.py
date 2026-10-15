"""Paper table bodies S8, S15 and S16, recomputed from saved predictions."""
from evaluation.specificity import BINS, FIXED, external, fixed_pools
from figures.checks import reciprocal
from figures.drawing import display

METHODS = ["BEACON", "GNNLink", "GCLink", "scRegulate", "RegGAIN", "GENELink", "scRegNet",
           "GRNBoost2", "GENIE3", "Inferelator", "Topology control"]
ASSAYS = {"k562": "K562", "tcell_resting": "Resting T cells",
          "tcell_stimulated": "Re-stimulated T cells", "rpe1": "RPE1"}


def row(cells):
    return " & ".join(map(str, cells)) + r" \\"


def reciprocal_body(frame):
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{\textbf{Pooled AP}} & \multicolumn{2}{c}{\textbf{All-TF mean AP}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"\textbf{Context} & \textbf{Method} & \textbf{All} & \textbf{Without} & \textbf{All} & \textbf{Without} \\",
             r"\midrule"]
    methods = ["BEACON", "GNNLink", "Topology control", "Prior target in-degree", "Reverse-edge rule"]
    for context in FIXED.values():
        if context != next(iter(FIXED.values())):
            lines.append(r"\addlinespace")
        part = frame[frame.context.eq(context)].groupby("method").mean(numeric_only=True)
        for i, method in enumerate(methods):
            values = part.loc[method]
            title = f"{context} ({values.share_reverse * 100:.0f}" + r"\%)" if i == 0 else ""
            lines.append(row([title, display(method), *[f"{values[k]:.3f}" for k in ("ap_all", "ap_without", "alltf_all", "alltf_without")]]))
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def fixed_body(frame):
    lines = [r"\begin{tabular}{l*{13}{r}}", r"\toprule",
             row(["Context", r"\begin{tabular}[b]{@{}c@{}}Revealed\\(\%)\end{tabular}",
                  r"\begin{tabular}[b]{@{}c@{}}Edges\\per TF\end{tabular}", *METHODS[:-1],
                  r"\begin{tabular}[b]{@{}c@{}}Degree\\baseline\end{tabular}"]), r"\midrule"]
    for (context, coverage), part in frame.groupby(["context", "coverage"], sort=False):
        means = part.groupby("method").gain.mean()
        best = means.max()
        cells = []
        for method in METHODS:
            if method not in means:
                cells.append("--")
            else:
                value = means[method]
                text = f"{value:.2f}"
                cells.append(r"\textbf{" + text + "}" if round(value, 2) == round(best, 2) else text)
        lines.append(row([context, f"{100 * coverage:g}", f"{part.revealed_per_regulator.mean():.1f}", *cells]))
    # The body closes the resizebox opened by the manuscript.
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}}", ""])


def external_body(frame, strata):
    lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Agreement with consensus} & \multicolumn{3}{c}{Residual AP / chance} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Assay & TFs & Edges per TF & BEACON & GNNLink & Random & BEACON & GNNLink & Random \\",
             r"\midrule"]
    for context, label in ASSAYS.items():
        part = frame[frame.context.eq(context)].set_index("method")
        values = [part.loc[m, "residual_ap_over_chance"] for m in ("BEACON", "GNNLink", "Random")]
        ratios = [r"\textbf{" + f"{v:.2f}" + "}" if i < 2 and v == max(values[:2]) else f"{v:.2f}"
                  for i, v in enumerate(values)]
        lines.append(row([label, int(part.regulators.iloc[0]), f"{part.revealed_per_regulator.iloc[0]:.1f}",
                          *[f"{part.loc[m, 'consensus_spearman']:.3f}" for m in ("BEACON", "GNNLink", "Random")], *ratios]))
    lines.extend([r"\midrule",
                  r"\multicolumn{9}{l}{\textit{Stratified by each evaluated regulator's own prior edges}} \\",
                  r"\multicolumn{2}{l}{Assay} & Prior edges & TFs & AP TFs & \multicolumn{2}{c}{Median Spearman} & \multicolumn{2}{c}{Residual AP / chance} \\",
                  r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
                  r"\multicolumn{2}{l}{} & & & & BEACON & GNNLink & BEACON & GNNLink \\"])
    lookup = strata.set_index(["context", "method", "prior_bin"])
    for context, label in ASSAYS.items():
        for i, name in enumerate(BINS):
            pair = [lookup.loc[(context, method, name)] for method in ("BEACON", "GNNLink")]
            a, b = pair
            for key in ("regulators", "finite_spearman_regulators", "ap_eligible_regulators"):
                if a[key] != b[key]:
                    raise ValueError(f"External regulator counts differ: {context}/{name}/{key}")
            n, nap = int(a.regulators), int(a.ap_eligible_regulators)
            if n != int(a.finite_spearman_regulators) or not 0 <= nap <= n:
                raise ValueError("External stratum eligibility changed")
            agreements = [f"{r.consensus_spearman:.3f}" if n else "--" for r in pair]
            values = [r.residual_ap_over_chance for r in pair]
            ratios = [(r"\textbf{" + f"{v:.2f}" + "}" if v == max(values) else f"{v:.2f}") for v in values] if nap else ["--", "--"]
            lines.append(row([r"\multicolumn{2}{l}{" + (label if i == 0 else "") + "}",
                              name.replace("-", "--"), n, nap, *agreements, *ratios]))
        if context != "rpe1":
            lines.append(r"\addlinespace[2pt]")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def generate(build):
    if build.selected in (None, "reciprocal_edges"):
        frame = reciprocal(build.bundle)
        frame.to_csv(build.directory / "reciprocal_edges.csv", index=False)
        build.table("reciprocal_edges", reciprocal_body(frame), "pooled AP and all-TF mean AP",
                    "Three-split means; without removes all reverse-revealed pairs.")
    if build.selected in (None, "specificity_fixed"):
        frame = fixed_pools(build.bundle)
        frame.to_csv(build.directory / "specificity_fixed.csv", index=False)
        build.table("specificity_fixed", fixed_body(frame), "mean within AP / mean leave-one-out consensus AP",
                    "Full-pool average-tie percentile consensus before symmetric reverse exclusion; mean of three fit-level ratios.")
    if build.selected in (None, "specificity_external"):
        frame, detail, strata = external(build.bundle, details=True)
        frame.to_csv(build.directory / "specificity_external.csv", index=False)
        detail.to_csv(build.directory / "specificity_external_per_regulator.csv", index=False)
        strata.to_csv(build.directory / "specificity_external_strata.csv", index=False)
        build.table("specificity_external", external_body(frame, strata), "consensus Spearman and residual AP / chance",
                    "Own-prior strata; consensus uses every evaluated regulator before stratification.")
