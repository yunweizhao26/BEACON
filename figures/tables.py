import numpy as np
import pandas as pd
from figures.build import POP, CONTEXT
from evaluation import metrics as ev
def tcell_evaluations(build): return build.tcells()
def emit(build, label, headers, rows, metric, notes, provisional=False, status="generated", **layout):
    if not rows: raise ValueError("Empty manuscript table")
    build.table(label.removeprefix("tab:"),tabular(headers,rows,**layout),metric,notes)
def escape(value):
    chars = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(chars.get(c, c) for c in str(value))

def number(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"

def tabular(headers, rows, *, blocks=None, reused=(), longtable=False, bold=()):
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Table row width differs from header")
    text_columns = {"Context", "Context and reference", "Method", "Pool", "Design", "Readout", "Evaluation", "Method / readout",
                    "Expression input", "Score", "Condition", "TF", "Assay"}
    alignment = "".join("l" if h in text_columns else "r" for h in headers)
    environment = "longtable" if longtable else "tabular"
    if longtable:
        alignment = r"p{0.49\textwidth}" + "r" * (len(headers) - 1)
    header = [r"\toprule", " & ".join(escape(h) for h in headers) + r" \\", r"\midrule"]
    lines = [r"\begin{" + environment + "}{" + alignment + "}", *header]
    if longtable:
        lines += [r"\endfirsthead", r"\multicolumn{" + str(len(headers)) + r"}{l}{Table \thetable\ continued.} \\",
                  *header, r"\endhead", r"\bottomrule", r"\endfoot"]
    for i, row in enumerate(rows):
        if i in (blocks or {}):
            title = blocks[i]
            if title:
                if i:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{" + str(len(headers)) + r"}{@{}l}{\textit{" + escape(title) + r"}} \\")
            elif i:
                lines.append(r"\addlinespace")
        cells = [escape(v) for v in row]
        for j, cell in enumerate(cells):
            if (i, j) in bold:
                value, _, rest = cell.partition(" [")
                cells[j] = r"\textbf{" + value + "}" + (" [" + rest if rest else "")
        if i in reused:
            cells[0] += r"$^\dagger$"
        lines.append(" & ".join(cells) + r" \\")
    if reused:
        lines += [r"\midrule", r"\multicolumn{" + str(len(headers)) +
                  r"}{@{}l}{$^\dagger$ Values from saved runs; not refitted.} \\"]
    if not longtable:
        lines.append(r"\bottomrule")
    return "\n".join([*lines, r"\end{" + environment + "}", ""])

def dimensions(build):
    rows, count_rows = [], []
    references = {1501: "hESC STRING", 1605: "mDC non-specific ChIP-seq",
                  1709: "mHSC-E specific ChIP-seq", 1801: "mESC perturbation"}
    for dataset, context in CONTEXT.items():
        manifest = build.case("fixed_pools",dataset,42)
        rows.append([references[dataset], *[manifest[k] for k in ("genes", "cells", "tfs", "reference_positives", "test_pairs", "test_positives")],
                     number(100 * manifest["test_prevalence"], 3)])
        counts = []
        for coverage in (".05", ".1", ".2", ".4", ".8"):
            values = [build.case("fixed_pools",dataset,seed,float(coverage))["train_positive"] for seed in (14, 42, 100)]
            if len(set(values)) != 1:
                raise ValueError("Training counts differ across splits")
            counts.append(values[0])
        count_rows.append([context, *counts])
    emit(build, "tab:reference_dimensions", ["Context and reference", "Genes", "Cells", "TFs", "Reference edges", "Test pairs", "Held-out edges", "Prevalence (%)"], rows,
         "protocol counts", "Protocol counts from run manifests.")
    emit(build, "tab:prior_edge_counts", ["Context", "5%", "10%", "20%", "40%", "80%"], count_rows,
         "known-edge counts", "Original coverage columns; verifies counts across all three splits.")
    rows = []
    for dataset in (201, 205):
        for coverage in (".05", ".2", ".8"):
            entries = [build.case("sergio",dataset,seed,float(coverage)) for seed in (14, 42, 100)]
            hidden = [x["hidden_reference_positives_sampled_as_unlabeled"] for x in entries]
            first = entries[0]
            rows.append([first["reference_positives"], int(float(coverage) * 100), first["train_positive"], first["train_unlabeled"],
                         f"{min(hidden)}--{max(hidden)}", first["test_positives"]])
    emit(build, "tab:simulator_truth", ["Graph edges", "Prior (%)", "Known edges", "Sampled unlabeled pairs", "True edges in sampled pairs", "Held-out edges"], rows,
         "simulator truth and training counts", "Counts from simulator manifests.")

def trrust(build):
    from figures.trrust import table_body
    table_body(build)

def scregnet(build, completion):
    from evaluation import seeds
    rows, blocks, bold = [], {}, set()
    for dataset, context in CONTEXT.items():
        frame = completion.query('dataset_id == @dataset and coverage == .8 and variant == "beacon"')
        selected = frame[frame.method.isin(["beacon", *POP])].copy()
        added = []
        for split in (42, 14, 100):
            entries = seeds.comparators(build.bundle, "fixed_pools" if dataset == 1709 else "scregnet_contexts", "scregnet",
                                        dataset=dataset, split_seed=split, coverage=.8)
            if sorted(c["seed"] for c in entries) != sorted(seeds.SEEDS):
                raise ValueError(f"scRegNet needs three optimization seeds: {context} split {split}")
            build.inputs.update(c["artifacts"]["predictions"] for c in entries)
            z = seeds.combine([build.bundle.arrays(c["artifacts"]["predictions"]) for c in entries])
            ref = build.case("fixed_pools", dataset, split, role="split")
            if not np.array_equal(ref["test"][tuple(z["edges"].T)], z["labels"]):
                raise ValueError("scRegNet pool differs from BEACON")
            added.append(dict(method="scRegNet", **ev.pool_metrics(z["labels"], z["scregnet"], z["edges"])))
        selected = pd.concat([selected, pd.DataFrame(added)], ignore_index=True)
        blocks[len(rows)] = context
        first = len(rows)
        for method, part in selected.groupby("method", sort=False):
            if len(part) != 3:
                raise ValueError("scRegNet table requires three splits")
            rows.append(["BEACON" if method == "beacon" else method, *[
                f"{part[m].mean():.4f} [{part[m].min():.4f}, {part[m].max():.4f}]" for m in ("auprc_trapezoid", "all_tf_ap", "auroc")]])
        for j in (1, 2, 3):
            means = [row[j].split(" [")[0] for row in rows[first:]]
            best = max(means, key=float)
            bold |= {(first + i, j) for i, value in enumerate(means) if value == best}
    emit(build, "tab:scregnet_fixed_pool", ["Method", "Pooled AUPRC", "All-TF mean AP", "AUROC"], rows,
         "trapezoid pooled AUPRC; all-TF mean AP; AUROC",
         "Means and split ranges over split seeds 42, 14 and 100 at 80% prior coverage in each fixed pool; BEACON and scRegNet are "
         "the means of three fits (optimization seeds 42, 14 and 100); scRegNet scores are checked against the same pair identities; "
         "bold marks the highest mean in each column of each fixed pool.", blocks=blocks, bold=bold)

def expression(build, metrics):
    cohort = metrics[(metrics.suite == "expression_controls") & metrics.coverage.eq(.8) & metrics.opt_seed.eq(42)].copy()
    cohort["dataset_id"] = cohort.dataset_id.astype(int)
    frame = cohort[cohort.method.isin(["beacon_gp", "beacon_decoder", "gnnlink", "genelink", "gclink"]) &
                   cohort.control.isin(["real", "cell_shuffled", "gene_permuted", "random"])].copy()
    frame["context"] = frame.dataset_id.map(CONTEXT)
    rows, blocks = [], {}
    methods = {"beacon_gp": "BEACON", "beacon_decoder": "BEACON pair decoder", "gnnlink": "GNNLink",
               "genelink": "GENELink", "gclink": "GCLink"}
    controls = {"real": "Real", "cell_shuffled": "Shuffled across cells", "gene_permuted": "Gene-permuted", "random": "Random"}
    if frame.groupby(["method", "control"]).ngroups != 20:
        raise ValueError("Expression controls require five readouts and four inputs")
    for method, display in methods.items():
        blocks[len(rows)] = ""
        for control, label in controls.items():
            part = frame[(frame.method == method) & (frame.control == control)]
            if len(part) != 12 or not part.groupby("context").size().eq(3).all():
                raise ValueError("Expression table missing context/split")
            values = part.groupby("context").auprc_trapezoid.mean()
            rows.append([display if control == "real" else "", label, *[number(values[c], 3) for c in CONTEXT.values()]])
    pops = cohort[cohort.method.isin(POP) & cohort.control.eq("real")]
    for method, part in pops.groupby("method"):
        if len(part) != 12:
            raise ValueError("Expression popularity controls need 12 real-input runs")
        values = part.groupby("dataset_id").auprc_trapezoid.mean()
        blocks[len(rows)] = ""
        rows.append([method, "None (prior only)", *[number(values[d], 3) for d in CONTEXT]])
    emit(build, "tab:expression_controls", ["Method / readout", "Expression input", *CONTEXT.values()], rows,
         "pooled trapezoid AUPRC",
         "Single fits with optimization seed 42 for every method. Groups rows by method with manuscript display names and input order. Preserves four context columns. "
         "Adds prior-only controls once because expression changes do not change their scores.", blocks=blocks)

def sensitivity(build):
    points = build.csv("summary/inducing_points.csv")
    rows = []
    for count, part in points.groupby("inducing_points"):
        if len(part) != 12:
            raise ValueError("Inducing-point setting missing one of 12 refits")
        rows.append([f"{int(count):,}", *[number(part[m].mean(), 4 if i < 3 else (1 if i == 3 else 0)) for i, m in enumerate(
            ("validation_average_precision", "test_average_precision", "test_auprc_trapezoid", "training_seconds", "peak_cuda_allocated_mb"))]])
    emit(build, "tab:gp_inducing_sensitivity", ["Inducing points", "Validation AP", "Test AP", "Test AUPRC", "Fit time (s)", "GPU (MiB)"], rows,
         "validation/test AP, test trapezoid AUPRC, seconds, allocated MiB",
         "Single fits with optimization seed 42. Inducing-point rows use the additive GP with early stopping and refit. Timing and memory come from these refits.")
    seeds = build.csv("summary/optimization_seeds.csv")
    rows = []
    for seed, part in seeds.groupby("seed"):
        if len(part) != 12:
            raise ValueError("Optimization seed missing context/split")
        grouped = part.groupby("context").auprc_trapezoid.mean()
        rows.append([int(seed), *[number(grouped[c]) for c in CONTEXT.values()], number(part.auprc_trapezoid.mean()), number(part.all_tf_ap.mean())])
    emit(build, "tab:optimization_seeds", ["Optimization seed", *CONTEXT.values(), "Overall AUPRC", "All-TF mean AP"], rows,
         "pooled trapezoid AUPRC; all-TF mean AP", "Each row is the single fit with one optimization seed; BEACON elsewhere is the mean of these three fits. Split-averaged seed rows and context columns.")

def response_tables(build):
    frame = tcell_evaluations(build)["per_tf_metrics"].query('method == "BEACON"')
    if frame.groupby("condition").positives.sum().to_dict() != {"resting": 3043, "stimulated": 1015}:
        raise ValueError("T-cell eligibility response totals must be 3,043 resting and 1,015 re-stimulated")
    rows = []
    for condition in ("resting", "stimulated"):
        metadata = build.csv(f"tcells/{condition}/eligibility.csv").set_index("gene_id")
        for entry in frame[frame.condition.eq(condition)].itertuples():
            if entry.tf not in metadata.index:
                raise ValueError(f"No donor/QC metadata for {entry.tf}")
            info = metadata.loc[entry.tf]
            rows.append(["Resting" if condition == "resting" else "Re-stimulated", info.symbol, int(info.Donor1_cells), int(info.Donor2_cells),
                         number(info.Donor1_on_target_fold, 2), number(info.Donor2_on_target_fold, 2), entry.pairs, entry.positives])
    emit(build, "tab:tcell_eligibility", ["Condition", "TF", "Cells D1", "Cells D2", "Fold D1", "Fold D2", "Targets", "Responses"], rows,
         "donor QC counts and configured tested/supported discovery counts",
         "T-cell eligibility: primary paired-donor definition. Donor metadata is unchanged; tested discovery denominators "
         "and responses come directly from the response evaluation: 3,043 resting and 1,015 re-stimulated responses.", True)


def specificity(build):
    from figures.recomputed_tables import generate
    generate(build)


def ablations(build, completion):
    from figures.render import COMPONENTS
    from figures.drawing import DS
    frame = completion.query('coverage == .8')
    rows = []
    for variant, method, label in COMPONENTS:
        part = frame[(frame.variant == variant) & (frame.method == method)]
        if len(part) != 12:
            raise ValueError(f"Incomplete ablation: {label}")
        rows.append((label, [part[part.dataset_id == d][m].mean() for m in ("auprc_trapezoid", "all_tf_ap") for d in DS]))
    best = [max(values[c] for _, values in rows) for c in range(8)]
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Pooled AUPRC} & \multicolumn{4}{c}{All-TF mean AP} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Model & hESC & mDC & mHSC-E & mESC & hESC & mDC & mHSC-E & mESC \\", r"\midrule"]
    for i, (label, values) in enumerate(rows):
        if label.startswith("Prior") and not rows[i - 1][0].startswith("Prior"):
            lines.append(r"\midrule")
        cells = [(r"\textbf{%.3f}" % v) if f"{v:.3f}" == f"{best[c]:.3f}" else f"{v:.3f}" for c, v in enumerate(values)]
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    build.table("ablations", "\n".join(lines), "pooled trapezoid AUPRC; all-TF mean AP",
                "Means across split seeds 14, 42 and 100 at 80% prior coverage; bold marks the highest value in each column.")


def runtime_stages(build):
    rows, records = [], build.bundle.manifest.get("timing", {})
    names = {"k562": "K562", "tcell_resting": "Resting T cells", "tcell_stimulated": "Re-stimulated T cells"}
    for context, label in names.items():
        record = records.get(f"{context}_three_seeds")
        if record is None:
            raise ValueError(f"Missing three-seed timing record: {context}")
        build.inputs.add(record)
        r = build.bundle.json(record)
        stages = pd.DataFrame(r["stage_seconds_per_fit"]).T
        median = lambda values: float(np.median(list(values)))
        loading = [r["fit_seconds"][s] - sum(r["stage_seconds_per_fit"][s].values()) for s in r["fit_seconds"]]
        rows.append([label, f"{r['problem_size']['candidate_pairs_excluding_self']:,}", number(r["total_seconds"], 1), number(r["fits_wall_seconds"], 1),
                     number(r["averaging_seconds"], 1), number(median(r["fit_seconds"].values()), 1), number(median(loading), 1),
                     *[number(stages[k].median(), 1) for k in ("factor_analysis", "representation_and_gp", "simple_classifiers", "candidate_scoring")],
                     number(median(r["final_gp_seconds_per_fit"].values()), 1),
                     f"{max(r['peak_rss_mib_per_fit'].values()):,.0f}", f"{max(r['peak_cuda_allocated_mib_per_fit'].values()):,.0f}"])
    emit(build, "tab:runtime_stages", ["Context", "Candidate pairs", "Total (s)", "Three fits (s)", "Averaging (s)", "Fit (s)",
                                       "Input loading (s)", "Factor analysis (s)", "Encoder and GP (s)", "Classifiers (s)", "Scoring (s)", "Final GP (s)",
                                       "Peak RSS (MiB)", "GPU (MiB)"], rows,
         "wall-clock seconds; peak memory per fit",
         "Three fits with optimization seeds 42, 14 and 100 run at the same time on one NVIDIA L40S with two CPU threads each; "
         "per-fit columns are medians over the three fits; Fit runs from loading the prepared input to the end of scoring, after the software imports; "
         "Total adds process start, imports and the per-pair averaging; Encoder and GP includes the internal validation fit, the encoder refit and the final GP fit.")
