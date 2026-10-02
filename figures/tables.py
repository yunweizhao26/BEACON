"""Manuscript table bodies with bundle-resolved evidence."""
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

def tabular(headers, rows, *, blocks=None, reused=(), longtable=False):
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Table row width differs from header")
    text_columns = {"Context", "Method", "Pool", "Design", "Readout", "Evaluation", "Method / readout",
                    "Expression input", "Score", "Condition", "TF"}
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
    emit(build, "tab:reference_dimensions", ["Context", "Genes", "Cells", "TFs", "Reference P", "Test pairs", "Test P", "P (%)"], rows,
         "protocol counts", "Protocol counts from run manifests.")
    emit(build, "tab:prior_edge_counts", ["Context", "5%", "10%", "20%", "40%", "80%"], count_rows,
         "revealed training-positive counts", "Original coverage columns; verifies counts across all three splits.")
    rows = []
    for dataset in (201, 205):
        for coverage in (".05", ".2", ".8"):
            entries = [build.case("sergio",dataset,seed,float(coverage)) for seed in (14, 42, 100)]
            hidden = [x["hidden_reference_positives_sampled_as_unlabeled"] for x in entries]
            first = entries[0]
            rows.append([first["reference_positives"], int(float(coverage) * 100), first["train_positive"], first["train_unlabeled"],
                         f"{min(hidden)}--{max(hidden)}", first["test_positives"]])
    emit(build, "tab:simulator_truth", ["Graph edges", "Prior (%)", "Train P", "Sampled U", "Hidden P in U", "Test P"], rows,
         "simulator truth and training counts", "Counts from simulator manifests.")

def trrust(build):
    ranking = build.csv("comparators/trrust/ranking_all_methods.csv")  # paired-monitor GRNBoost2
    paired = build.csv("comparators/trrust/ranking_paired.csv")
    per_tf = build.csv("comparators/trrust/ranking_all_methods_per_tf.csv")
    rows, blocks = [], {}
    for evaluation, suffix in (("all", "baseline_comparison"), ("overlap_removed", "baseline_comparison_novel_to_dorothea")):
        cohort = per_tf[(per_tf.evaluation == evaluation) & (per_tf.method == "BEACON")][["tf", "positives", "pairs"]].drop_duplicates()
        if cohort.empty or cohort.tf.duplicated().any():
            raise ValueError(f"Inconsistent TRRUST cohort counts: {evaluation}")
        title = "All TRRUST relationships" if evaluation == "all" else "DoRothEA overlap removed"
        blocks[len(rows)] = (f"{title}: {len(cohort):,} regulators, {int(cohort.positives.sum()):,} supported pairs, "
                             f"{int(cohort.pairs.sum()):,} eligible pairs")
        comparison = build.csv(f"comparators/trrust/{suffix}/method_comparison.csv").set_index("method")
        for entry in ranking[ranking.evaluation.eq(evaluation)].sort_values("auroc", ascending=False, kind="stable").itertuples():
            difference, pvalue = "--", "--"
            if entry.method != "BEACON":
                if entry.method in comparison.index:
                    item = comparison.loc[entry.method]
                    difference = f"{item.beacon_minus_method_auc:.4f} [{item.difference_ci_low:.4f}, {item.difference_ci_high:.4f}]"
                    pvalue = number(item.one_sided_signflip_p)
                else:
                    item = paired[(paired.evaluation == evaluation) & (paired.comparator == entry.method) & (paired.metric == "auroc")].iloc[0]
                    difference = f"{item['mean']:.4f} [{item.ci_low:.4f}, {item.ci_high:.4f}]"
            rows.append(["Inferelator 3.0" if entry.method == "Inferelator" else entry.method,
                         number(entry.auroc), difference, pvalue, number(entry.average_precision)])
    emit(build, "tab:trrust_comparison", ["Method", "Macro AUROC", "Difference [95% CI]", "One-sided P", "Mean AP"], rows,
         "within-regulator AUROC and AP (equal regulator/seed weight)",
         "Five columns with cohort counts in each overlap-block heading; methods ordered by descending AUROC within each block. "
         "Adds both prior-only controls; out-degree is explicitly constant within each regulator. P values for new controls "
         "are -- because the evaluation reports bootstrap intervals but no sign-flip P values for them.", blocks=blocks)

def scregnet(build, completion):
    frame = completion.query('dataset_id == 1709 and coverage == .8 and ratio == 5 and corruption == 0 and variant == "beacon"')
    selected = frame[frame.method.isin(["beacon", *POP])].copy()
    added = []
    for seed in (14, 42, 100):
        z = build.comparator("scregnet",1709,seed)
        ref = build.case("fixed_pools",1709,seed,role="predictions")
        if not np.array_equal(z["edges"], ref["edges"]) or not np.array_equal(z["labels"], ref["labels"]):
            raise ValueError("scRegNet pool differs from BEACON")
        keys = [k for k in z if k not in ("edges", "labels") and not k.startswith("validation_")]
        if len(keys) != 1:
            raise ValueError(f"Ambiguous scRegNet score: {keys}")
        added.append(dict(method="scRegNet", **ev.pool_metrics(z["labels"], z[keys[0]], z["edges"])))
    selected = pd.concat([selected, pd.DataFrame(added)], ignore_index=True)
    rows = []
    for method, part in selected.groupby("method", sort=False):
        if len(part) != 3:
            raise ValueError("scRegNet table requires three splits")
        rows.append(["BEACON" if method == "beacon" else method, *[
            f"{part[m].mean():.4f} [{part[m].min():.4f}, {part[m].max():.4f}]" for m in ("auprc_trapezoid", "all_tf_ap", "auroc")]])
    emit(build, "tab:scregnet_fixed_pool", ["Method", "Pooled AUPRC", "All-TF mean AP", "AUROC"], rows,
         "trapezoid pooled AUPRC; all-TF mean AP; AUROC",
         "mHSC-E means and split ranges; BEACON and both prior controls are recomputed, and saved scRegNet scores are checked against the same pair identities.")

def expression(build, metrics):
    frame = metrics[(metrics.suite == "expression_controls") & metrics.method.isin(
        ["beacon_gp", "beacon_decoder", "gnnlink", "genelink", "gclink"]) & metrics.run.str.match(
        r"c\.8/DS\d+_s\d+_beacon_(real|cell_shuffled|gene_permuted|random)$")].copy()
    frame["dataset_id"] = frame.run.str.extract(r"DS(\d+)")[0].astype(int)
    frame["context"] = frame.dataset_id.map(CONTEXT)
    frame["control"] = frame.run.str.extract(r"_beacon_(.*)$")[0]
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
    pops = metrics[(metrics.suite == "expression_controls") & metrics.method.isin(POP) & metrics.run.str.match(r"c\.8/DS\d+_s\d+_beacon_real$")].copy()
    pops["dataset_id"] = pops.run.str.extract(r"DS(\d+)")[0].astype(int)
    for method, part in pops.groupby("method"):
        if len(part) != 12:
            raise ValueError("Expression popularity controls need 12 real-input runs")
        values = part.groupby("dataset_id").auprc_trapezoid.mean()
        blocks[len(rows)] = ""
        rows.append([method, "Prior only (expression independent)", *[number(values[d], 3) for d in CONTEXT]])
    emit(build, "tab:expression_controls", ["Method / readout", "Expression input", *CONTEXT.values()], rows,
         "pooled trapezoid AUPRC",
         "Groups rows by method with manuscript display names and input order. Preserves four context columns. "
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
         "Inducing-point rows use the additive GP with early stopping and refit. Timing and memory come from these refits.")
    seeds = build.csv("summary/optimization_seeds.csv")
    rows = []
    for seed, part in seeds.groupby("seed"):
        if len(part) != 12:
            raise ValueError("Optimization seed missing context/split")
        grouped = part.groupby("context").auprc_trapezoid.mean()
        rows.append([int(seed), *[number(grouped[c]) for c in CONTEXT.values()], number(part.auprc_trapezoid.mean()), number(part.all_tf_ap.mean())])
    emit(build, "tab:optimization_seeds", ["Optimization seed", *CONTEXT.values(), "Overall AUPRC", "All-TF mean AP"], rows,
         "pooled trapezoid AUPRC; all-TF mean AP", "Split-averaged seed rows and context columns.")

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

