import numpy as np
from evaluation import robustness as rb
from figures.tables import escape

TABLES = {"robustness_reverse_blocked": "reverse_blocked", "robustness_ten_splits": "ten_splits",
          "robustness_references": "references"}
DIGITS = {"ratio": 2}


def number(value, digits):
    return "--" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def signed(value, digits):
    return f"{float(value):+.{digits}f}"


def longtable(headers, blocks, alignment):
    width = len(headers)
    header = [r"\toprule", " & ".join(r"\textbf{" + escape(h) + "}" for h in headers) + r" \\", r"\midrule"]
    lines = [r"\begin{longtable}{" + alignment + "}", *header, r"\endfirsthead",
             r"\multicolumn{" + str(width) + r"}{l}{Table \thetable\ continued.} \\", *header, r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for i, (title, rows) in enumerate(blocks):
        if any(len(cells) != width for cells in rows):
            raise ValueError("Table row width differs from header")
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(width) + r"}{@{}l}{\textit{" + escape(title) + r"}} \\")
        lines += [" & ".join(cells) + r" \\" for cells in rows]
    return "\n".join([*lines, r"\end{longtable}", ""])


def bold_best(cells_by_method, values_by_method, digits):
    for column, metric in enumerate(rb.METRICS):
        shown = {m: round(float(v[metric]), digits[metric]) for m, v in values_by_method.items()}
        best = max(shown.values())
        for method, value in shown.items():
            if value == best:
                cells_by_method[method][column + 1] = r"\textbf{" + cells_by_method[method][column + 1] + "}"


def block_label(per_fit, dataset):
    row = per_fit[per_fit.dataset.eq(dataset)].iloc[0]
    return rb.label(row.reference, row.panel)


def three_split_body(result, part):
    per_fit, paired, settings = result["per_fit"], result["paired"], result["settings"]
    frame = per_fit[per_fit.part.eq(part)]
    digits = {m: DIGITS.get(m, 3) for m in rb.METRICS}
    blocks = []
    for (dataset, coverage), group in frame.groupby(["dataset", "coverage"], sort=False):
        splits = [s for s in (42, 14, 100) if s in set(group.split_seed)]
        if len(splits) != group.split_seed.nunique():
            raise ValueError("Three-split parts use split seeds 42, 14 and 100")
        title = f"{block_label(per_fit, dataset)}, {100 * coverage:g}% prior coverage, {len(splits)} splits"
        if part == "reverse_blocked":
            counts = settings[settings.part.eq(part) & settings.dataset.eq(dataset) & settings.coverage.eq(coverage)].set_index("split_seed")
            removed = [f"{int(counts.loc[s, 'removed_known_edges']):,}" for s in splits]
            title += ", with " + ", ".join(removed[:-1]) + " and " + removed[-1] + " known edges removed"
        means = group.groupby("method", sort=False)[list(rb.METRICS)].mean()
        compared = paired[paired.part.eq(part) & paired.dataset.eq(dataset) & paired.coverage.eq(coverage)].set_index(["method", "metric"])
        cells = {}
        for method in [m for m in rb.METHODS if m in means.index]:
            higher = "--" if method == "BEACON" else ", ".join(
                f"{int(compared.loc[(method, m), 'beacon_higher'])}/{int(compared.loc[(method, m), 'splits'])}" for m in rb.LEAD)
            cells[method] = [escape(method), *[number(means.loc[method, m], digits[m]) for m in rb.METRICS], higher]
        bold_best(cells, {m: means.loc[m] for m in cells}, digits)
        blocks.append((title, list(cells.values())))
    return longtable(["Method", *rb.METRICS.values(), "BEACON higher"], blocks, "l" + "r" * 6)


def ten_split_body(result):
    per_fit, paired = result["per_fit"], result["paired"]
    frame = per_fit[per_fit.part.eq("ten_splits")]
    digits = {m: DIGITS.get(m, 3) for m in rb.METRICS}
    blocks = []
    for (dataset, coverage), group in frame.groupby(["dataset", "coverage"], sort=False):
        title = f"{block_label(per_fit, dataset)}, {100 * coverage:g}% prior coverage, {group.split_seed.nunique()} splits"
        stats = group.groupby("method", sort=False)[list(rb.METRICS)].agg(["mean", "std"])
        cells, means = {}, {}
        for method in [m for m in rb.METHODS if m in stats.index]:
            cells[method] = [escape(method), *[f"{number(stats.loc[method, (m, 'mean')], digits[m])} ({number(stats.loc[method, (m, 'std')], digits[m])})"
                                               for m in rb.METRICS]]
            means[method] = {m: stats.loc[method, (m, "mean")] for m in rb.METRICS}
        bold_best(cells, means, digits)
        rows = list(cells.values())
        compared = paired[paired.part.eq("ten_splits") & paired.dataset.eq(dataset) & paired.coverage.eq(coverage)].set_index(["method", "metric"])
        for method in [m for m in rb.METHODS[1:] if m in cells]:
            rows.append([r"BEACON $-$ " + escape(method), *[
                f"{signed(compared.loc[(method, m), 'difference_mean'], digits[m])} [{signed(compared.loc[(method, m), 't_low'], digits[m])}, "
                f"{signed(compared.loc[(method, m), 't_high'], digits[m])}]" for m in rb.METRICS]])
            if method in rb.TESTED_METHODS:
                rows.append([r"\quad Sign-flip $P$", *[number(compared.loc[(method, m), "sign_flip_p"], 4) if m in rb.TESTED else "--"
                                                     for m in rb.METRICS]])
        blocks.append((title, rows))
    return longtable(["Method", *rb.METRICS.values()], blocks, "l" + "r" * 5)


def generate(build):
    selected = [name for name in TABLES if build.selected in (None, name)]
    if not selected:
        return
    result = rb.run(build.bundle)
    build.inputs.update(rb.records(build.bundle))
    result["per_fit"].to_csv(build.directory / "robustness_per_fit.csv", index=False)
    result["paired"].to_csv(build.directory / "robustness_paired.csv", index=False)
    result["settings"].to_csv(build.directory / "robustness_settings.csv", index=False)
    result["reverse_groups"].to_csv(build.directory / "robustness_reverse_groups.csv", index=False)
    metric = "pooled AUPRC (trapezoid), pooled AP, all-TF mean AP, within-regulator AP and specificity ratio"
    seeds = "BEACON and every trained comparator are scored by the mean of their predictions over optimization seeds 42, 14 and 100; the baselines are deterministic."
    if "robustness_reverse_blocked" in selected:
        build.table("robustness_reverse_blocked", three_split_body(result, "reverse_blocked"), metric,
                    "Means over split seeds 42, 14 and 100. Known edges whose reverse is a validation or test pair are removed and "
                    "training unlabeled pairs exclude such reverses. " + seeds + " BEACON higher: splits in which BEACON exceeds the "
                    "method on pooled AUPRC, pooled AP, all-TF mean AP and within-regulator AP. Bold marks the highest value in each block.")
    if "robustness_ten_splits" in selected:
        build.table("robustness_ten_splits", ten_split_body(result), metric,
                    "Mean (standard deviation) over split seeds 42, 14, 100 and 1-7; BEACON minus method with its 95% t interval over the "
                    "ten splits; exact two-sided sign-flip P over the 1,024 sign assignments. " + seeds + " Bold marks the highest mean in each block.")
    if "robustness_references" in selected:
        build.table("robustness_references", three_split_body(result, "references"), metric,
                    "Standard protocol at 80% prior coverage; means over split seeds 42, 14 and 100. " + seeds + " BEACON higher: splits in "
                    "which BEACON exceeds the method on pooled AUPRC, pooled AP, all-TF mean AP and within-regulator AP. Bold marks the "
                    "highest value in each block.")
