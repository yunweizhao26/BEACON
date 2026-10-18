from evaluation.trrust import endpoints

TITLES = {"all": "All TRRUST relationships", "overlap_removed": "DoRothEA overlap removed"}
PANEL_METHODS = {"BEACON": "BEACON", "GNNLink": "GNNLink", "GENELink": "GENELink", "GRNBoost2": "GRNBoost2", "GENIE3": "GENIE3",
                 "Inferelator": "Inferelator", "Degree logistic": "Topology control", "Prior target in-degree": "Prior target in-degree",
                 "Prior regulator out-degree": "Prior regulator out-degree"}


def _results(build):
    result = endpoints(build.bundle)
    build.inputs.update(result["inputs"])
    return result


def table_body(build):
    from figures.tables import number, tabular
    result = _results(build)
    rows, blocks = [], {}
    for evaluation, title in TITLES.items():
        cohort = result["cohorts"].set_index("evaluation").loc[evaluation]
        blocks[len(rows)] = (f"{title}: {int(cohort.regulators):,} regulators, {int(cohort.supported):,} supported pairs, "
                             f"{int(cohort.candidates):,} eligible pairs")
        ranking = result["ranking"].query("evaluation == @evaluation")
        comparison = result["method_comparison"].query("evaluation == @evaluation and metric == 'auroc'").set_index("method")
        for entry in ranking.sort_values("auroc", ascending=False, kind="stable").itertuples():
            difference, pvalue = "--", "--"
            if entry.method != "BEACON":
                item = comparison.loc[entry.method]
                difference = f"{item.difference:.4f} [{item.ci_low:.4f}, {item.ci_high:.4f}]"
                pvalue = number(item.one_sided_signflip_p)
            rows.append(["Inferelator 3.0" if entry.method == "Inferelator" else entry.method,
                         number(entry.auroc), difference, pvalue, number(entry.average_precision)])
    if not rows:
        raise ValueError("Empty manuscript table")
    text = tabular(["Method", "Macro AUROC", "Difference [95% CI]", "One-sided P", "Mean AP"], rows, blocks=blocks)
    build.table("trrust_comparison", text, "within-regulator AUROC and AP of seed-mean scores (equal regulator weight)",
                "Five columns with cohort counts in each block heading; methods ordered by descending AUROC within each block. "
                "Scores are means over the five task seeds; the deterministic baselines are computed once. "
                "Regulator popularity is constant within each regulator.")


def panel(build):
    result = _results(build)
    ranking = {e: result["ranking"].query("evaluation == @e").set_index("method") for e in TITLES}
    value = {metric: {shown: [float(ranking[e].loc[method, metric]) for e in TITLES] for method, shown in PANEL_METHODS.items()}
             for metric in ("auroc", "average_precision")}
    per_tf = result["per_tf_metrics"].query("method == 'BEACON'")
    value["chance"] = [float(per_tf.loc[per_tf.evaluation.eq(e), "expected_random_ap"].mean()) for e in TITLES]
    return value
