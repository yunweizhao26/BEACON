import hashlib
import re
from evaluation.factor_components import tables, strongest

COLS = ["8", "16", "32", "64", "128", "256"]
ABBR = {"GNNLink": "GNNLink", "Topology control": "Degree", "Prior target in-degree": "Target pop.",
        "Prior regulator out-degree": "Regulator pop.", "RegGAIN": "RegGAIN", "Target detection": "Detection",
        "Validation prevalence constant": "Constant", "GENELink": "GENELink", "GCLink": "GCLink", "scRegulate": "scRegulate",
        "scRegNet": "scRegNet", "Inferelator": "Inferelator", "GENIE3": "GENIE3", "GRNBoost2": "GRNBoost2",
        "Expression correlation": "Expr. corr."}
ARROW = {"above zero": r"$^{\uparrow}$", "below zero": r"$^{\downarrow}$", "includes zero": ""}
def fmt(v, signed=False):
    if abs(v) >= 1:
        s = f"{v:.2f}"
    else:
        s = f"{v:#.3g}"
        if "e" in s:
            s = f"{v:.5f}"
    if signed and v > 0:
        s = "+" + s
    return s.replace("-", "$-$")

def order(label):
    ctx = ["hESC", "mDC", "mHSC-E", "mESC"]
    groups = [("Sampled pairs", r"^Sampled pairs: (.*)$"), ("Human sampled pairs", r"^Human sampled pairs: (.*)$"),
              ("Fixed pools, pooled AUPRC", r"^(\S+), (\d+)%: pooled AUPRC$"), ("Fixed pools, all-TF mean AP", r"^(\S+), (\d+)%: all-TF AP$"),
              ("Same-regulator concordance", r"^(\S+): same-regulator concordance$"), ("Calibrated Brier score, 80\\% coverage", r"^(\S+), 80%: calibrated Brier$"),
              ("SERGIO", r"^SERGIO (\w+), (\d+)%: (\w+)$"), ("K562", r"^K562: (.*)$"), ("Primary T cells", r"^T cells (\w+): (.*)$"),
              ("RPE1", r"^RPE1: (.*)$"), ("TRRUST", r"^TRRUST (.*): (.*)$")]
    for gi, (g, pat) in enumerate(groups):
        m = re.match(pat, label)
        if not m:
            continue
        x = m.groups()
        if g.startswith("Fixed"):
            return gi, (ctx.index(x[0]), int(x[1])), f"{x[0]}, {x[1]}\\%"
        if g.startswith(("Same", "Calibrated")):
            return gi, (ctx.index(x[0]),), x[0]
        if g == "SERGIO":
            met = {"ap_over_prevalence": "AP/prevalence", "auroc": "AUROC"}[x[2]]
            return gi, (x[0] != "sparse", int(x[1]), met), f"{x[0].capitalize()}, {x[1]}\\%, {met}"
        if g == "K562":
            k = ["response AP", "top-50 response fraction", "top-100 response fraction", "top-200 response fraction", "binding AP", "joint AP"]
            names = ["Response AP", "Top-50 response fraction", "Top-100 response fraction", "Top-200 response fraction", "Binding AP", "Joint AP"]
            return gi, (k.index(x[0]),), names[k.index(x[0])]
        if g == "Primary T cells":
            cond = {"resting": "Resting", "stimulated": "Re-stimulated"}[x[0]]
            met = {"AP": "AP", "top-100 response fraction": "top-100 fraction"}[x[1]]
            return gi, (x[0] != "resting", met != "AP"), f"{cond}, {met}"
        if g == "RPE1":
            k = ["top-100 mean absolute effect", "response AP"]; n = ["Top-100 mean absolute response", "Response AP"]
            return gi, (k.index(x[0]),), n[k.index(x[0])]
        if g == "TRRUST":
            sub = {"all": "all pairs", "overlap removed": "overlap removed"}[x[0]]
            met = {"same-regulator AUROC": "same-regulator AUROC", "AP": "AP"}[x[1]]
            return gi, (x[0] != "all", met == "AP"), f"{'AP' if met == 'AP' else 'AUROC'}, {sub}"
        names = {"mean AUROC": "Mean AUROC", "mean AUPRC": "Mean AUPRC", "AUROC minus topology": "AUROC minus degree baseline", "AUPRC minus topology": "AUPRC minus degree baseline"}
        return gi, (list(names).index(x[0]),), names[x[0]]
    raise ValueError(label)

def rows_for(values, comps, cols, colkey, groups_seen=None):
    best = strongest(comps)
    v = values.copy(); v["k"] = v[colkey].astype(str)
    c = comps.copy(); c["k"] = c.k.astype(str)
    lines, last = [], None
    meta = v.groupby("endpoint").label.first()
    keyed = sorted(meta.items(), key=lambda kv: order(kv[1])[:2])
    for endpoint, label in keyed:
        gi, _, name = order(label)
        group = ["Sampled pairs (44 settings)", "Human sampled pairs (12 settings)", "Fixed pools, pooled AUPRC", "Fixed pools, all-TF mean AP",
                 "Same-regulator concordance", "Calibrated Brier score, 80\\% coverage", "SERGIO simulations", "K562", "Primary T cells", "RPE1", "TRRUST"][gi]
        if group != last:
            lines.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (len(cols) + 2, group)); last = group
        b = best.get(endpoint)
        cells = [r"\quad " + name, ABBR[b] if b else "--"]
        for k in cols:
            val = v[(v.endpoint == endpoint) & (v.k == k)].value.iloc[0]
            mark = ""
            if b:
                st = c[(c.endpoint == endpoint) & (c.k == k) & (c.baseline == b)].interval_status.iloc[0]
                mark = ARROW[st]
            cells.append(fmt(val, signed="minus" in label) + mark)
        lines.append(" & ".join(cells) + r" \\")
    return lines, best

def generate(build, destination=None):
    result = tables(build.bundle)
    build.inputs.update(result["inputs"])
    kv, kc = result["endpoints"], result["comparisons"]
    k_lines, k_best = rows_for(kv, kc, COLS, "k")
    build.table("factor_components", "\n".join(k_lines) + "\n", "Endpoint-specific metrics and paired intervals", "Prespecified 64-component reference; single fits with optimization seed 42 for BEACON and every comparator")
    if build.selected is None or build.selected == "factor_components":
        for name in ("endpoints", "comparisons"):
            path = build.directory / f"factor_components_{name}.csv"
            with path.open("x") as stream:
                result[name].to_csv(stream, index=False)
            build.outputs.append(dict(output=path.name, metric="Table S13 " + name, notes="evaluation.factor_components",
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(len(k_lines)); print("\n".join(k_lines[:8]))
