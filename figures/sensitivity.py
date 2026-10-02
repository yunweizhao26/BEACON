"""Factor and input sensitivity table rows; K=64 stays prespecified."""
import re
import numpy as np
import pandas as pd
COLS = ["8", "16", "32", "64", "128", "256", "PCA64"]
ABBR = {"GNNLink": "GNNLink", "Topology control": "Topology", "Prior target in-degree": "In-degree",
        "Prior regulator out-degree": "Out-degree", "RegGAIN": "RegGAIN", "Target detection": "Detection",
        "Validation prevalence constant": "Constant"}
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

def strongest(frame, ref):
    r = frame[frame.k.astype(str) == ref].copy()
    r["key"] = np.where(r.direction == "lower", r.baseline_value, -r.baseline_value)
    return r.sort_values(["endpoint", "key", "baseline"]).groupby("endpoint").head(1).set_index("endpoint").baseline

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
        names = {"mean AUROC": "Mean AUROC", "mean AUPRC": "Mean AUPRC", "AUROC minus topology": "AUROC minus topology", "AUPRC minus topology": "AUPRC minus topology"}
        return gi, (list(names).index(x[0]),), names[x[0]]
    raise ValueError(label)

def rows_for(values, comps, cols, colkey, groups_seen=None):
    best = strongest(comps, "64")
    v = values.copy(); v["k"] = v[colkey].astype(str).replace({"FA64": "64"})
    c = comps.copy(); c["k"] = c.k.astype(str).replace({"FA64": "64"})
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

def generate(build, destination):
    kv = build.csv("factor_endpoints"); kc = build.csv("factor_comparisons")
    k_lines, k_best = rows_for(kv, kc, COLS, "k")
    iv = build.csv("input_endpoints"); ic = build.csv("input_comparisons")
    idiff = build.csv("input_differences")
    ICOLS = ["64", "PCA64", "scGPT_FA64", "scGPT_PCA64"]
    i_lines, i_best = rows_for(iv, ic, ICOLS, "input")
    # append paired input contrasts to each input row
    out, it = [], iter(i_lines)
    meta = iv.groupby("endpoint").label.first()
    keyed = [e for e, _ in sorted(meta.items(), key=lambda kv: order(kv[1])[:2])]
    j = 0
    for line in i_lines:
        if line.startswith(r"\multicolumn"):
            out.append(line.replace("{%d}" % (len(ICOLS) + 2), "{%d}" % (len(ICOLS) + 4))); continue
        e = keyed[j]; j += 1
        extra = []
        for left in ("PCA64", "scGPT_FA64"):
            r = idiff[(idiff.endpoint == e) & (idiff.left == left) & (idiff.right == "FA64")].iloc[0]
            d = r.difference
            txt = f"{d:+.3f}" if abs(d) >= 0.1 else f"{d:+.4f}" if abs(d) >= 0.001 else f"{d:+.5f}"
            extra.append(txt.replace("-", "$-$") + ARROW[r.interval_status])
        out.append(line[:-3].rstrip() + " & " + " & ".join(extra) + r" \\")
    i_lines = out
    build.table("factor_components", "\n".join(k_lines) + "\n", "Endpoint-specific metrics and paired intervals", "Prespecified 64-component reference")
    build.table("input_features", "\n".join(i_lines) + "\n", "Endpoint-specific metrics and paired input differences", "FA, PCA and frozen scGPT inputs")
    print(len(k_lines), len(i_lines)); print("\n".join(k_lines[:8])); print("\n".join(i_lines[-12:]))
