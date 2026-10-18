import numpy as np
import pandas as pd
from evaluation import metrics as ev
from figures.build import CONTEXT, POP
from figures.tables import emit, number

SPLITS = (42, 14, 100)
POOLS = (("validation", "Validation pool"), ("test", "Test pool"))
ROLES = {"validation": ("validation_predictions", "decoder_validation_predictions"), "test": ("predictions", "decoder_predictions")}
SINGLE = "Single kernel on pair input"
ROWS = (("BEACON", "GP"), ("BEACON", "Decoder"), (SINGLE, "GP"), (SINGLE, "Decoder"), (POP[0], "Prior only"), (POP[1], "Prior only"))


def scores(build, ensemble=False):
    fits = {(f["dataset"], f["split_seed"]): f for f in build.bundle.manifest["kernel_ablation"]["fits"]}
    rows = []
    for dataset, context in CONTEXT.items():
        for split in SPLITS:
            fit = fits[(dataset, split)]
            if fit["seed"] != 42 or fit["coverage"] != .8:
                raise ValueError("The single-kernel fits are seed-42 fits at 80% prior coverage")
            train = build.case("fixed_pools", dataset, split, role="split")["train"]
            for pool, _ in POOLS:
                if ensemble:
                    gp, decoder = (build.ensemble("fixed_pools", dataset, split, role=role) for role in ROLES[pool])
                else:
                    gp, decoder = (build.case("fixed_pools", dataset, split, role=role) for role in ROLES[pool])
                build.inputs.add(fit[pool])
                single = build.bundle.arrays(fit[pool])
                edges, labels = gp["edges"], gp["labels"]
                for other in (decoder, single):
                    if not (np.array_equal(other["edges"], edges) and np.array_equal(other["labels"], labels)):
                        raise ValueError(f"{context} split {split} {pool}: the designs score different pairs")
                values = {("BEACON", "GP"): gp["beacon"], ("BEACON", "Decoder"): decoder["decoder"],
                          (SINGLE, "GP"): single["gp"], (SINGLE, "Decoder"): single["decoder"],
                          **{(name, "Prior only"): score for name, score in ev.popularity(train, edges).items()}}
                for (design, readout), score in values.items():
                    measured = ev.pool_metrics(labels, score, edges)
                    rows.append(dict(design=design, readout=readout, pool=pool, context=context, split_seed=split,
                                     ap=measured["average_precision"], all_tf_ap=measured["all_tf_ap"]))
    return pd.DataFrame(rows)


def generate(build, ensemble=False):
    frame = scores(build, ensemble)
    rows, blocks, reused, bold = [], {}, set(), set()
    for pool, title in POOLS:
        blocks[len(rows)] = title
        first = len(rows)
        for design, readout in ROWS:
            part = frame[(frame.pool == pool) & (frame.design == design) & (frame.readout == readout)]
            if len(part) != 12 or not part.groupby("context").size().eq(3).all():
                raise ValueError(f"Incomplete kernel ablation: {pool}, {design}, {readout}")
            context = part.groupby("context").ap.mean()
            if design == SINGLE:
                reused.add(len(rows))
            rows.append([design, readout, number(context.mean(), 3), *[number(context[c], 3) for c in CONTEXT.values()],
                         number(part.all_tf_ap.mean(), 3)])
        gp = [i for i in range(first, len(rows)) if rows[i][1] == "GP"]
        for j in range(2, 8):
            best = max(float(rows[i][j]) for i in gp)
            bold |= {(i, j) for i in gp if float(rows[i][j]) == best}
    emit(build, "tab:design_selection", ["Design", "Readout", "Mean", *CONTEXT.values(), "All-TF mean AP"], rows,
         "pooled AP per context and its four-context mean; all-TF mean AP; validation and test pools",
         ("BEACON three-seed mean" if ensemble else "BEACON seed-42 fits") + " and the saved single-kernel fits (seed 42, "
         "once per split) at 80% prior coverage, means over split seeds 14, 42 and 100; popularity rankings of the known "
         "edges on the same pairs; bold marks the highest GP-readout value in each column of each pool.", blocks=blocks, reused=reused, bold=bold)
    return frame
