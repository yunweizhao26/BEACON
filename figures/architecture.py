#!/usr/bin/env python3
"""Draw the BEACON architecture diagram (encoder, training decoder and GP scorer) from beacon_model.py settings."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"

ENC, DEC, GP, IO = "#d7efe8", "#fbe3cf", "#e6e0f3", "#eef0f2"
ENC_EDGE, DEC_EDGE, GP_EDGE, IO_EDGE = "#1b9e77", "#d95f02", "#7570b3", "#5f6b76"
SHAPE = "#5f6b76"

# Parameter counts implied by beacon_model.py with 64 factor-analysis inputs and 16-dimensional representations.
ENCODER_PARAMS = (64 * 256 + 256) + (256 * 256 + 256) + (256 * 16 + 16) + 2 * 16
DECODER_PARAMS = (64 * 64 + 64) + (64 + 1)
GP_PARAMS = 500 * 32 + 500 + 500 * 501 // 2 + 7


def box(ax, x, y, w, h, title, sub=None, face=IO, edge=IO_EDGE, size=10, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=face, ec=edge, lw=1.3, zorder=3))
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=size, weight="bold" if bold else None, zorder=4)
        ax.text(x, y - h * 0.22, sub, ha="center", va="center", fontsize=size - 1.5, color="#333333", zorder=4, linespacing=1.3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=size, weight="bold" if bold else None, zorder=4)


def arrow(ax, start, end, color="#333333", style="-|>", dashed=False, lw=1.4):
    ax.annotate("", xy=end, xytext=start, zorder=2,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=13))


def panel(ax, x0, y0, x1, y1, face, edge, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=face, ec=edge, lw=1.2, alpha=0.35, zorder=0))
    ax.text(x0 + 1.2, y1 - 2.2, title, ha="left", va="center", fontsize=11.5, weight="bold", color=edge, zorder=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.text(50, 70.6, "BEACON architecture", ha="center", va="center", fontsize=15, weight="bold")

    # Gene encoder, shared by the regulator and target roles.
    cx = 16
    box(ax, cx, 63.5, 23, 5.5, "Single-cell RNA-seq X", "[G genes × C cells]")
    arrow(ax, (cx, 60.6), (cx, 56.4))
    ax.text(cx + 1, 58.5, "factor analysis, 64 factors", ha="left", va="center", fontsize=8.5, color=SHAPE, style="italic")
    box(ax, cx, 53.5, 23, 5.2, r"Gene features $x_i$", "[G × 64]")
    arrow(ax, (cx, 50.8), (cx, 47.4))
    ax.add_patch(FancyBboxPatch((cx - 13, 13.2), 26, 34.4, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=ENC, ec=ENC_EDGE, lw=1.6, ls="--", alpha=0.6, zorder=1))
    ax.text(cx, 45.2, r"Gene encoder $f_\theta$", ha="center", va="center", fontsize=11.5, weight="bold", color=ENC_EDGE, zorder=4)
    ax.text(cx, 42.7, "one set of weights for every gene", ha="center", va="center", fontsize=8.5, color=ENC_EDGE, zorder=4)
    layers = [("Linear 64 → 256", "ReLU"), ("Linear 256 → 256", "ReLU"), ("Linear 256 → 16", None), ("LayerNorm (16)", None)]
    ys = [38.3, 32.1, 25.9, 19.7]
    for (name, act), y in zip(layers, ys):
        box(ax, cx, y, 21, 4.4, name + (f" + {act}" if act else ""), face="white", edge=ENC_EDGE, size=9.5, bold=False)
    for y0, y1 in zip(ys, ys[1:]):
        arrow(ax, (cx, y0 - 2.4), (cx, y1 + 2.4))
    ax.text(cx, 15.3, f"{ENCODER_PARAMS:,} parameters", ha="center", va="center", fontsize=8.5, color=ENC_EDGE, zorder=4)
    arrow(ax, (cx, 12.6), (cx, 9.9))
    box(ax, cx, 6.6, 25, 6.2, r"Gene representation $z_i \in \mathbb{R}^{16}$",
        "[G × 16], used as regulator or target", face=ENC, edge=ENC_EDGE)

    # Representations feed both stages.
    bus = 32.5
    ax.plot([cx + 12.6, bus], [6.6, 6.6], color="#333333", lw=1.4, zorder=2)
    ax.plot([bus, bus], [6.6, 55.0], color="#333333", lw=1.4, zorder=2)
    arrow(ax, (bus, 55.0), (37.2, 55.0))
    arrow(ax, (bus, 19.5), (37.2, 19.5))
    ax.text(bus - 0.8, 29, r"$z_i$ (regulator), $z_j$ (target)", rotation=90, ha="center", va="center", fontsize=9, color="#333333")

    # Stage 1: pair decoder used only to train the encoder.
    panel(ax, 35.5, 38.5, 99.3, 68.5, DEC, DEC_EDGE, "Stage 1  ·  Train the encoder with a pair decoder (training only)")
    y = 55.0
    box(ax, 44.4, y, 13.8, 7.6, "Ordered pair (i, j)", "edge (y = 1) or sampled\nunlabeled pair (y = 0)", face="white", edge=DEC_EDGE, size=9.5)
    box(ax, 59.6, y, 14.5, 7.6, r"Pair features $a_{ij}$",
        r"$[z_i,\ z_j,\ z_i \odot z_j,\ |z_i - z_j|]$" + "\n[B × 64]", face="white", edge=DEC_EDGE, size=9.5)
    box(ax, 74.6, y, 13, 7.6, "Decoder MLP", "Linear 64 → 64 + ReLU\nLinear 64 → 1: logit $s_{ij}$", face="white", edge=DEC_EDGE, size=9.5)
    box(ax, 90.4, y, 15.6, 7.6, "Class-weighted BCE", "positive weight k = 5\nedges vs unlabeled pairs", face=DEC, edge=DEC_EDGE, size=9.5)
    for x0, x1 in [(51.6, 52.1), (67.1, 67.8), (81.4, 82.3)]:
        arrow(ax, (x0, y), (x1, y))
    arrow(ax, (90.4, 50.9), (90.4, 46.3), color="#c0392b", dashed=True, style="-")
    ax.plot([90.4, 30.0], [46.3, 46.3], color="#c0392b", lw=1.4, ls="--", zorder=2)
    arrow(ax, (30.0, 46.3), (29.4, 46.3), color="#c0392b")
    ax.text(62, 47.6, r"backpropagation updates the decoder and the shared encoder $f_\theta$",
            ha="center", va="center", fontsize=9, color="#c0392b")
    ax.text(67.4, 41.6, f"Minibatch: 32 observed edges + 160 sampled unlabeled pairs  ·  AdamW (lr 0.001, weight decay 0.0001)  ·  50 epochs\n"
            f"Decoder: {DECODER_PARAMS:,} parameters, discarded after training",
            ha="center", va="center", fontsize=8.8, color="#333333", linespacing=1.4)

    # Stage 2: sparse variational GP gives the scores.
    panel(ax, 35.5, 1.0, 99.3, 34.5, GP, GP_EDGE, "Stage 2  ·  Score every ordered pair with a sparse variational GP (output)")
    y = 19.5
    box(ax, 43.5, y, 12, 7.2, "Frozen encoder", r"$z_i,\ z_j$ fixed" + "\nafter Stage 1", face="white", edge=GP_EDGE, size=9.5)
    box(ax, 57.3, y, 12.5, 7.2, r"Pair input $u_{ij}$", r"concat$(z_i, z_j)$" + "\n[32], order-sensitive", face="white", edge=GP_EDGE, size=9.5)
    box(ax, 74.0, y, 17.5, 9.0, "Sparse variational GP",
        r"kernel $\sigma^2 \exp(-\Vert u-u' \Vert^2 / 2\ell^2)$" + "\nM = 500 learned inducing inputs\n" + r"$q(f_M) = \mathcal{N}(m, S)$",
        face="white", edge=GP_EDGE, size=9.5)
    box(ax, 91.3, y, 12.5, 7.2, "Probit score", r"$p_{ij} = \Phi\left(\mu_{ij} / \sqrt{1 + v_{ij}}\right)$" + "\nranked TF–target list",
        face=GP, edge=GP_EDGE, size=9.5)
    for x0, x1 in [(49.6, 50.9), (63.7, 65.1), (82.9, 84.9)]:
        arrow(ax, (x0, y), (x1, y))
    ax.text(67.4, 7.6, "Fitted on the same labeled pairs by maximizing the variational ELBO (Bernoulli likelihood, probit link)\n"
            f"AdamW (lr 0.001, weight decay 0.01)  ·  minibatch 1,024 pairs  ·  50 epochs  ·  {GP_PARAMS:,} parameters\n"
            "(inducing locations 16,000; variational mean and Cholesky factor 125,750; kernel scale, length scale and mean 3)",
            ha="center", va="center", fontsize=8.8, color="#333333", linespacing=1.4)

    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"beacon_architecture.{suffix}", dpi=300, bbox_inches="tight")
    print(OUT / "beacon_architecture.png", ENCODER_PARAMS, DECODER_PARAMS, GP_PARAMS)


if __name__ == "__main__":
    main()
