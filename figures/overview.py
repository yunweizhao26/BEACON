"""Figure 1: BEACON method and evaluation overview."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LINE = dict(color="#444444", lw=.75)
MAIN, TOP = 2.6, 4.9
EVALUATIONS = [(4.5, "Fixed candidate pools\nheld-out reference recovery\nin four BEELINE contexts"),
               (2.6, "K562\nperturbation response\nand promoter binding"),
               (.7, "Primary T cells and RPE1\nperturbation response\nTRRUST literature targets")]


def draw():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.15, 2.75), layout="constrained")
    ax.set_xlim(0, 15.1); ax.set_ylim(0, 6.6); ax.axis("off")

    def box(x, y, w, h, text, face="#F5F5F5", fs=5.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=.07",
                                    facecolor=face, edgecolor="#333333", linewidth=.75))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7, shrinkA=0, shrinkB=0, **LINE))

    def path(points):
        """Orthogonal connector: plain segments, then an arrowhead on the last segment."""
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, solid_capstyle="butt", **LINE)
        arrow(*points[-2], *points[-1])

    ax.text(.1, 6.25, "Inputs", weight="bold", fontsize=7.2)
    ax.text(6.6, 6.25, "BEACON model", ha="center", weight="bold", fontsize=8.2)
    ax.text(11.7, 6.25, "Evaluation", weight="bold", fontsize=7.2)

    box(.1, 4.3, 1.6, 1.2, "Partial GRN\nrecorded TF → target\nedges", face="#EEF1EE", fs=4.8)
    box(.1, 2.1, 1.6, 1.0, "Single-cell RNA-seq\ngenes × cells", face="#EEF1EE", fs=4.8)
    box(2.15, 4.25, 1.5, 1.3, "Labeled pairs\n$\\mathcal{P}$: observed edges\n$\\mathcal{N}$: sampled\nunlabeled pairs", face="#EEF1EE", fs=4.6)
    box(2.15, 2.1, 1.5, 1.0, "Factor analysis\ngene features $x_i$", face="#DDEFE9", fs=4.9)
    box(4.1, 4.2, 2.0, 1.4, "Pair decoder\n(trains the encoder)\n$[z_i,\\,z_j,\\,z_i\\odot z_j,\\,|z_i-z_j|]$\nclass-weighted\ncross-entropy", face="#DDEFE9", fs=4.6)
    box(4.1, 1.85, 2.0, 1.5, "Shared encoder $f_\\theta$\none representation\n$z_i=f_\\theta(x_i)$ per gene\nas regulator or target", face="#DDEFE9", fs=4.8)
    box(6.55, 1.7, 2.6, 1.8, "Sparse variational\nGP classifier\non ordered pairs $u_{ij}=(z_i, z_j)$\nkernel $\\kappa_{\\mathrm{reg}}+\\kappa_{\\mathrm{tgt}}+\\kappa_{\\mathrm{pair}}$\nepochs set on a 10% internal split,\nthen refit on all labeled pairs", face="#E8E1F0", fs=4.6)
    box(9.6, 2.05, 1.4, 1.1, "Ranked\nTF → target\ncandidates", face="#F2E6D9", fs=5.0)
    for y, text in EVALUATIONS:
        box(11.7, y - .55, 3.3, 1.1, text, fs=4.9)

    # Every link is horizontal or vertical.
    arrow(1.7, MAIN, 2.15, MAIN); arrow(3.65, MAIN, 4.1, MAIN); arrow(6.1, MAIN, 6.55, MAIN); arrow(9.15, MAIN, 9.6, MAIN)
    arrow(1.7, TOP, 2.15, TOP); arrow(3.65, TOP, 4.1, TOP)
    arrow(5.1, 3.35, 5.1, 4.2, style="<|-|>")
    path([(2.9, 5.55), (2.9, 5.9), (7.85, 5.9), (7.85, 3.5)])
    ax.text(7.95, 4.75, "labeled pairs\nfit the GP", fontsize=4.4, color="#444444", va="center")
    bus = 11.35
    ax.plot([11.0, bus], [MAIN, MAIN], **LINE)
    ax.plot([bus, bus], [EVALUATIONS[-1][0], EVALUATIONS[0][0]], solid_capstyle="butt", **LINE)
    for y, _ in EVALUATIONS:
        arrow(bus, y, 11.7, y)
    return fig
