"""Figure 1: BEACON method and evaluation overview."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#233140"
DATA = ("#FFFFFF", INK)
LEARN = ("#EAF0F7", "#5B7DA8")
SCORE = ("#DFF1EF", "#2C7A7B")
OUTPUT = ("#FCF1D4", "#C49A2A")
EVALUATE = ("#DCEFDD", "#2F855A")
LEGEND = [(DATA, "Data"), (LEARN, "Representation learning"), (SCORE, "Pair scoring"),
          (OUTPUT, "Ranked output"), (EVALUATE, "Evaluation")]
TOP, BOTTOM, MIDDLE = (4.3, 5.7), (.65, 2.05), (2.55, 3.8)
EVALUATIONS = [((4.45, 5.7), "Held-out edges", "Fixed pools in four BEELINE contexts"),
               ((2.55, 3.8), "K562 assays", "Perturb-seq responses and promoter binding"),
               ((.65, 1.9), "Primary T cells and RPE1", "Perturb-seq responses and TRRUST target genes")]


def draw():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 16.3); ax.set_ylim(-.75, 6.85); ax.axis("off")

    def box(x0, x1, rows, title, note=None, style=LEARN, stacked=False, title_size=7.4, note_size=6.1, align="center"):
        y0, y1 = rows
        face, edge = style
        if stacked:
            for shift in (.16, .08):
                ax.add_patch(FancyBboxPatch((x0 - shift, y0 + shift), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=.14",
                                            facecolor="white", edgecolor=edge, linewidth=.9))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=.14",
                                    facecolor=face, edgecolor=edge, linewidth=1.1))
        x = (x0 + x1) / 2 if align == "center" else x0 + .18
        middle = (y0 + y1) / 2
        if note is None:
            ax.text(x, middle, title, ha=align, va="center", fontsize=title_size, weight="bold", linespacing=1.15)
        else:
            ax.text(x, middle + .06, title, ha=align, va="bottom", fontsize=title_size, weight="bold", linespacing=1.15)
            ax.text(x, middle - .06, note, ha=align, va="top", fontsize=note_size, linespacing=1.2)

    def block(x0, y, x1):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="simple,head_length=.55,head_width=.75,tail_width=.32",
                                     mutation_scale=9, facecolor="white", edgecolor=INK, linewidth=.9, shrinkA=1, shrinkB=1))

    def line(points, head=True, both=False):
        xs, ys = zip(*points)
        ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=INK, lw=.9, solid_capstyle="butt")
        if head:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="<|-|>" if both else "-|>", mutation_scale=8,
                                         color=INK, lw=.9, shrinkA=0, shrinkB=0))

    def header(x, text, ha="left"):
        ax.text(x, 6.55, text, ha=ha, va="center", fontsize=7, weight="bold", color="black")

    header(.05, "I N P U T S"); header(5.55, "B E A C O N   M O D E L", "center"); header(11.35, "E V A L U A T I O N")
    top, bottom = sum(TOP) / 2, sum(BOTTOM) / 2

    box(.2, 1.9, TOP, "Partial GRN", "known edges", DATA, stacked=True)
    box(.2, 1.9, BOTTOM, "scRNA-seq", "genes × cells", DATA, stacked=True)
    box(2.3, 4.1, TOP, "Labeled\npairs", "known edges,\nunlabeled pairs", DATA, stacked=True)
    box(2.3, 4.1, BOTTOM, "Factor\nanalysis", "gene features")
    box(4.5, 6.6, TOP, "Pair\ndecoder", "trains the encoder")
    box(4.5, 6.6, BOTTOM, "Shared\nencoder", "one vector per gene")
    box(7.0, 8.8, BOTTOM, "GP\nclassifier", r"$\kappa_{\mathrm{reg}}+\kappa_{\mathrm{tgt}}+\kappa_{\mathrm{pair}}$", SCORE)
    box(9.15, 10.8, MIDDLE, "Ranked\nTF → target\ncandidates", style=OUTPUT)
    for rows, title, note in EVALUATIONS:
        box(11.35, 16.25, rows, title, note, EVALUATE, align="left")

    block(1.9, top, 2.3); block(4.1, top, 4.5)
    block(1.9, bottom, 2.3); block(4.1, bottom, 4.5); block(6.6, bottom, 7.0)
    line([(5.55, BOTTOM[1]), (5.55, TOP[0])], both=True)
    line([(3.2, TOP[1]), (3.2, 6.15), (7.9, 6.15), (7.9, BOTTOM[1])])
    ax.text(7.75, 3.3, "labeled pairs\nalso fit the GP", fontsize=6.1, style="italic", va="center", ha="right")
    line([(8.8, bottom), (10.0, bottom), (10.0, MIDDLE[0])])
    bus = 11.05
    line([(10.8, sum(MIDDLE) / 2), (bus, sum(MIDDLE) / 2)], head=False)
    centers = [sum(rows) / 2 for rows, _, _ in EVALUATIONS]
    line([(bus, centers[-1]), (bus, centers[0])], head=False)
    for y in centers:
        line([(bus, y), (11.35, y)])

    # One-line legend, spaced by measured label widths and centered.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [ax.text(0, -.44, label, fontsize=6.6, va="center") for _, label in LEGEND]
    to_data = ax.transData.inverted()
    widths = [to_data.transform((t.get_window_extent(renderer).width, 0))[0] - to_data.transform((0, 0))[0] for t in labels]
    total = sum(.45 + w for w in widths) + .7 * (len(widths) - 1)
    x = 8.15 - total / 2
    for ((face, edge), _), text, width in zip(LEGEND, labels, widths):
        ax.add_patch(Rectangle((x, -.6), .32, .32, facecolor=face, edgecolor=edge, linewidth=1.0))
        text.set_x(x + .45)
        x += .45 + width + .7
    return fig
