"""Fig. 1 (pipeline of the G_cv-TAD teacher-student framework; writes figs/pipeline.pdf and figs/pipeline.png).

Two-column width (7 in). Layout coordinates are in pixels of a 2000 x 910 canvas (y downward). Light fills, black text.
The EEG traces are illustrative (synthetic), not recorded data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
INK = "black"
GREY = "#8A90A6"
IN_BG = "#F4F5F9"
TEA_BG, TEA_LINE, TEA_FILL = "#F2EFFA", "#A99DD8", "#D3CBF0"
TEA_SLABS = ("#E2DCF5", "#D7CFF1", "#CCC2EC", "#C1B5E7")
STU_BG, STU_LINE, STU_FILL = "#EAF6F3", "#86C7BB", "#B5E0D7"
STU_BLOCKS = ("#CDEAE4", "#B3DFD6", "#9AD3C8")
LOS_BG, LOS_LINE, LOS_FILL = "#FDF3EB", "#EDB088", "#F7CFB1"


def box(ax, x0, y0, x1, y1, fc, ec="none", lw=1.0, r=16):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def text(ax, x, y, s, size=7.5, bold=False, italic=False, ha="center", ls=1.2):
    ax.text(x, y, s, fontsize=size, color=INK, ha=ha, va="center", fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal", linespacing=ls, zorder=6)


def arrow(ax, a, b, color=GREY, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, zorder=5,
                                 shrinkA=0, shrinkB=0))


def pill(ax, x0, y0, x1, y1, s, fc):
    box(ax, x0, y0, x1, y1, fc, r=(y1 - y0) / 2)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2, s, 6.6, bold=True)


def block3d(ax, x, y, w, h, d, fc, side):
    """Cuboid with front face (x, y, w, h) and depth d (up and to the right)."""
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], fc=fc, ec=side, lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], fc=side, ec=side,
                         lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fc=fc, ec=side, lw=0.6, zorder=4))


def trace(ax, x0, y0, x1, y1, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 220)
    s = sum(rng.uniform(0.4, 1) * np.sin(2 * np.pi * (f * t + rng.uniform())) for f in rng.uniform(4, 20, 6))
    s = s + 0.5 * rng.standard_normal(len(t))
    ax.plot(x0 + t * (x1 - x0), (y0 + y1) / 2 - 0.45 * (y1 - y0) * s / np.abs(s).max(), color=INK, lw=0.8, zorder=4)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.0, 7.0 * 910 / 2010))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-5, 2005)
    ax.set_ylim(905, -5)
    ax.axis("off")

    # Input
    box(ax, 0, 2, 315, 793, IN_BG)
    text(ax, 158, 42, "Input", 10.5, bold=True)
    text(ax, 158, 115, "consecutive 30-s epochs,\n100 Hz", 7.5)
    for y0, lab, seed in ((205, r"epoch $t-1$", 3), (423, r"epoch $t$", 7)):
        box(ax, 30, y0, 288, y0 + 167, "white", "#D5D8E3", 0.9)
        trace(ax, 45, y0 + 15, 273, y0 + 105, seed)
        text(ax, 158, y0 + 132, lab, 7.5)
    text(ax, 158, 648, "Teacher:\nC3, C4, EMG, EOG", 7.5, bold=True)
    text(ax, 158, 745, "Student:\nC4 and/or E1", 7.5, bold=True)
    arrow(ax, (318, 187), (366, 187))
    arrow(ax, (318, 607), (366, 607))

    # Teacher
    box(ax, 372, 2, 1238, 372, TEA_BG)
    text(ax, 405, 40, "Teacher · SleepGPT-BiLSTM · 136M", 9.5, bold=True, ha="left")
    pill(ax, 1068, 27, 1213, 67, "FROZEN", TEA_FILL)
    box(ax, 405, 115, 590, 283, "white", TEA_LINE, 1.2)
    text(ax, 497, 199, "Patch +\nSTFT\ntokens", 7.5, bold=True)
    arrow(ax, (592, 198), (638, 198), TEA_LINE)
    for i, c in enumerate(TEA_SLABS):
        block3d(ax, 648 + 54 * i, 122, 54, 148, 12, c, "#A99DD8")
    text(ax, 765, 297, "12 transformer layers", 7.3, bold=True)
    arrow(ax, (895, 198), (958, 198), TEA_LINE)
    box(ax, 963, 115, 1205, 283, TEA_FILL)
    text(ax, 1084, 185, "BiLSTM", 8.5, bold=True)
    text(ax, 1084, 219, "21-epoch window", 7.3)
    text(ax, 808, 336, "fine-tuned with CE on the 25% labelled patients, then frozen", 7.3, italic=True)

    # Student
    box(ax, 372, 422, 1238, 793, STU_BG)
    text(ax, 405, 461, "Student · causal CNN-GRU · 30k", 9.5, bold=True, ha="left")
    pill(ax, 1052, 447, 1213, 487, "TRAINED", STU_FILL)
    for (x, y, w, h, d, c, lab) in ((412, 545, 41, 152, 10, STU_BLOCKS[0], "16@50"),
                                    (512, 563, 58, 117, 12, STU_BLOCKS[1], "32@10"),
                                    (612, 577, 58, 89, 12, STU_BLOCKS[2], "32@5")):
        block3d(ax, x, y, w, h, d, c, "#6FB7A9")
        text(ax, x + w / 2 + 5, 729, lab, 7.3, bold=True)
    arrow(ax, (697, 611), (720, 611), STU_LINE)
    box(ax, 728, 552, 850, 672, "white", STU_LINE, 1.2)
    text(ax, 789, 612, "pool\n32", 7.5, bold=True)
    arrow(ax, (852, 611), (893, 611), STU_LINE)
    box(ax, 898, 552, 1027, 672, STU_FILL)
    text(ax, 962, 612, "GRU\n64", 8, bold=True)
    ax.add_patch(FancyArrowPatch((918, 545), (1016, 545), connectionstyle="arc3,rad=-0.9", arrowstyle="-|>",
                                 mutation_scale=9, color="#6FB7A9", lw=2.0, zorder=5))
    text(ax, 970, 715, r"$h_{t-1} \rightarrow h_t$", 7.5)
    arrow(ax, (1029, 611), (1070, 611), STU_LINE)
    box(ax, 1076, 552, 1206, 672, "white", STU_LINE, 1.2)
    text(ax, 1141, 612, "linear\n5", 7.5, bold=True)
    text(ax, 405, 760, "filters@kernel · one epoch at a time · real-time streaming", 7.3, italic=True, ha="left")

    # Outputs
    arrow(ax, (1240, 187), (1281, 187), TEA_LINE)
    box(ax, 1287, 67, 1537, 308, "white", TEA_LINE, 1.4)
    text(ax, 1411, 146, "Soft labels", 8, bold=True)
    text(ax, 1411, 192, r"$p_{t-1},\ p_t$", 9.5)
    text(ax, 1411, 228, r"$T = 2$", 8)
    arrow(ax, (1540, 187), (1588, 187), TEA_LINE)
    arrow(ax, (1240, 607), (1281, 607), STU_LINE)
    box(ax, 1287, 487, 1537, 728, "white", STU_LINE, 1.4)
    text(ax, 1411, 587, "Predictions", 8, bold=True)
    text(ax, 1411, 632, r"$q_{t-1},\ q_t$", 9.5)
    arrow(ax, (1540, 607), (1588, 607), STU_LINE)

    # Transition-aware distillation
    box(ax, 1595, 2, 1998, 793, LOS_BG)
    text(ax, 1796, 60, "Transition-aware\ndistillation", 9.5, bold=True, ls=1.15)
    box(ax, 1618, 131, 1974, 275, "white", LOS_LINE, 1.2)
    text(ax, 1796, 188, "confidence", 8, bold=True)
    text(ax, 1796, 225, r"$c_{t-1} = \mathrm{max}_k\ p_{t-1,k}$", 8)
    box(ax, 1618, 300, 1974, 400, "white", LOS_LINE, 1.2)
    text(ax, 1796, 336, "previous stage", 8, bold=True)
    text(ax, 1796, 372, r"$s = \mathrm{argmax}_k\ p_{t-1,k}$", 8)
    box(ax, 1618, 425, 1974, 570, "white", LOS_LINE, 1.2)
    text(ax, 1796, 467, "disease-aware prior", 8, bold=True)
    text(ax, 1796, 517, "$G_{cv}[s,k]$ from labelled\npatients", 7.8)
    box(ax, 1618, 600, 1974, 760, LOS_FILL)
    text(ax, 1796, 660, "Student loss", 8.5, bold=True)
    text(ax, 1796, 703, r"$0.3\,\mathcal{L}_{\mathrm{CE}} + 0.2\,\mathcal{L}_{G_{cv}}$", 9.5)

    # Gradient path
    dash = "#E8A276"
    ax.plot([1796, 1796, 800], [795, 838, 838], color=dash, lw=1.3, ls=(0, (4, 3)), zorder=4)
    ax.plot([800, 800], [838, 812], color=dash, lw=1.3, ls=(0, (4, 3)), zorder=4)
    arrow(ax, (800, 815), (800, 797), dash)
    text(ax, 1298, 872, "gradients update the student only", 8, italic=True)

    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs/pipeline.pdf")
    fig.savefig(HERE / "figs/pipeline.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
