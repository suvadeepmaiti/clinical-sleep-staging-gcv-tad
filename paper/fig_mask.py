"""Sleep-mask deployment of the student (writes figs/mask.pdf and figs/mask.png).

Two-column width (7 in). Layout coordinates are in pixels of a 2000 x 880 canvas (y downward). Same palette and
helpers as fig_pipeline.py. Memory and compute figures are the estimates for the INT8 EEG+EOG student (not measured
on a device).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

from fig_pipeline import (GREY, IN_BG, LOS_BG, LOS_FILL, LOS_LINE, STU_BG, STU_FILL, STU_LINE, arrow, box, text)

HERE = Path(__file__).resolve().parent
MASK_FILL, MASK_LINE, CUP_FILL = "#E3E6EF", "#B9BFD0", "#D3D8E5"


def electrode(ax, x, y, fc, label, dx=0, dy=-38):
    ax.add_patch(Circle((x, y), 15, fc=fc, ec="black", lw=0.8, zorder=6))
    text(ax, x + dx, y + dy, label, 7.3, bold=True)


def polyarrow(ax, pts, color=GREY, lw=1.3):
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw, zorder=5,
            solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color, lw)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.0, 7.0 * 880 / 2010))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-5, 2005)
    ax.set_ylim(875, -5)
    ax.axis("off")

    # Sleep mask
    box(ax, 0, 2, 470, 790, IN_BG)
    text(ax, 235, 42, "Sleep mask", 10.5, bold=True)
    ax.plot([0, 60], [400, 395], color=MASK_LINE, lw=6, solid_capstyle="round", zorder=2)
    ax.plot([410, 470], [395, 400], color=MASK_LINE, lw=6, solid_capstyle="round", zorder=2)
    ax.add_patch(FancyBboxPatch((55, 250), 360, 250, boxstyle="round,pad=0,rounding_size=110", fc=MASK_FILL,
                                ec=MASK_LINE, lw=1.2, zorder=3))
    for cx in (155, 315):
        ax.add_patch(Ellipse((cx, 415), 135, 95, fc=CUP_FILL, ec=MASK_LINE, lw=0.8, zorder=4))
    electrode(ax, 165, 305, STU_FILL, "Fp1")
    electrode(ax, 305, 305, STU_FILL, "Fp2")
    electrode(ax, 235, 290, "white", "REF", dy=-36)
    electrode(ax, 82, 420, LOS_FILL, "E1", dx=0, dy=55)
    electrode(ax, 388, 420, LOS_FILL, "E2", dx=0, dy=55)
    text(ax, 235, 545, "soft electrodes in the lining,\nelectronics module on the strap", 7.3, italic=True)
    for y, fc, lab in ((640, STU_FILL, "EEG (frontal)"), (690, LOS_FILL, "EOG (outer canthi)"),
                       (740, "white", "reference / ground")):
        ax.add_patch(Circle((95, y), 13, fc=fc, ec="black", lw=0.8, zorder=6))
        text(ax, 125, y, lab, 7.5, ha="left")
    arrow(ax, (473, 400), (517, 400))

    # Analog front-end
    box(ax, 522, 240, 768, 560, "white", GREY, 1.2)
    text(ax, 645, 300, "Analog\nfront-end", 8.5, bold=True)
    text(ax, 645, 440, "instrumentation\namplifier\n24-bit $\\Sigma\\Delta$ ADC\n2 ch, 250 Hz", 7.5, ls=1.3)
    arrow(ax, (771, 400), (815, 400))
    text(ax, 793, 370, "SPI", 6.8)

    # Microcontroller
    box(ax, 820, 2, 1590, 790, STU_BG)
    text(ax, 850, 42, "Microcontroller · Cortex-M4F class", 9.5, bold=True, ha="left")
    for (x0, x1, title, body) in ((845, 1065, "Resample", "anti-alias low-pass\ndecimate to 100 Hz"),
                                  (1105, 1310, "Normalise", "z-score, fixed\ntraining constants"),
                                  (1350, 1565, "Epoch buffer", "2 × 3000 samples\nDMA double buffer")):
        box(ax, x0, 95, x1, 235, "white", STU_LINE, 1.2)
        text(ax, (x0 + x1) / 2, 132, title, 8, bold=True)
        text(ax, (x0 + x1) / 2, 190, body, 6.1)
    arrow(ax, (1068, 165), (1102, 165), STU_LINE)
    arrow(ax, (1313, 165), (1347, 165), STU_LINE)
    polyarrow(ax, [(1457, 238), (1457, 268), (955, 268), (955, 345)], STU_LINE)
    text(ax, 1206, 290, "every 30 s: wake CPU", 7, italic=True)

    box(ax, 845, 310, 1565, 560, STU_FILL)
    text(ax, 1205, 335, "Student CNN-GRU · INT8 · 31k parameters", 8.5, bold=True)
    for (x0, x1, lab) in ((865, 1045, "3 × Conv1D"), (1075, 1175, "pool\n32"),
                          (1205, 1335, "GRU\n64"), (1365, 1545, "linear\n5")):
        box(ax, x0, 365, x1, 470, "white", STU_LINE, 1.0)
        text(ax, (x0 + x1) / 2, 400 if x0 == 865 else 418, lab, 7.4, bold=True)
    text(ax, 955, 448, "16@50 · 32@10 · 32@5", 5.6)
    for (a, b) in ((1045, 1075), (1175, 1205), (1335, 1365)):
        arrow(ax, (a + 2, 418), (b - 2, 418), STU_LINE)
    ax.add_patch(FancyArrowPatch((1225, 473), (1315, 473), connectionstyle="arc3,rad=0.9", arrowstyle="-|>",
                                 mutation_scale=9, color="#6FB7A9", lw=1.8, zorder=5))
    text(ax, 1270, 537, r"$h_{t-1} \rightarrow h_t$ kept between epochs", 7.2)

    for (x0, x1, title, body) in ((845, 1075, "Flash", "$\\approx$32 KB weights\n+ code"),
                                  (1095, 1325, "RAM", "$\\approx$12 KB peak\n64-value GRU state"),
                                  (1345, 1565, "Compute", "2.7M MACs / epoch\n$\\approx$0.1% duty cycle")):
        box(ax, x0, 590, x1, 760, "white", STU_LINE, 1.2)
        text(ax, (x0 + x1) / 2, 632, title, 8, bold=True)
        text(ax, (x0 + x1) / 2, 700, body, 6.3)

    # Output
    box(ax, 1640, 2, 1998, 790, LOS_BG)
    text(ax, 1819, 42, "Output", 10.5, bold=True)
    box(ax, 1662, 110, 1976, 250, "white", LOS_LINE, 1.2)
    text(ax, 1819, 148, "On-mask log", 8, bold=True)
    text(ax, 1819, 205, "1 byte per epoch\n$\\approx$1 KB per night", 7.4)
    box(ax, 1662, 330, 1976, 510, LOS_FILL)
    text(ax, 1819, 375, r"Sleep stage $\hat{y}_t$", 8.5, bold=True)
    text(ax, 1819, 445, "W / N1 / N2 / N3 / REM\nevery 30 s, causal\n(no future epochs)", 7.4)
    box(ax, 1662, 590, 1976, 760, "white", LOS_LINE, 1.2)
    text(ax, 1819, 630, "Bluetooth LE", 8, bold=True)
    text(ax, 1819, 695, "phone or clinician app\nlive hypnogram", 7.4)
    polyarrow(ax, [(1568, 418), (1610, 418), (1610, 420), (1659, 420)], LOS_LINE)
    arrow(ax, (1819, 327), (1819, 253), LOS_LINE)
    arrow(ax, (1819, 513), (1819, 587), LOS_LINE)

    # Power
    box(ax, 0, 815, 1998, 873, "#F4F5F9")
    text(ax, 999, 844, "Li-Po battery + power management · CPU sleeps between epochs · front-end and radio dominate "
                       "the energy budget; inference $\\approx$0.1–0.3 mJ per epoch (estimate)", 7.4, italic=True)

    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs/mask.pdf")
    fig.savefig(HERE / "figs/mask.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
