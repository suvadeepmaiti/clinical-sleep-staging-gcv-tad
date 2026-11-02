"""Fig. 3 (distillation map, paper version of slides/fig_distillation_map.png: panels A, B, C, E) and its macros.

Reads <results>/analysis/distillation_map.json (built by analysis/distillation_map.py): iSLEEPS fold 1, the unlabelled training
patient with the median G_cv-TAD-vs-CE outcome. Writes figs/distill_map.pdf and figs/distill_numbers.tex
(\\dmap{key} for the example night, \\dstat{cohort}{key} for whole-cohort shares in %: disagreements won
(gright, ceright), at/near a change (nearchange), epochs and loss weight at stage changes (echange, wchange) and on
scorer N1 (enone, wnone)).
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
D = json.loads((repo_paths.ANALYSIS / "distillation_map.json").read_text())
INK, MUTED = "#1B2140", "#5B6380"
STAGE_COL = ["#E9C46A", "#8ECAE6", "#5E7FB8", "#1B2140", "#C0392B"]  # W, N1, N2, N3, REM (as in the slides)
OURS, CE, GREEN, RED = "#E8833A", "#9AA3B5", "#0E9384", "#D62828"


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7})
    ex = D["example"]["median_outcome"]
    y, p, w, cat = np.array(ex["y"]), np.array(ex["p"]), np.array(ex["w"]), np.array(ex["cat"])
    pc, pg = np.array(ex["pred_ce"]), np.array(ex["pred_gtad_cv"])
    t = np.arange(len(y)) * 30 / 3600
    fig, ax = plt.subplots(4, 1, figsize=(3.45, 2.85), sharex=True,
                           gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.42]})

    ax[0].step(t, np.array([4, 2, 1, 0, 3])[y], where="post", color=INK, lw=0.6)
    ax[0].scatter(t[cat == 2], np.full((cat == 2).sum(), 4.75), marker="|", s=6, color=RED, lw=0.5, clip_on=False)
    ax[0].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "R", "W"])
    ax[0].set_ylim(-0.4, 4.4)
    ax[0].set_title("(a) Scorer hypnogram (ticks: stage changes)", loc="left", fontsize=7, pad=5)

    ax[1].stackplot(t, p.T[[3, 2, 1, 4, 0]], colors=[STAGE_COL[i] for i in (3, 2, 1, 4, 0)], step="post", lw=0)
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    ax[1].set_title(r"(b) Teacher soft labels $p_t$ ($T=2$)", loc="left", fontsize=7, pad=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COL[i]) for i in (0, 4, 1, 2, 3)]
    ax[1].legend(handles, ["W", "REM", "N1", "N2", "N3"], ncol=5, frameon=False, fontsize=5.8, loc="lower right",
                 bbox_to_anchor=(1.0, 0.98), handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderpad=0)

    rel = np.where(w > 0, w / w[w > 0].mean(), np.nan)  # NaN: first epoch of a run (no t-1)
    ax[2].fill_between(t, 1, rel, where=np.nan_to_num(rel, nan=1) >= 1, step="post", color=OURS, lw=0)
    ax[2].fill_between(t, 1, rel, where=np.nan_to_num(rel, nan=1) < 1, step="post", color=CE, lw=0)
    ax[2].axhline(1, color=INK, lw=0.5, ls="--")
    ax[2].legend([plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=CE)],
                 ["above night mean", "below night mean"], ncol=2, frameon=False, fontsize=5.8, loc="upper right",
                 handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderpad=0)
    ax[2].set_ylim(0, max(np.nanmax(rel) * 1.3, 1.5))  # headroom for the legend
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_title(r"(c) Transition weighting coefficient, $w_t/\bar{w}$", loc="left", fontsize=7, pad=3)

    dis = pc != pg
    groups = ((dis & (pg == y), GREEN, r"G$_{cv}$-TAD right"), (dis & (pc == y), CE, "CE right"),
              (dis & (pc != y) & (pg != y), RED, "both wrong"))
    for m, col, lab in groups:
        ax[3].vlines(t[m], 0, 1, color=col, lw=0.6, label=f"{lab} ({m.sum()})")
    ax[3].set_yticks([])
    ax[3].set_ylim(0, 1)
    ax[3].set_title("(d) Epochs where the students disagree", loc="left", fontsize=7, pad=3)
    ax[3].legend(frameon=False, fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -1.35),
                 handlelength=0.8, columnspacing=1.0, handletextpad=0.4)
    ax[3].set_xlabel("Hours from recording start", labelpad=1)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_linewidth(0.5)
        a.spines["bottom"].set_linewidth(0.5)
        a.tick_params(colors=MUTED, labelsize=6, width=0.5, length=2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.2, hspace=0.6)
    fig.savefig(HERE / "figs/distill_map.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "figs/distill_map.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    n = {"gright": int(groups[0][0].sum()), "ceright": int(groups[1][0].sum()), "bothwrong": int(groups[2][0].sum()),
         "tacc": round(100 * ex["teacher_acc"]), "changes": round(ex["changes_per_h"], 1)}
    lines = ["% generated by fig_distill_map.py from slides/distillation_map.json",
             f"% example night: {ex['patient']} (iSLEEPS fold 1, unlabelled training patient, median outcome)",
             r"\newcommand{\dmap}[1]{\csname dmap@#1\endcsname}",
             r"\newcommand{\dstat}[2]{\csname dstat@#1@#2\endcsname}"]
    lines += [rf"\expandafter\def\csname dmap@{k}\endcsname{{{v}}}" for k, v in n.items()]
    for c, s in D["stats"].items():
        d = s["disagreement"]
        for k, v in (("gright", d["gtad_cv_right"]), ("ceright", d["ce_right"]), ("nearchange", d["at_or_near_change"])):
            lines.append(rf"\expandafter\def\csname dstat@{c}@{k}\endcsname{{{100 * v:.0f}}}")
        cs = s["categories"]["at_change"]
        for k, v in (("echange", cs["epoch_share"]), ("wchange", cs["weight_share"]),
                     ("enone", s["n1"]["epoch_share"]), ("wnone", s["n1"]["weight_share"])):
            lines.append(rf"\expandafter\def\csname dstat@{c}@{k}\endcsname{{{100 * v:.0f}}}")
    (HERE / "figs/distill_numbers.tex").write_text("\n".join(lines) + "\n")
    print(ex["patient"], n)


if __name__ == "__main__":
    main()
