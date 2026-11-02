"""Per unlabelled iSLEEPS training patient (fold 1): teacher accuracy vs net share of CE/G_cv-TAD disagreements won by
G_cv-TAD (fig_teacher_vs_gain.png; data from distillation_map.json)."""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
HERE = repo_paths.ANALYSIS  # inputs and outputs
d = json.loads((HERE / "distillation_map.json").read_text())
pp, ex = d["stats"]["isleeps"]["per_patient"], d["example"]
s = sorted(pp)
x = np.array([100 * pp[k]["teacher_acc"] for k in s])
yv = np.array([100 * (pp[k]["gtad_cv_right"] - pp[k]["ce_right"]) / max(pp[k]["disagreements"], 1) for k in s])
r, pr = pearsonr(x, yv)
rho, prho = spearmanr(x, yv)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhline(0, color="#5B6380", lw=0.9, ls="--")
ax.scatter(x, yv, s=38, color=np.where(yv > 0, "#0E9384", "#C0392B"), alpha=0.85, edgecolor="white", lw=0.6)
b = np.polyfit(x, yv, 1)
xx = np.linspace(x.min(), x.max(), 50)
ax.plot(xx, np.polyval(b, xx), color="#1B2140", lw=1.4)
for k, lab in ((ex["median_outcome"]["patient"], "median patient"), (ex["worst_outcome"]["patient"], "worst patient")):
    i = s.index(k)
    ax.annotate(f"{k} ({lab})", (x[i], yv[i]), xytext=(8, -12 if yv[i] < 0 else 8), textcoords="offset points",
                fontsize=9, color="#1B2140")
ax.set_ylim(min(yv) - 12, max(yv) + 8)
ax.set_xlabel("Teacher accuracy on the patient's night (%)", fontsize=11, color="#5B6380")
ax.set_ylabel("G$_{cv}$-TAD wins − CE wins\n(% of their disagreements)", fontsize=11, color="#5B6380")
ax.text(0.02, 0.97, f"{(yv > 0).sum()} of {len(s)} patients above 0\nPearson r = {r:.2f} (p = {pr:.3f})\n"
        f"Spearman ρ = {rho:.2f} (p = {prho:.3f})", transform=ax.transAxes, va="top", fontsize=10, color="#1B2140")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors="#5B6380")
fig.tight_layout()
fig.savefig(HERE / "fig_teacher_vs_gain.png", dpi=220, transparent=True)
json.dump({"n": len(s), "above_zero": int((yv > 0).sum()), "pearson_r": float(r), "pearson_p": float(pr),
           "spearman_rho": float(rho), "spearman_p": float(prho)}, open(HERE / "teacher_vs_gain.json", "w"))
print(len(s), (yv > 0).sum(), round(r, 2), round(pr, 4), round(rho, 2), round(prho, 4))
