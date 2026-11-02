"""Clinician-facing outputs: hypnodensity of one test patient and calibration of the students (writes fig_hypnodensity.png,
fig_reliability.png and calibration.json).

All probabilities are the saved test softmax outputs (temperature 1) of the patient-wise 10-fold runs:
25% labels (CE, KD, G_cv-TAD) from <cohort>/label_efficiency_25pct, 100% labels (CE) from the full-label runs.
ECE: 15 equal-width bins of the top-class probability. Transition epochs: scorer stage differs from the previous or next
contiguous epoch. Selective prediction: the least confident epochs are flagged for review at a fixed review rate.
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
HERE = repo_paths.ANALYSIS  # outputs
LOW = {k: repo_paths.RESULTS / k / "label_efficiency_25pct" for k in ("isleeps", "naps", "isruc")}
NPZ = {"isleeps": (repo_paths.PREPROCESSED["isleeps"], "SN"), "naps": (repo_paths.PREPROCESSED["naps"], ""),
       "isruc": (repo_paths.PREPROCESSED["isruc"], "")}
FULL = {"isleeps": [(repo_paths.RESULTS / "isleeps" / "context_students_cv10_s1/{d}", "fold_*/test_predictions.npz",
                     "context_ce_only_prob")],
        "naps": [(repo_paths.RESULTS / "naps" / "naps_cv10_ce", "folds/fold_*/test_predictions.npz", "{d}_ce_only_prob")],
        "isruc": [(repo_paths.RESULTS / "isruc" / "isruc_cv10_ce", "folds/fold_*/test_predictions.npz", "{d}_ce_only_prob")]}
NAMES = {"isleeps": "iSLEEPS (stroke)", "naps": "NAPS (iRBD)", "isruc": "ISRUC-S1 (sleep disorders)"}
INPUTS = {"eeg": "c4", "eeg_eog": "eeg_eog", "eog": "eog"}
MODELS = [("ce", "CE, 25%", "#9AA3B5"), ("kd", "KD, 25%", "#5E7FB8"), ("gtad_cv", "G$_{cv}$-TAD, 25%", "#E8833A"),
          ("ce_full", "CE, 100%", "#2E3550")]
STAGE_COL = ["#E9C46A", "#8ECAE6", "#5E7FB8", "#1B2140", "#C0392B"]
INK, MUTED = "#1B2140", "#5B6380"
BINS = 15
MIN_BIN = 200  # epochs; sparser low-confidence bins are not plotted (still counted in ECE)


def ordered_labels(key):
    d, prefix = NPZ[key]
    y, subj, idx = [], [], []
    for p in sorted(d.glob(f"{prefix}*.npz"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
        with np.load(p) as z:
            y.append(z["y"].astype(int)); subj.append(np.full(len(z["y"]), z["subject"].item()))
            idx.append(z["epoch_index"])
    return np.concatenate(y), np.concatenate(subj), np.concatenate(idx)


def transition_mask(y, subj, idx):
    """True where the scorer stage differs from the previous or next contiguous epoch of the same recording."""
    same = (subj[1:] == subj[:-1]) & (np.diff(idx) == 1)
    change = same & (y[1:] != y[:-1])
    m = np.zeros(len(y), bool)
    m[1:] |= change
    m[:-1] |= change
    return m


def load(key, inp):
    """Pooled test predictions of the 10 folds: dict model -> (prob, y, transition mask)."""
    y, subj, idx = ordered_labels(key)
    out = {k: [] for k, _, _ in MODELS}
    for fold in sorted(LOW[key].glob("fold_*")):
        tm = json.loads((fold / "test_metrics.json").read_text())
        mask = np.isin(subj, tm["test_subjects"])
        yt, tr = y[mask], transition_mask(y[mask], subj[mask], idx[mask])
        for k in ("ce", "kd", "gtad_cv"):
            p = np.load(fold / inp / f"{k}_test_prob.npy")
            assert len(p) == len(yt)
            out[k].append((p, yt, tr))
    src, pattern, pkey = FULL[key][0]
    base = Path(str(src).format(d=INPUTS[inp]))
    for f in sorted(base.glob(pattern)):
        with np.load(f) as z:
            yf, sf, ix = z["y_true"], z["subject"], z["epoch_index"]
            out["ce_full"].append((z[pkey.format(d=INPUTS[inp])], yf, transition_mask(yf, sf, ix)))
    return {k: tuple(np.concatenate([t[i] for t in out[k]]) for i in range(3)) for k in out}


def calibration(p, y):
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    edges = np.linspace(0, 1, BINS + 1)
    b = np.clip(np.digitize(conf, edges) - 1, 0, BINS - 1)
    ece, curve = 0.0, []
    for i in range(BINS):
        m = b == i
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            if m.sum() >= MIN_BIN:  # plotted bins only; ECE uses all bins
                curve.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(ece), curve


def selective(p, y, review):
    """Accuracy and N1 recall on the epochs kept after flagging the `review` fraction with lowest confidence."""
    conf = p.max(1)
    keep = conf >= np.quantile(conf, review) if review > 0 else np.ones(len(y), bool)
    pred = p.argmax(1)
    return {"accuracy": float((pred[keep] == y[keep]).mean()),
            "flagged_error_share": float(((pred != y) & ~keep).sum() / max((pred != y).sum(), 1))}


def main():
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, key in zip(axes, LOW):
        res[key] = {}
        for inp in INPUTS:
            data = load(key, inp)
            res[key][inp] = {}
            for k, label, col in MODELS:
                p, y, tr = data[k]
                ece, curve = calibration(p, y)
                conf, correct = p.max(1), p.argmax(1) == y
                res[key][inp][k] = {
                    "ece": ece, "accuracy": float(correct.mean()), "mean_conf": float(conf.mean()),
                    "conf_transition": float(conf[tr].mean()), "conf_stable": float(conf[~tr].mean()),
                    "acc_transition": float(correct[tr].mean()), "acc_stable": float(correct[~tr].mean()),
                    "transition_share": float(tr.mean()),
                    "review": {f"{int(100 * r)}": selective(p, y, r) for r in (0.0, 0.1, 0.2)}, "curve": curve}
                if inp == "eeg_eog":
                    xs, ys, ns = zip(*curve)
                    ax.plot(xs, ys, "-o", color=col, ms=3.5, lw=1.8, label=f"{label}  (ECE {100 * ece:.1f}%)")
        ax.plot([0, 1], [0, 1], "--", color="#C5CAD9", lw=1.2)
        ax.set_title(NAMES[key], fontsize=13, color=INK, fontweight="bold")
        ax.set_xlabel("Confidence (top-class probability)", fontsize=11, color=MUTED)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left")
        ax.grid(color="#E3E6EF", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=11, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "fig_reliability.png", dpi=220, transparent=True)
    plt.close(fig)

    # Hypnodensity: iSLEEPS fold 1, the same representative patient as the hypnogram slide (median G_cv-TAD - CE gain).
    ex = json.loads((HERE / "deck_data.json").read_text())["prediction_example"]
    key, inp, pat = "isleeps", "eeg_eog", ex["patient"]
    fold = LOW[key] / "fold_01"
    subj_list = json.loads((fold / "test_metrics.json").read_text())["test_subjects"]
    y, subj, idx = ordered_labels(key)
    mask = np.isin(subj, subj_list)
    ys, ss = y[mask], subj[mask]
    sel = ss == pat
    t = np.arange(sel.sum()) * 30 / 3600
    rows = [("ce", "CE student, 25% labels"), ("gtad_cv", "G$_{cv}$-TAD student, 25% labels")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25, 1.25]})
    level = np.array([4, 2, 1, 0, 3])[ys[sel]]
    axes[0].step(t, level, where="post", color=INK, lw=1.1)
    axes[0].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "W"], fontsize=9, color=MUTED)
    axes[0].set_title("Scorer hypnogram", loc="left", fontsize=11, color=INK, fontweight="bold", pad=6)
    trs = transition_mask(ys[sel], ss[sel], idx[mask][sel])
    for ax, (k, title) in zip(axes[1:], rows):
        p = np.load(fold / inp / f"{k}_test_prob.npy")[sel]
        ax.stackplot(t, p.T[[3, 2, 1, 4, 0]], colors=[STAGE_COL[i] for i in (3, 2, 1, 4, 0)], step="post", lw=0)
        conf = p.max(1)
        low = conf < 0.6
        ax.scatter(t[low], np.full(low.sum(), 1.035), s=10, color="#C0392B", marker="|", clip_on=False)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=9, color=MUTED)
        acc = (p.argmax(1) == ys[sel]).mean()
        ax.set_title(f"{title}: stage probabilities   (accuracy {100 * acc:.0f}%, "
                     f"{100 * low.mean():.0f}% of epochs below 0.6 confidence, marked in red)", loc="left", fontsize=11,
                     color=INK, fontweight="bold", pad=12)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[-1].set_xlabel("Hours from recording start", fontsize=10, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COL[i]) for i in (0, 4, 1, 2, 3)]
    fig.legend(handles, ["W", "REM", "N1", "N2", "N3"], loc="upper right", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "fig_hypnodensity.png", dpi=220, transparent=True)
    plt.close(fig)
    res["hypnodensity_example"] = {"cohort": key, "input": inp, "patient": pat,
                                   "transition_share": float(trs.mean())}
    (HERE / "calibration.json").write_text(json.dumps(res, indent=1))
    for key in LOW:
        r = res[key]["eeg_eog"]
        print(key, {k: (round(100 * v["ece"], 1), round(v["conf_transition"], 2), round(v["conf_stable"], 2),
                        round(v["acc_transition"], 2), round(v["acc_stable"], 2), round(v["review"]["0"]["accuracy"], 3),
                        round(v["review"]["10"]["accuracy"], 3), round(v["review"]["20"]["accuracy"], 3),
                        round(v["review"]["20"]["flagged_error_share"], 2)) for k, v in r.items()})


if __name__ == "__main__":
    main()
