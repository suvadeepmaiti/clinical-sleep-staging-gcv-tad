"""Numbers and figures for the presentation deck, all read from the saved results (no numbers typed by hand).

Writes slides/deck_data.json and slides/fig_*.png. Run after make_tables.py; build_deck.py reads both.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
sys.path.insert(0, str(repo_paths.REPO / "paper"))
from make_tables import COHORTS, INPUTS, full_label, holm  # noqa: E402

NPZ = {"isleeps": (repo_paths.PREPROCESSED["isleeps"], "SN"), "naps": (repo_paths.PREPROCESSED["naps"], ""),
       "isruc": (repo_paths.PREPROCESSED["isruc"], "")}
OUT = repo_paths.ANALYSIS
STAGES = ["W", "N1", "N2", "N3", "REM"]
KINDS = ("ce", "kd", "gtad_cv")
INK, MUTED = "#1B2140", "#5B6380"
COL = {"truth": "#1B2140", "ce": "#9AA3B5", "kd": "#5E7FB8", "gtad_cv": "#E8833A"}


def labels(key, subjects):
    d, _ = NPZ[key]
    order = sorted(subjects, key=lambda s: int("".join(filter(str.isdigit, s)) or 0))
    ys = {s: np.load(d / f"{s}.npz")["y"].astype(int) for s in order}
    return order, ys


def fold_detail(y, pred):
    c = confusion_matrix(y, pred, labels=range(5))
    return {"accuracy": float((y == pred).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=range(5))),
            "kappa": float(cohen_kappa_score(y, pred)), "stage_f1": f1_score(y, pred, average=None, labels=range(5)).tolist(),
            "cm": c}


def full_label_detail(key):
    """100%-label teacher and CE students: fold-mean metrics, per-stage F1 and test confusion summed over the 10 folds."""
    src, per = COHORTS[key][3], {}
    if key == "isleeps":
        for p in sorted((src / "sleepgpt4_cv10_s1/folds").glob("fold_*/test_predictions.npz")):
            with np.load(p) as z:
                per.setdefault("teacher", []).append(fold_detail(z["y_true"], z["teacher_pred"]))
        for inp, _, d in INPUTS:
            for p in sorted((src / "context_students_cv10_s1" / d).glob("fold_*/test_predictions.npz")):
                with np.load(p) as z:
                    per.setdefault(inp, []).append(fold_detail(z["y_true"], z["context_ce_only_prob"].argmax(-1)))
    else:
        for p in sorted(src.glob("folds/fold_*/test_metrics.json")):
            m = json.loads(p.read_text())["models"]
            for name, k in [("teacher", "teacher")] + [(inp, f"{d}_ce_only") for inp, _, d in INPUTS]:
                t = m[k]["test"]
                per.setdefault(name, []).append({"accuracy": t["accuracy"], "macro_f1": t["macro_f1"], "kappa": t["kappa"],
                                                 "stage_f1": [t[f"f1_{s}"] for s in STAGES],
                                                 "cm": np.array(t["confusion_matrix"])})
    out = {}
    for name, folds in per.items():
        assert len(folds) == 10, (key, name, len(folds))
        out[name] = {m: float(np.mean([f[m] for f in folds])) for m in ("accuracy", "macro_f1", "kappa")}
        out[name]["stage_f1"] = np.mean([f["stage_f1"] for f in folds], 0).tolist()
        out[name]["confusion"] = sum(f["cm"] for f in folds).tolist()
    return out


def results():
    out, tests = {}, {}
    for key, (name, group, low, _) in COHORTS.items():
        summary = json.loads((low / "summary.json").read_text())
        full, _ = full_label(key)
        r = {"name": name, "group": group, "teacher_low": summary["teacher"]["macro_f1_mean"],
             "teacher_full": float(np.mean([f["macro_f1"] for f in full["teacher"]])), "inputs": {},
             "teacher_low_detail": {**{m: summary["teacher"][f"{m}_mean"] for m in ("accuracy", "macro_f1", "kappa")},
                                    "stage_f1": [summary["teacher"][f"f1_{st}_mean"] for st in STAGES]}}
        for inp, label, _ in INPUTS:
            e = r["inputs"][inp] = {"label": label}
            for k in KINDS:
                s = summary[f"{inp}_{k}"]
                e[k] = {m: s[f"{m}_mean"] for m in ("macro_f1", "kappa", "accuracy")}
                e[k]["stage_f1"] = [s[f"f1_{st}_mean"] for st in STAGES]
            e["ce_full"] = {m: float(np.mean([f[m] for f in full[inp]])) for m in ("macro_f1", "kappa", "accuracy", "f1_N1")}
        for t in json.loads((low / "stats.json").read_text())["tests"]:
            tests[(key, t["input"], t["a"], t["b"])] = t
        # Confusion matrices summed over the 10 test folds.
        cm = {}
        for fold in sorted(low.glob("fold_*")):
            tm = json.loads((fold / "test_metrics.json").read_text())["models"]
            for inp, _, _ in INPUTS:
                for k in KINDS:
                    c = np.array(tm[f"{inp}_{k}"]["test"]["confusion_matrix"])
                    cm[f"{inp}_{k}"] = cm.get(f"{inp}_{k}", 0) + c
            c = np.array(tm["teacher"]["test"]["confusion_matrix"])
            cm["teacher"] = cm.get("teacher", 0) + c
        r["confusion"] = {k: v.tolist() for k, v in cm.items()}
        r["full_detail"] = full_label_detail(key)
        for inp, _, _ in INPUTS:  # same numbers as make_tables.full_label
            assert abs(r["full_detail"][inp]["macro_f1"] - r["inputs"][inp]["ce_full"]["macro_f1"]) < 1e-6, (key, inp)
        assert abs(r["full_detail"]["teacher"]["macro_f1"] - r["teacher_full"]) < 1e-6, key
        pri = json.loads((low / "fold_01/transition_priors.json").read_text())
        r["prior_fold01"] = {k: pri[k] for k in ("M_mean", "M_std", "gtad", "gtad_cv", "patients")}
        out[key] = r
    for pair in {(k[2], k[3]) for k in tests}:
        keys = [k for k in tests if (k[2], k[3]) == pair]
        for k, p in zip(keys, holm(np.array([tests[k]["p_wilcoxon"] for k in keys]))):
            tests[k]["p_holm"] = float(p)
    for (key, inp, a, b), t in tests.items():
        out[key]["inputs"][inp].setdefault("tests", {})[f"{a}_vs_{b}"] = {
            k: t[k] for k in ("n_patients", "mean_diff", "median_diff", "a_better", "p_wilcoxon", "p_holm")}
    return out


def plot_hypnogram(ax, y, color, label, hours=True):
    level = np.array([4, 2, 1, 0, 3])[y]  # W top, REM, N1, N2, N3 bottom
    t = np.arange(len(y)) * 30 / 3600
    ax.step(t, level, where="post", color=color, lw=1.1)
    rem = np.ma.masked_where(y != 4, level)
    ax.step(t, rem, where="post", color="#C0392B", lw=2.6)
    ax.set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "W"], fontsize=9, color=MUTED)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, t[-1])
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C5CAD9")
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)


def awakenings_per_h(y):
    s = np.flatnonzero(y != 0)
    y = y[s[0]:s[-1] + 1]
    return ((y[:-1] != 0) & (y[1:] == 0)).sum() / (len(y) * 30 / 3600)


def fig_healthy_vs_patient(frag):
    """One typical night per group: the recording whose awakenings/h is closest to its cohort mean."""
    picks = [("sleepedf20", repo_paths.PREPROCESSED["sleepedf20"], sorted((repo_paths.PREPROCESSED["sleepedf20"]).glob("SC*.npz")),
              "Healthy adult (Sleep-EDF-20)"), ("isruc", None, None, "Sleep-disorder patient (ISRUC-S1)")]
    split = json.loads((repo_paths.RESULTS / "isruc" / "label_efficiency_25pct/fold_01/split.json").read_text())
    picks[1] = ("isruc", None, [repo_paths.PREPROCESSED["isruc"] / f"{s}.npz" for s in split["train"] + split["val"] + split["test"]],
                picks[1][3])
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=False)
    chosen = {}
    for ax, (key, _, paths, title) in zip(axes, picks):
        target = frag[key]["awakenings_per_h"]["mean"]
        ys = {p.stem: np.load(p)["y"].astype(int) for p in paths}
        best = min(ys, key=lambda s: abs(awakenings_per_h(ys[s]) - target))
        y = ys[best]
        s = np.flatnonzero(y != 0)
        y = y[max(s[0] - 20, 0):s[-1] + 20]
        chosen[key] = {"recording": best, "awakenings_per_h": float(awakenings_per_h(ys[best]))}
        if key == "isruc":  # decorative hypnogram on the title slide
            (OUT / "title_hypnogram.json").write_text(json.dumps(y.tolist()))
        plot_hypnogram(ax, y, INK, f"{title}: {chosen[key]['awakenings_per_h']:.1f} awakenings/h "
                                   f"(cohort mean {target:.1f})")
    axes[1].set_xlabel("Hours from sleep onset", fontsize=9, color=MUTED)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(OUT / "fig_hypnograms.png", dpi=220, transparent=True)
    plt.close(fig)
    return chosen


def fig_predictions():
    """iSLEEPS fold 1, EEG+EOG students: the test patient whose G_cv-TAD-minus-CE macro F1 gain is the fold median."""
    key, inp = "isleeps", "eeg_eog"
    low = COHORTS[key][2] / "fold_01"
    subj = json.loads((low / "test_metrics.json").read_text())["test_subjects"]
    order, ys = labels(key, subj)
    pred = {k: np.load(low / inp / f"{k}_test_prob.npy").argmax(-1) for k in KINDS}
    bounds = np.cumsum([0] + [len(ys[s]) for s in order])
    assert bounds[-1] == len(pred["ce"])
    f1 = lambda y, p: f1_score(y, p, average="macro", labels=np.unique(y))  # noqa: E731
    gains = {s: f1(ys[s], pred["gtad_cv"][a:b]) - f1(ys[s], pred["ce"][a:b]) for s, a, b in zip(order, bounds, bounds[1:])}
    med = sorted(gains, key=gains.get)[len(gains) // 2]
    i = order.index(med)
    a, b = bounds[i], bounds[i + 1]
    y = ys[med]
    fig, axes = plt.subplots(4, 1, figsize=(11, 5.6), sharex=True)
    info = {"patient": med, "n_test_patients": len(order)}
    rows = [("truth", y, "Scorer (AASM)")] + [(k, pred[k][a:b], n) for k, n in
                                               (("ce", "CE student, 25% labels"), ("kd", "KD student"),
                                                ("gtad_cv", "G$_{cv}$-TAD student"))]
    for ax, (k, p, n) in zip(axes, rows):
        if k != "truth":
            info[k] = {"macro_f1": float(f1(y, p)), "n1_f1": float(f1_score(y, p, labels=[1], average="macro")),
                       "accuracy": float((y == p).mean())}
            n = f"{n}   macro F1 {info[k]['macro_f1']:.2f}, N1 F1 {info[k]['n1_f1']:.2f}"
        plot_hypnogram(ax, p, COL[k] if k != "ce" else "#6F788F", n)
    axes[-1].set_xlabel("Hours from recording start", fontsize=9, color=MUTED)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(OUT / "fig_predictions.png", dpi=220, transparent=True)
    plt.close(fig)
    return info


EQUATIONS = {
    "eq_kd": r"$\mathcal{L}_{KD}=0.3\,\mathcal{L}_{CE}+0.5\,T^{2}\,\mathrm{KL}(p_t\,\|\,q_t)$",
    "eq_tad": r"$\mathcal{L}_{TAD}=\frac{T^{2}}{L-1}\sum_{t=2}^{L}\,c_{t-1}\,\mathrm{KL}(p_t\,\|\,q_t),"
              r"\quad c_{t-1}=\max_k\,p_{t-1,k}$",
    "eq_lg": r"$\mathcal{L}_{G}=\frac{T^{2}}{L-1}\sum_{t=2}^{L}c_{t-1}\sum_{k}G[s,k]\left(p_{t,k}\log\frac{p_{t,k}}{q_{t,k}}"
             r"-p_{t,k}+q_{t,k}\right)$",
    "eq_gcv": r"$G_{cv}=\frac{1-\bar{M}}{1+\lambda\,\mathrm{CV}},\qquad \mathrm{CV}=\frac{\sigma}{\bar{M}+\epsilon},"
              r"\qquad \lambda=0.5$",
    "eq_total": r"$\mathcal{L}=0.3\,\mathcal{L}_{CE}+0.2\,\mathcal{L}_{G_{cv}}$",
}


def fig_equations():
    plt.rcParams["mathtext.fontset"] = "cm"
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=22, color=INK)
        fig.savefig(OUT / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def main():
    paper = repo_paths.REPO / "paper"
    frag = json.loads((repo_paths.ANALYSIS / "fragmentation.json").read_text())
    eff = json.loads((repo_paths.ANALYSIS / "efficiency.json").read_text())
    isruc = json.loads((repo_paths.RESULTS / "isruc" / "isruc_cv10_ce/extra_metrics.json").read_text())
    data = {"results": results(), "fragmentation": frag, "efficiency": eff,
            "isruc_benchmark": {"fold_mean": isruc["fold_mean"], "subjects_91_100": isruc["subjects_91_100"]}}
    data["literature"] = json.loads((paper / "literature_isruc.json").read_text())
    data["hypnograms"] = fig_healthy_vs_patient(frag)
    data["prediction_example"] = fig_predictions()
    fig_equations()
    (OUT / "deck_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({"hypnograms": data["hypnograms"], "prediction_example": data["prediction_example"]}, indent=1))


if __name__ == "__main__":
    main()
