"""Generate the paper's LaTeX tables and number macros from the saved results (no numbers typed by hand).

Writes tables/main.tex (25%-label results, all cohorts), tables/isruc.tex (iSLEEPS and ISRUC-1 vs published),
tables/cohorts.tex and numbers.tex (\\res{cohort}{input}{model}{metric} macros used in the text).
Significance: two-sided Wilcoxon signed-rank on per-patient macro F1 (stats.json of each 25% run),
Holm-corrected within each comparison over all cohort x input settings available. Run again after
every new result; cohorts without stats.json yet are shown without markers and flagged in the caption.
"""
import sys
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
RESULTS, ANALYSIS = repo_paths.RESULTS, repo_paths.ANALYSIS
INPUTS = [("eeg", "EEG (C4)", "c4"), ("eeg_eog", "EEG+EOG", "eeg_eog"), ("eog", "EOG (E1)", "eog")]
COHORTS = {  # key: (name, population, 25% run, full-label run)
    "isleeps": ("iSLEEPS", "Stroke", RESULTS / "isleeps/label_efficiency_25pct", RESULTS / "isleeps"),
    "naps": ("NAPS", "iRBD", RESULTS / "naps/label_efficiency_25pct", RESULTS / "naps/naps_cv10_ce"),
    "isruc": ("ISRUC-S1", "Sleep disorders", RESULTS / "isruc/label_efficiency_25pct", RESULTS / "isruc/isruc_cv10_ce"),
}
ALPHA = 0.05


def fold_metrics(y, pred):
    return {"macro_f1": f1_score(y, pred, average="macro", labels=range(5)), "kappa": cohen_kappa_score(y, pred),
            "f1_N1": f1_score(y, pred, average=None, labels=range(5))[1], "accuracy": float((y == pred).mean())}


def full_label(key):
    """Per-fold test metrics of the 100%-label teacher and CE students, and the total number of scored epochs."""
    src, per = COHORTS[key][3], {}
    if key == "isleeps":
        with (src / "sleepgpt4_cv10_s1/per_fold_metrics.csv").open() as stream:
            per["teacher"] = [{k: float(r[k]) for k in ("macro_f1", "kappa", "f1_N1", "accuracy")}
                              for r in csv.DictReader(stream) if r["model"] == "teacher"]
        for inp, _, d in INPUTS:
            per[inp] = []
            for p in sorted((src / "context_students_cv10_s1" / d).glob("fold_*/test_predictions.npz")):
                with np.load(p) as z:
                    per[inp].append(fold_metrics(z["y_true"], z["context_ce_only_prob"].argmax(-1)))
        n_epochs = len(np.load(src / "sleepgpt4_cv10_s1/out_of_fold_predictions.npz")["y_true"])
    else:
        rows = [json.loads(p.read_text()) for p in sorted(src.glob("folds/fold_*/test_metrics.json"))]
        per["teacher"] = [r["models"]["teacher"]["test"] for r in rows]
        for inp, _, d in INPUTS:
            per[inp] = [r["models"][f"{d}_ce_only"]["test"] for r in rows]
        n_epochs = sum(r["models"]["teacher"]["test"]["n_epochs"] for r in rows)
    assert all(len(v) == 10 for v in per.values()), (key, {k: len(v) for k, v in per.items()})
    return per, n_epochs


def holm(pvals):
    order, out, running = np.argsort(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = running
    return out


def main():
    res, tests, notes, cohort_rows = {}, {}, [], []
    for key, (name, population, low, _) in COHORTS.items():
        summary = json.loads((low / "summary.json").read_text())
        full, n_epochs = full_label(key)
        folds = summary["teacher"]["folds"]
        if folds < 10:
            notes.append(f"{name}: {folds} of 10 folds")
        r = res[key] = {}
        for m in ("macro_f1", "kappa", "f1_N1", "accuracy"):
            r[("teacher", "low", m)] = (summary["teacher"][f"{m}_mean"], summary["teacher"][f"{m}_sd"])
            r[("teacher", "full", m)] = (np.mean([f[m] for f in full["teacher"]]), np.std([f[m] for f in full["teacher"]], ddof=1))
            for inp, _, _ in INPUTS:
                for kind in ("ce", "kd", "gtad_cv"):
                    s = summary[f"{inp}_{kind}"]
                    r[(inp, kind, m)] = (s[f"{m}_mean"], s[f"{m}_sd"])
                v = [f[m] for f in full[inp]]
                r[(inp, "ce_full", m)] = (np.mean(v), np.std(v, ddof=1))
        split = json.loads((low / "fold_01/split.json").read_text())
        n = len(split["train"]) + len(split["val"]) + len(split["test"])
        cohort_rows.append((name, population, n, n_epochs, f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
                            len(split["labelled"])))
        stats = low / "stats.json"
        if stats.exists() and json.loads(stats.read_text())["n_folds"] == 10:
            for t in json.loads(stats.read_text())["tests"]:
                tests[(key, t["input"], t["a"], t["b"])] = t
        else:
            notes.append(f"{name}: significance pending")
    # Holm within each comparison family across the available settings.
    for pair in {(k[2], k[3]) for k in tests}:
        keys = [k for k in tests if (k[2], k[3]) == pair]
        for k, p in zip(keys, holm(np.array([tests[k]["p_wilcoxon"] for k in keys]))):
            tests[k]["p_holm"] = float(p)

    def cell(v, bold=False):
        s = f"{v:.2f}"
        return rf"\textbf{{{s}}}" if bold else s

    metrics = ("accuracy", "macro_f1", "kappa")
    classes = ("W", "N1", "N2", "N3", "REM")
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Test accuracy (Acc), macro F1 (MF1) and Cohen's $\kappa$ (mean over "
             r"10 patient-wise folds) of models trained on 100\% of the labels (CE) and with 25\% of the training patients "
             r"labelled (CE or G$_{cv}$-TAD), and per-class F1 with 25\% labels (teacher: CE; students: G$_{cv}$-TAD). "
             r"Teacher: 4-channel SleepGPT-BiLSTM; students: 30k-parameter causal CNN-GRU. Best G$_{cv}$-TAD student of each cohort in bold."
             + (" \\textbf{TODO: " + "; ".join(notes) + ".}" if notes else "") + "}",
             r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ll|ccc|ccc|ccc|ccccc}", r"\hline",
             r" & & \multicolumn{3}{c|}{100\% labels} & \multicolumn{6}{c|}{25\% labelled patients} & "
             r"\multicolumn{5}{c}{Per-class F1, 25\%} \\",
             r" & & \multicolumn{3}{c|}{CE} & \multicolumn{3}{c|}{CE} & \multicolumn{3}{c|}{G$_{cv}$-TAD} & & & & & \\",
             r"Cohort & Model" + r" & Acc & MF1 & $\kappa$" * 3 + r" & W & N1 & N2 & N3 & REM \\", r"\hline"]
    for key, (name, _p, low, _f) in COHORTS.items():
        r = res[key]
        summary = json.loads((low / "summary.json").read_text())
        lines.append(rf"\multirow{{4}}{{*}}{{{name}}} & Teacher & "
                     + " & ".join(cell(r[("teacher", "full", m)][0]) for m in metrics) + " & "
                     + " & ".join(cell(r[("teacher", "low", m)][0]) for m in metrics) + " & " + " & ".join(["--"] * 3) + " & "
                     + " & ".join(cell(summary["teacher"][f"f1_{c}_mean"]) for c in classes) + r" \\")
        # Bold: best G_cv-TAD student of the cohort (over the three inputs), per metric and per class; ties as printed.
        g = {inp: {**{m: round(r[(inp, "gtad_cv", m)][0], 2) for m in metrics},
                   **{c: round(summary[f"{inp}_gtad_cv"][f"f1_{c}_mean"], 2) for c in classes}} for inp, _, _ in INPUTS}
        top = {k: max(v[k] for v in g.values()) for k in (*metrics, *classes)}
        for inp, label, _ in sorted(INPUTS, key=lambda i: ("eog", "eeg", "eeg_eog").index(i[0])):
            row = [cell(r[(inp, "ce_full", m)][0]) for m in metrics]
            row += [cell(r[(inp, "ce", m)][0]) for m in metrics]
            row += [cell(g[inp][k], g[inp][k] == top[k]) for k in (*metrics, *classes)]
            lines.append(f" & {label} & " + " & ".join(row) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables/main.tex").write_text("\n".join(lines) + "\n")

    # Cohorts, with sleep fragmentation from fragmentation.py (Sleep-EDF-20: healthy reference, not used for training).
    frag = json.loads((ANALYSIS / "fragmentation.json").read_text())
    fm = ("awakenings_per_h", "shifts_per_h", "n1_pct")
    fkeys = [(k, m) for k in COHORTS for m in fm]
    fsig = dict(zip(fkeys, holm(np.array([frag[k][m]["p_vs_healthy"] for k, m in fkeys])) < ALPHA))
    short = {"Sleep disorders": "Sleep dis."}
    c = [r"\begin{table}[t]", r"\centering", r"\caption{Cohorts and sleep fragmentation (mean over recordings, within the sleep "
         r"period). Lab.: training patients with scorer labels in the 25\% setting; Awak.: awakenings per hour; Shifts: stage "
         r"changes per hour; N1: \% of sleep. iRBD: isolated REM sleep behaviour disorder. Sleep-EDF-20 (healthy, 25--34~y, "
         r"R\&K scoring; " + str(frag["sleepedf20"]["n"]) + r" nights) is shown for reference only. $^{*}$ differs from "
         r"Sleep-EDF-20 (Mann--Whitney U, Holm-corrected, $p<0.05$).}", r"\label{tab:cohorts}", r"\small", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{llrrrrrr}", r"\hline", r"Cohort & Group & N & Epochs & Lab. & Awak. & Shifts & N1 \\", r"\hline"]
    for (a, b, n, e, _s, lab), key in zip(cohort_rows, COHORTS):
        f = [f"{frag[key][m]['mean']:.1f}" + ("$^{*}$" if fsig[(key, m)] else "") for m in fm]
        c.append(f"{a} & {short.get(b, b)} & {n} & {e:,} & {lab} & {f[0]} & {f[1]} & {f[2]} \\\\".replace(",", "{,}"))
    h = frag["sleepedf20"]
    c.append(r"\hline")
    c.append(f"Sleep-EDF-20 & Healthy & 20 & {h['n_epochs']:,} & -- & {h[fm[0]]['mean']:.1f} & {h[fm[1]]['mean']:.1f} & "
             f"{h[fm[2]]['mean']:.1f} \\\\".replace(",", "{,}"))
    c += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "tables/cohorts.tex").write_text("\n".join(c) + "\n")

    # iSLEEPS and ISRUC-1 vs published: literature values from literature_isleeps.json and literature_isruc.json
    # (copied from the cited papers, with sources); ours from the 10-fold results (res; ISRUC 100% from extra_metrics.json).
    ex = json.loads((RESULTS / "isruc/isruc_cv10_ce/extra_metrics.json").read_text())
    assert ex["n_folds"] == 10 and len(ex["subjects_91_100_present"]) == 10
    lit_isruc = json.loads((OUT / "literature_isruc.json").read_text())
    lit_isleeps = json.loads((OUT / "literature_isleeps.json").read_text())
    lit_sleepedf = json.loads((OUT / "literature_sleepedf.json").read_text())
    # Sleep-EDF-20 LOSO (20 folds, Fpz-Cz + EOG): pooled over all test epochs, as the cited works report.
    sedf = json.loads((RESULTS / "sleepedf20/sleepedf20_loso/summary.json").read_text())
    assert all(sedf[k]["folds"] == 20 for k in sedf)
    sedf_map = {"teacher_full": ("teacher", "full"), "teacher_25": ("teacher", "low"),
                "student_ce_full": ("eeg_eog", "ce_full"), "student_gtad_cv_25": ("eeg_eog", "gtad_cv")}
    res["sleepedf"] = {(*sedf_map[k], m): (v[f"{m}_mean"], v[f"{m}_sd"]) for k, v in sedf.items()
                       for m in ("macro_f1", "kappa", "f1_N1", "accuracy")}
    eff = {r["model"]: r for r in json.loads((ANALYSIS / "efficiency.json").read_text())["rows"]}
    pstr = lambda n: "--" if n is None else (f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k")  # noqa: E731
    marks = {"SE-ResNet-BiLSTM": "a", "MM-Net": "b", "MSSC-BiMamba": "c", "SleepSMC (all 9 ch.)": "d",
             "SleepSMC (6-ch. EEG at test)": "d"}
    names = {"SleepSMC (all 9 ch.)": "SleepSMC", "SleepSMC (6-ch. EEG at test)": "SleepSMC, EEG"}
    isruc_ch = {"MSSC-BiMamba": "10", "SleepSMC (all 9 ch.)": "9", "SleepSMC (6-ch. EEG at test)": "6"}

    def q2(v):  # rounded half up to 2 decimals
        return str(Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP))

    def lit_row(m, ch):
        mk = f"$^{{{marks[m['name']]}}}$" if m["name"] in marks else ""
        return [f"{names.get(m['name'], m['name'])}{mk} \\cite{{{m['cite']}}}", ch, pstr(m["params"]),
                q2(m["acc"] / 100), q2(m["mf1"] / 100), "--" if m["kappa"] is None else q2(m["kappa"])]

    def our(label, ch, params, a, f1, kap):
        return [label, ch, pstr(params), f"{a:.2f}", f"{f1:.2f}", f"{kap:.2f}"]

    def ours(key, full, teacher_ch="4"):
        r = res[key]
        return [our("Ours: teacher", teacher_ch, eff["teacher"]["params"], *full["teacher"]),
                our(r"Ours: student, \gtad, 25\%", "2", eff["student_2ch"]["params"],
                    *(full["student"] if "student" in full else
                      (r[("eeg_eog", "gtad_cv", m)][0] for m in ("accuracy", "macro_f1", "kappa"))))]

    def bold(rows):  # best value of each metric column within a block (ties as printed)
        for col in (3, 4, 5):
            top = max(float(c[col]) for c in rows if c[col] != "--")
            for c in rows:
                if c[col] != "--" and float(c[col]) == top:
                    c[col] = rf"\textbf{{{c[col]}}}"
        return [" & ".join(c) + r" \\" for c in rows]

    ri = res["isleeps"]
    full_isleeps = {k: [ri[(k, "full" if k == "teacher" else "ce_full", m)][0] for m in ("accuracy", "macro_f1", "kappa")]
                    for k in ("teacher", "eeg_eog")}
    fm = ex["fold_mean"]
    full_isruc = {"teacher": [fm["teacher"][m] for m in ("acc", "macro_f1", "kappa")],
                  "eeg_eog": [fm["eeg_eog_ce_only"][m] for m in ("acc", "macro_f1", "kappa")]}
    t = [r"\begin{table}[t]", r"\centering", r"\caption{Healthy (Sleep-EDF-20) and patient cohorts against published results (100\% labels "
         r"unless marked 25\%; Ch.: input channels). Ours: Sleep-EDF-20 leave-one-subject-out, pooled over test "
         r"epochs as in the cited works; patient cohorts patient-wise 10-fold CV, fold means. Published protocols "
         r"(folds, recordings) differ from ours. Best per cohort in bold.}",
         r"\label{tab:isruc}", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{lcrccc}", r"\hline",
         r"Method & Ch. & Params & Acc & MF1 & $\kappa$ \\", r"\hline", r"\multicolumn{6}{l}{\textit{Sleep-EDF-20}} \\"]
    pooled = lambda k: [sedf[k]["pooled"][m] for m in ("accuracy", "macro_f1", "kappa")]  # noqa: E731
    t += bold([lit_row(m, m["channels"]) for m in lit_sleepedf["rows"]]
              + ours("sleepedf", {"teacher": pooled("teacher_full"), "student": pooled("student_gtad_cv_25")}, "2"))
    t += [r"\hline", r"\multicolumn{6}{l}{\textit{iSLEEPS}} \\"]
    t += bold([lit_row(m, m["channels"]) for m in lit_isleeps["rows"]] + ours("isleeps", full_isleeps))
    t += [r"\hline", r"\multicolumn{6}{l}{\textit{ISRUC-S1}} \\"]
    t += bold([lit_row(m, isruc_ch.get(m["name"], "--")) for m in lit_isruc["multi_channel"] if m["cite"]]
              + ours("isruc", full_isruc))
    t += [r"\hline", r"\multicolumn{6}{l}{\parbox{0.95\columnwidth}{\scriptsize $^{a}$C4; patient-wise split not stated. "
          r"$^{b}$7 EEG/EOG/EMG + 8 cardiorespiratory channels; plus a 5.8M pretrained encoder. $^{c}$50 subjects, "
          r"25-fold. $^{d}$5-fold; trained with EEG, EOG and EMG.}}", r"\end{tabular}", r"\end{table}"]
    (OUT / "tables/isruc.tex").write_text("\n".join(t) + "\n")

    # Number macros for the text: \res{isleeps}{eeg_eog}{gtad_cv}{macro_f1} (fold mean) and
    # \gain{naps}{eeg}{gtad_cv}{kd} (mean per-patient macro F1 difference, points).
    # Undefined values (e.g. significance not computed yet) print a bold ?? instead of nothing.
    m = ["% generated by make_tables.py",
         r"\newcommand{\res}[4]{\ifcsname res@#1@#2@#3@#4\endcsname\csname res@#1@#2@#3@#4\endcsname\else\textbf{??}\fi}",
         r"\newcommand{\gain}[4]{\ifcsname dlt@#1@#2@#3@#4\endcsname\csname dlt@#1@#2@#3@#4\endcsname\else\textbf{??}\fi}"]
    for key, r in res.items():
        for (inp, kind, metric), (mean, _) in r.items():
            m.append(rf"\expandafter\def\csname res@{key}@{inp}@{kind}@{metric}\endcsname{{{mean:.2f}}}")
    for (key, inp, a, b), t_ in tests.items():
        d = f"{100 * t_['mean_diff']:+.1f}".replace("-", "$-$")
        m.append(rf"\expandafter\def\csname dlt@{key}@{inp}@{a}@{b}\endcsname{{{d}}}")
    m.append(r"\newcommand{\frag}[2]{\ifcsname frag@#1@#2\endcsname\csname frag@#1@#2\endcsname\else\textbf{??}\fi}")
    for key, v in frag.items():
        for metric in ("awakenings_per_h", "shifts_per_h", "n1_pct", "sleep_eff"):
            m.append(rf"\expandafter\def\csname frag@{key}@{metric}\endcsname{{{v[metric]['mean']:.1f}}}")
    eff = json.loads((ANALYSIS / "efficiency.json").read_text())["rows"]
    for r_ in eff:
        for metric, val in (("params", f"{r_['params']:,}".replace(",", "{,}")), ("fp32_kb", f"{r_['fp32_bytes'] / 1e3:.1f}"),
                            ("int8_kb", f"{r_['int8_bytes'] / 1e3:.0f}"), ("mmacs", f"{r_['macs'] / 1e6:.1f}"),
                            ("ms", f"{r_['latency_ms']:.2f}"), ("gmacs", f"{r_['macs'] / 1e9:.1f}"), ("ratio", f"{eff[0]['params'] / r_['params']:,.0f}".replace(",", "{,}"))):
            m.append(rf"\expandafter\def\csname eff@{r_['model']}@{metric}\endcsname{{{val}}}")
    m.append(r"\newcommand{\eff}[2]{\ifcsname eff@#1@#2\endcsname\csname eff@#1@#2\endcsname\else\textbf{??}\fi}")
    (OUT / "numbers.tex").write_text("\n".join(m) + "\n")
    print("wrote tables/main.tex, tables/cohorts.tex, tables/isruc.tex, numbers.tex")
    for line in notes:
        print("NOTE:", line)


if __name__ == "__main__":
    main()
