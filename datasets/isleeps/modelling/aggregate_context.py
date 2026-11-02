"""Aggregate the 10-fold context-student results (and the teacher) into tables.

For each input (eog, c4, eeg_eog) and variant (ce_only, no_kd): fold mean +/- sample SD of test
metrics, pooled out-of-fold metrics over all test epochs, per-patient mean, and the inner-validation
macro F1 of the selected checkpoints. The recommended variant per input is the one with the highest
mean inner-validation macro F1 (test scores never choose). Writes RESULTS.md, summary.csv and
summary.json into the results folder.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from protocol_isleeps import metrics, write_json  # noqa: E402

KEYS = ["accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
INPUT_NAMES = {"eog": "EOG only (E1-M2)", "c4": "EEG only (C4)", "eeg_eog": "EEG + EOG (C4 + E1-M2)"}
VARIANT_NAMES = {"ce_only": "CE only", "no_kd": "TAD only (No KD)", "full_tad": "Full TAD"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repo_paths.RESULTS / "isleeps" / "context_students_cv10")
    parser.add_argument("--teacher", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10")
    parser.add_argument("--n-folds", type=int, default=10)
    args = parser.parse_args()
    rows, table = [], {}
    for inp in INPUT_NAMES:
        for kind in VARIANT_NAMES:
            folds, val, ys, ps, subj = [], [], [], [], []
            for f in range(1, args.n_folds + 1):
                tm = args.results / inp / f"fold_{f:02d}" / "test_metrics.json"
                done = args.results / inp / "checkpoints" / f"fold_{f:02d}" / f"{kind}_complete.json"
                if not tm.exists() or not done.exists():
                    continue
                m = json.loads(tm.read_text())["models"].get(f"context_{kind}")
                if m is None:
                    continue
                folds.append([m["test"][k] for k in KEYS])
                val.append(json.loads(done.read_text())["validation"]["macro_f1"])
                with np.load(args.results / inp / f"fold_{f:02d}" / "test_predictions.npz") as d:
                    ys.append(d["y_true"]); ps.append(d[f"context_{kind}_pred"]); subj.append(d["subject"])
            if not folds:
                continue
            folds = np.array(folds)
            y, p, s = np.concatenate(ys), np.concatenate(ps), np.concatenate(subj)
            pooled = metrics(y, p)
            per_patient = np.array([[metrics(y[s == u], p[s == u])[k] for k in KEYS] for u in np.unique(s)])
            entry = {"input": inp, "variant": kind, "folds": len(folds), "patients": int(len(np.unique(s))),
                     "val_macro_f1_mean": float(np.mean(val)),
                     **{f"{k}_fold_mean": float(folds[:, i].mean()) for i, k in enumerate(KEYS)},
                     **{f"{k}_fold_sd": float(folds[:, i].std(ddof=1)) if len(folds) > 1 else 0.0 for i, k in enumerate(KEYS)},
                     **{f"{k}_patient_mean": float(per_patient[:, i].mean()) for i, k in enumerate(KEYS)},
                     **{f"{k}_pooled": float(pooled[k]) for k in KEYS}}
            rows.append(entry)
            table[(inp, kind)] = entry
    teacher = []
    for f in range(1, args.n_folds + 1):
        tm = args.teacher / "folds" / f"fold_{f:02d}" / "test_metrics.json"
        if tm.exists():
            teacher.append([json.loads(tm.read_text())["models"]["teacher"]["test"][k] for k in KEYS])
    teacher = np.array(teacher)
    lines = [f"# Context-aware students distilled from the 4-channel SleepGPT teacher", "",
             f"Folds with results: teacher {len(teacher)}; students per row below. Checkpoints selected on "
             "inner-validation macro F1; the recommended variant per input is chosen on mean inner-validation "
             "macro F1, never on test.", "",
             "## Test, fold mean ± SD", "",
             "| Input | Student | Folds | Val F1 | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if len(teacher):
        mt, st = teacher.mean(0), teacher.std(0, ddof=1) if len(teacher) > 1 else np.zeros(len(KEYS))
        lines.append("| 4-channel teacher | SleepGPT-BiLSTM | %d | – | " % len(teacher) +
                     " | ".join(f"{mt[i]:.3f} ± {st[i]:.3f}" if i < 3 else f"{mt[i]:.3f}" for i in range(len(KEYS))) + " |")
    best = {}
    for inp in INPUT_NAMES:
        cands = [table[(inp, k)] for k in VARIANT_NAMES if (inp, k) in table]
        if cands:
            best[inp] = max(cands, key=lambda e: e["val_macro_f1_mean"])["variant"]
        for e in cands:
            mark = " **(chosen on val)**" if best.get(inp) == e["variant"] else ""
            lines.append(f"| {INPUT_NAMES[inp]} | {VARIANT_NAMES[e['variant']]}{mark} | {e['folds']} | {e['val_macro_f1_mean']:.3f} | " +
                         " | ".join(f"{e[k + '_fold_mean']:.3f} ± {e[k + '_fold_sd']:.3f}" if k in KEYS[:3] else f"{e[k + '_fold_mean']:.3f}"
                                    for k in KEYS) + " |")
    lines += ["", "## Pooled out-of-fold (all test epochs)", "",
              "| Input | Student | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in rows:
        lines.append(f"| {INPUT_NAMES[e['input']]} | {VARIANT_NAMES[e['variant']]} | " + " | ".join(f"{e[k + '_pooled']:.3f}" for k in KEYS) + " |")
    out = args.results
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(out / "summary.json", {"best_variant_by_validation": best, "rows": rows})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
