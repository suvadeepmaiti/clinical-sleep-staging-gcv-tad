"""Paired per-patient tests for the label-efficiency runs (test macro F1 per patient from saved test probabilities)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

INPUTS, KINDS = ("eeg", "eeg_eog", "eog"), ("ce", "kd", "gtad_cv")
PAIRS = [("gtad_cv", "ce"), ("gtad_cv", "kd"), ("kd", "ce"), ("gtad_cv", "ce_full"), ("kd", "ce_full")]
# Full-label CE student test predictions (100% labels, same folds): file pattern and probability key per input.
FULL = {"isleeps": {i: (f"{d}/fold_*/test_predictions.npz", "context_ce_only_prob")
                    for i, d in [("eeg", "c4"), ("eeg_eog", "eeg_eog"), ("eog", "eog")]},
        "naps": {i: ("folds/fold_*/test_predictions.npz", f"{k}_ce_only_prob")
                 for i, k in [("eeg", "c4"), ("eeg_eog", "eeg_eog"), ("eog", "eog")]}}
FULL["isruc"] = FULL["naps"]


def per_patient(y, pred, subj):
    return {str(s): f1_score(y[subj == s], pred[subj == s], average="macro", labels=np.unique(y[subj == s]))
            for s in np.unique(subj)}


def labels(npz_dir, prefix):
    y, subj = [], []
    for p in sorted(Path(npz_dir).glob(f"{prefix}*.npz"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
        with np.load(p, allow_pickle=False) as d:
            y.append(d["y"].astype(np.int64)); subj.append(np.full(len(d["y"]), d["subject"].item()))
    return np.concatenate(y), np.concatenate(subj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--prefix", default="SN")
    ap.add_argument("--full", type=Path, required=True, help="full-label CE results folder")
    ap.add_argument("--layout", choices=list(FULL), required=True)
    a = ap.parse_args()
    y, subj = labels(a.npz, a.prefix)
    per = {f"{i}_{k}": {} for i in INPUTS for k in KINDS + ("ce_full",)}
    for i, (pattern, key) in FULL[a.layout].items():
        for path in sorted(a.full.glob(pattern)):
            with np.load(path, allow_pickle=False) as z:
                per[f"{i}_ce_full"].update(per_patient(z["y_true"], z[key].argmax(-1), z["subject"]))
    for fold in sorted(a.results.glob("fold_*")):
        if not (fold / "test_metrics.json").exists():
            continue
        tm = json.loads((fold / "test_metrics.json").read_text())
        mask = np.isin(subj, tm["test_subjects"])
        yt, st = y[mask], subj[mask]
        for i in INPUTS:
            for k in KINDS:
                pred = np.load(fold / i / f"{k}_test_prob.npy").argmax(-1)
                assert len(pred) == len(yt), (fold, i, k)
                stored = tm["models"][f"{i}_{k}"]["test"]["macro_f1"]
                assert abs(f1_score(yt, pred, average="macro", labels=range(5)) - stored) < 1e-6, (fold, i, k)
                per[f"{i}_{k}"].update(per_patient(yt, pred, st))
    out = {"n_folds": len(list(a.results.glob("fold_*/test_metrics.json"))), "tests": []}
    for i in INPUTS:
        for p, q in PAIRS:
            A, B = per[f"{i}_{p}"], per[f"{i}_{q}"]
            assert set(A) == set(B), (i, p, q)
            s = sorted(A)
            d = np.array([A[x] - B[x] for x in s])
            out["tests"].append({"input": i, "a": p, "b": q, "n_patients": len(s), "median_diff": float(np.median(d)),
                                 "mean_diff": float(d.mean()), "a_better": int((d > 0).sum()),
                                 "p_wilcoxon": float(wilcoxon(d).pvalue)})
    out["per_patient_macro_f1"] = per
    (a.results / "stats.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"| Input | Comparison | Patients | Mean Δ | Median Δ | A better | p |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for t in out["tests"]:
        print(f"| {t['input']} | {t['a']} vs {t['b']} | {t['n_patients']} | {t['mean_diff']:+.3f} | {t['median_diff']:+.3f} "
              f"| {t['a_better']}/{t['n_patients']} | {t['p_wilcoxon']:.2g} |")


if __name__ == "__main__":
    main()
