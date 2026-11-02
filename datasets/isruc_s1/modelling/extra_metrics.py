"""Balanced accuracy / weighted F1 / kappa from saved CV test predictions (CBraMod-style metrics), no retraining."""
import sys
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)

MODELS = ("teacher", "eeg_eog_ce_only", "c4_ce_only", "eog_ce_only")


def metrics(y, p):
    return {"acc": accuracy_score(y, p), "bacc": balanced_accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"), "kappa": cohen_kappa_score(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=repo_paths.RESULTS / "isruc" / "isruc_cv10_ce")
    a = ap.parse_args()
    paths = [p for p in sorted(a.results.glob("folds/fold_*/test_predictions.npz"))]
    data = [dict(np.load(p, allow_pickle=False)) for p in paths]
    subset = [f"ISRUC1_{i:03d}" for i in range(91, 101)]  # CBraMod's test subjects
    out = {"n_folds": len(paths), "fold_mean": {}, "fold_sd": {}, "pooled": {}, "subjects_91_100": {}}
    for m in MODELS:
        per = [metrics(d["y_true"], d[f"{m}_prob"].argmax(-1)) for d in data]
        out["fold_mean"][m] = {k: float(np.mean([f[k] for f in per])) for k in per[0]}
        out["fold_sd"][m] = {k: float(np.std([f[k] for f in per], ddof=1)) for k in per[0]}
        y = np.concatenate([d["y_true"] for d in data]); p = np.concatenate([d[f"{m}_prob"].argmax(-1) for d in data])
        s = np.concatenate([d["subject"] for d in data])
        out["pooled"][m] = metrics(y, p)
        mask = np.isin(s, subset)
        out["subjects_91_100"][m] = metrics(y[mask], p[mask])
        out["subjects_91_100_present"] = sorted(set(s[mask].tolist()))
    (a.results / "extra_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(paths)} folds; subjects 91-100 present: {len(out['subjects_91_100_present'])}/10")
    for view in ("fold_mean", "pooled", "subjects_91_100"):
        print(f"\n{view}\n| Model | Acc | Balanced acc | Macro F1 | Weighted F1 | Kappa |\n|---|---:|---:|---:|---:|---:|")
        for m in MODELS:
            r = out[view][m]
            print(f"| {m} | {r['acc']:.3f} | {r['bacc']:.3f} | {r['macro_f1']:.3f} | {r['weighted_f1']:.3f} | {r['kappa']:.3f} |")


if __name__ == "__main__":
    main()
