"""Sleep fragmentation of each cohort from the scorer hypnograms (writes fragmentation.json).

Per recording, within the sleep period (first to last non-wake epoch), counting only transitions between
contiguous epochs (epoch_index step 1): stage shifts per hour, awakenings (sleep -> W) per hour, N1 % of total
sleep time and sleep efficiency. Also the mean between-patient coefficient of variation of the row-normalized
transition matrices (entries with mean probability >= 0.01), i.e. the heterogeneity G_cv discounts.
Sleep-EDF-20 (healthy, R&K stages 3+4 merged) is a reference only; it is not used for training.
Mann-Whitney U tests compare each patient cohort with Sleep-EDF-20 (one value per recording).
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
COHORTS = repo_paths.PREPROCESSED


def recordings(key):
    if key == "sleepedf20":
        return sorted(COHORTS[key].glob("SC*.npz"))
    s = json.loads((repo_paths.RESULTS / key / "label_efficiency_25pct/fold_01/split.json").read_text())
    return [COHORTS[key] / f"{r}.npz" for r in sorted(s["train"] + s["val"] + s["test"])]


def per_recording(path):
    with np.load(path) as z:
        y, idx = z["y"], z["epoch_index"]
    n_epochs = len(y)
    sleep = np.flatnonzero(y != 0)
    y, idx = y[sleep[0]:sleep[-1] + 1], idx[sleep[0]:sleep[-1] + 1]
    contiguous = np.diff(idx) == 1
    a, b = y[:-1][contiguous], y[1:][contiguous]
    hours = len(y) * 30 / 3600
    m = np.zeros((5, 5))
    np.add.at(m, (a, b), 1)
    return {"shifts_per_h": float((a != b).sum() / hours), "awakenings_per_h": float(((a != 0) & (b == 0)).sum() / hours),
            "n1_pct": float(100 * (y == 1).sum() / (y != 0).sum()), "sleep_eff": float(100 * (y != 0).mean()),
            "n_epochs": n_epochs, "M": (m / np.maximum(m.sum(1, keepdims=True), 1)).tolist()}


def main():
    raw = {k: [per_recording(p) for p in recordings(k)] for k in COHORTS}
    out = {}
    for k, rows in raw.items():
        M = np.array([r["M"] for r in rows])
        mean, sd = M.mean(0), M.std(0)
        cv = float((sd / (mean + 1e-6))[mean >= 0.01].mean())
        out[k] = {"n": len(rows), "n_epochs": sum(r["n_epochs"] for r in rows), "transition_cv": cv}
        for m in ("shifts_per_h", "awakenings_per_h", "n1_pct", "sleep_eff"):
            v = np.array([r[m] for r in rows])
            out[k][m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v))}
            if k != "sleepedf20":
                ref = [r[m] for r in raw["sleepedf20"]]
                out[k][m]["p_vs_healthy"] = float(mannwhitneyu(v, ref).pvalue)
    (repo_paths.ANALYSIS / "fragmentation.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k, v["n"], *(f"{m}={v[m]['mean']:.1f}±{v[m]['sd']:.1f}" + (f" p={v[m]['p_vs_healthy']:.1g}" if "p_vs_healthy" in v[m] else "")
                          for m in ("shifts_per_h", "awakenings_per_h", "n1_pct", "sleep_eff")), f"CV={v['transition_cv']:.2f}")


if __name__ == "__main__":
    main()
