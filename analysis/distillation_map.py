"""Distillation map: where the G_cv-TAD loss pushed the student, epoch by epoch (writes fig_distillation_map.png and
distillation_map.json).

Fold 1 of each cohort, EEG+EOG students, inference only with the saved checkpoints:
  teacher soft labels p_t = softmax(z_t / T) (T = 2) from the fold's 25% teacher,
  loss weight      w_t = c_{t-1} * sum_k G_cv[s, k] p_{t,k},  c_{t-1} = max_k p_{t-1,k}, s = argmax_k p_{t-1,k},
  remaining gap    d_t = T^2 c_{t-1} sum_k G_cv[s, k] (p log(p/q) - p + q)  for the trained CE and G_cv-TAD students,
on consecutive epochs of the same continuous run (as in training). Epoch categories from the scorer hypnogram:
at a stage change (stage differs from the previous or next epoch), near one (within 2 epochs), stable.
The map shows the unlabelled training patient of iSLEEPS fold 1 with the median number of stage changes per hour
(the loss acted on training patients; their scorer labels were never used in training and serve only as reference).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
import label_efficiency as le  # noqa: E402
import engine_multi  # noqa: E402  (build is looked up at call time: naps_cv swaps in the masked teacher)
from train_students_context import SleepStudentNet, context_windows, runs_of, stream_predict, temporal_indices  # noqa: E402

HERE = repo_paths.ANALYSIS  # outputs
RESULTS = {k: repo_paths.RESULTS / k / "label_efficiency_25pct" for k in ("isleeps", "naps", "isruc")}
FOLD = 1


def categories(y, runs):
    """0 stable, 1 near a change (<= 2 epochs), 2 at a change; -1 outside runs."""
    cat = np.full(len(y), -1)
    for r in runs:
        yr = y[r]
        change = np.flatnonzero(yr[1:] != yr[:-1])  # change between i and i+1
        at = np.zeros(len(r), bool)
        at[change] = True
        at[change + 1] = True
        dist = np.full(len(r), 10**6)
        for c in np.flatnonzero(at):
            dist = np.minimum(dist, np.abs(np.arange(len(r)) - c))
        cat[r] = np.where(at, 2, np.where(dist <= 2, 1, 0))
    return cat


def per_epoch(p, q, G, runs, T):
    """Loss weight w_t and gap d_t for every epoch that has a predecessor in its run (NaN otherwise)."""
    w, d = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for r in runs:
        prev, cur = r[:-1], r[1:]
        c = p[prev].max(1)
        g = G[p[prev].argmax(1)]
        w[cur] = c * (g * p[cur]).sum(1)
        if q is not None:
            pc, qc = np.clip(p[cur], 1e-8, 1), np.clip(q[cur], 1e-8, 1)
            d[cur] = T ** 2 * c * (g * (pc * np.log(pc / qc) - pc + qc)).sum(1)
    return w, d


def run_cohort(key, device):
    data, x4, folds, montage, config, member_name, student_input, _ = {"isleeps": le.isleeps, "naps": le.naps,
                                                                        "isruc": le.isruc}[key]()
    fold, folder = folds[FOLD], RESULTS[key] / f"fold_{FOLD:02d}"
    split = json.loads((folder / "split.json").read_text())
    lab = le.labelled_subset(fold["train"], montage, 0.25, 42 + FOLD)
    assert lab == split["labelled"], "labelled patients differ from the saved split"
    T, t = config["temperature"], config["teacher"]
    train_all, _, _ = temporal_indices(data, fold["train"])
    runs = runs_of(data, train_all)
    # Teacher soft labels on all training epochs.
    member = {**t["member"], "name": member_name, "seed_offset": 1}
    teacher = engine_multi.build(member, config, le.normalization(data, lab)).to(device)
    teacher.load_state_dict(torch.load(folder / "teacher/teacher.pt", map_location=device, weights_only=True)
                            ["model_state_dict"])
    windows = torch.as_tensor(context_windows(data, t["window"], t["target"]), device=device)
    logits = engine_multi.predict(teacher, x4.to(device), windows, np.asarray(train_all), True)
    del teacher
    torch.cuda.empty_cache()
    p = np.zeros((len(data["y"]), 5), np.float32)
    p[train_all] = torch.softmax(logits[torch.as_tensor(train_all, device=device)].float() / T, -1).cpu().numpy() \
        if logits.shape[0] == len(data["y"]) else torch.softmax(logits.float() / T, -1).cpu().numpy()
    # Students (EEG+EOG): streaming predictions on the training runs.
    _, _, x = student_input("eeg_eog", lab)
    x = x.to(device)
    q, pred = {}, {}
    for kind, f in (("ce", "ce/ce_only.pt"), ("gtad_cv", "gtad_cv/gtad_cv.pt")):
        m = SleepStudentNet(n_channels=x.shape[1]).to(device)
        m.load_state_dict(torch.load(folder / "eeg_eog/checkpoints" / f, map_location=device, weights_only=True)
                          ["model_state_dict"])
        z = stream_predict(m, x, runs, device).float()
        q[kind] = torch.softmax(z / T, -1).cpu().numpy()
        pred[kind] = z.argmax(-1).cpu().numpy()
    del x
    torch.cuda.empty_cache()
    G = np.array(json.loads((folder / "transition_priors.json").read_text())["gtad_cv"])
    w, d_g = per_epoch(p, q["gtad_cv"], G, runs, T)
    _, d_c = per_epoch(p, q["ce"], G, runs, T)
    y = data["y"]
    cat = categories(y, runs)
    ok = ~np.isnan(w)
    stats = {"epochs": int(ok.sum()), "categories": {}}
    for ci, name in enumerate(("stable", "near", "at_change")):
        m = ok & (cat == ci)
        stats["categories"][name] = {"epoch_share": float(m.sum() / ok.sum()), "weight_share": float(w[m].sum() / w[ok].sum()),
                                     "gap_ce": float(np.nanmean(d_c[m])), "gap_gtad_cv": float(np.nanmean(d_g[m]))}
    stats["n1"] = {"epoch_share": float((ok & (y == 1)).sum() / ok.sum()),
                   "weight_share": float(w[ok & (y == 1)].sum() / w[ok].sum())}
    disagree = ok & (pred["ce"] != pred["gtad_cv"])
    stats["disagreement"] = {"share": float(disagree.mean() / ok.mean()),
                             "at_or_near_change": float((disagree & (cat >= 1)).sum() / max(disagree.sum(), 1)),
                             "gtad_cv_right": float((disagree & (pred["gtad_cv"] == y)).sum() / max(disagree.sum(), 1)),
                             "ce_right": float((disagree & (pred["ce"] == y)).sum() / max(disagree.sum(), 1))}
    example = None
    if key == "isleeps":
        # Per unlabelled training patient: stage changes per hour and who wins the CE vs G_cv-TAD disagreements.
        per = {}
        for s in split["unlabelled"]:
            idx = np.flatnonzero(data["subject"] == s)
            dd = disagree[idx]
            per[s] = {"changes_per_h": float((cat[idx] == 2).sum() / 2 / (len(idx) * 30 / 3600)),
                      "disagreements": int(dd.sum()),
                      "gtad_cv_right": int((dd & (pred["gtad_cv"][idx] == y[idx])).sum()),
                      "ce_right": int((dd & (pred["ce"][idx] == y[idx])).sum()),
                      "teacher_acc": float((p[idx].argmax(1) == y[idx]).mean())}
        stats["per_patient"] = per
        net = {s: (v["gtad_cv_right"] - v["ce_right"]) / max(v["disagreements"], 1) for s, v in per.items()}
        order = sorted(net, key=net.get)
        picks = {"median_outcome": order[len(order) // 2], "worst_outcome": order[0]}

        def pack(s):
            idx = np.flatnonzero(data["subject"] == s)
            return {"patient": s, **per[s], "y": y[idx].tolist(), "p": p[idx].tolist(), "w": np.nan_to_num(w[idx]).tolist(),
                    "d_ce": np.nan_to_num(d_c[idx]).tolist(), "d_gtad_cv": np.nan_to_num(d_g[idx]).tolist(),
                    "pred_ce": pred["ce"][idx].tolist(), "pred_gtad_cv": pred["gtad_cv"][idx].tolist(),
                    "cat": cat[idx].tolist()}
        example = {k: pack(s) for k, s in picks.items()}
    return stats, example


def main():
    device = torch.device("cuda:0")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out, example = {}, None
    for key in RESULTS:
        part = HERE / f"distillation_map_{key}.json"  # per-cohort cache, so a later failure keeps finished cohorts
        if part.exists():
            saved = json.loads(part.read_text())
        else:
            stats, ex = run_cohort(key, device)
            saved = {"stats": stats, "example": ex}
            part.write_text(json.dumps(saved))
        out[key] = saved["stats"]
        example = saved["example"] or example
        print(key, json.dumps(saved["stats"]), flush=True)
    (HERE / "distillation_map.json").write_text(json.dumps({"stats": out, "example": example}))


if __name__ == "__main__":
    main()
