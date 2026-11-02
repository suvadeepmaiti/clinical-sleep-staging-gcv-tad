"""Label-efficiency experiment (iSLEEPS, NAPS or ISRUC-1): only a fraction of each fold's training patients is labelled.

Per fold (iSLEEPS: folds of results/sleepgpt4_cv10_s1; NAPS: folds of naps/results/naps_cv10_ce; ISRUC-1: folds
of results/isruc/isruc_cv10_ce; validation and test patients unchanged and labelled):
1. Labelled set: `--fraction` of the training patients, stratified by montage (iSLEEPS, ISRUC-1) or study type
   (NAPS) (seed 42 + fold); the
   other training patients are unlabelled (their scorer labels are never used for training/selection).
2. Teacher: the 4-channel SleepGPT-BiLSTM (C3, C4, chin EMG, E1; NAPS: absent C3/C4 masked, naps_cv.py),
   same recipe as the full-label run (CE, 25 epochs max, patience 8 after 10, seed offset 1), trained on the
   labelled patients only.
3. Pseudo-labels: teacher softmax(logits / T) (soft) and argmax (hard) on every epoch.
4. Context-aware students (EEG = C4-M1 (NAPS: C4, else C3), EOG = E1, EEG+EOG), streaming inference, best
   checkpoint by inner-validation macro F1:
     ce      : CE on the labelled patients only (no teacher)
     kd      : 0.3*CE + 0.5*KD on labelled + unlabelled patients
     tad     : 0.3*CE + 0.2*TAD (plain transition term, "No KD") on labelled + unlabelled patients
     gtad    : 0.3*CE + 0.2*G-weighted transition term (train_students_gtad.py), labelled + unlabelled
     gtad_cv : same with the patient-variability prior
   where CE targets are scorer labels for labelled patients and teacher hard pseudo-labels for
   unlabelled ones, and G / G_cv come from the labelled patients' scorer transitions only.
After each fold, SUMMARY.md / summary.json (fold mean +/- SD of the test metrics) are rewritten in --output.
"""
import repo_paths
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_students_context import (INPUTS, TEACHER_CHANNELS, VARIANTS, context_windows, load_data, metrics,  # noqa: E402
                                    normalization, runs_of, stream_predict, temporal_indices, train_variant,
                                    write_json)
from train_students_gtad import train_gtad, transition_priors  # noqa: E402
from engine_multi import predict as predict_teacher, train_member  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANTS.setdefault("kd", {"alpha": 0.3, "beta": 0.5, "gamma": 0.0})
STUDENT_INPUTS = {"eeg": "c4", "eeg_eog": "eeg_eog", "eog": "eog"}
KEYS = ["accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]


def labelled_subset(train, montage, fraction, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for m in sorted({montage[s] for s in train}):
        group = sorted(s for s in train if montage[s] == m)
        rng.shuffle(group)
        chosen += group[:max(1, int(round(fraction * len(group))))]
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--inputs", nargs="+", default=["eeg", "eeg_eog"], choices=list(STUDENT_INPUTS))
    parser.add_argument("--variants", nargs="+", default=["ce", "kd", "gtad", "gtad_cv"],
                        choices=["ce", "kd", "tad", "gtad", "gtad_cv"])
    parser.add_argument("--output", type=Path, help="default: results/label_efficiency_<pct>pct")
    parser.add_argument("--dataset", choices=["isleeps", "naps", "isruc"], default="isleeps")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out = (args.output or repo_paths.RESULTS / args.dataset / f"label_efficiency_{int(round(args.fraction * 100))}pct").resolve()
    data, x4, folds, montage, config, member_name, student_input, title = {"isleeps": isleeps, "naps": naps,
                                                                           "isruc": isruc}[args.dataset]()
    t = config["teacher"]
    windows = torch.as_tensor(context_windows(data, t["window"], t["target"]), device=device)
    y_np = data["y"]
    y = torch.from_numpy(y_np).to(device)
    T = config["temperature"]
    for k in args.folds:
        fold, name = folds[k], f"fold_{k:02d}"
        lab = labelled_subset(fold["train"], montage, args.fraction, 42 + k)
        unlab = sorted(set(fold["train"]) - set(lab))
        train_lab, _, _ = temporal_indices(data, lab)
        train_all, _, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        unlab_idx = np.setdiff1d(train_all, train_lab)
        folder = out / name
        split = {**fold, "fraction": args.fraction, "labelled": lab, "unlabelled": unlab,
                 "n_labelled_epochs": len(train_lab), "n_unlabelled_epochs": len(unlab_idx)}
        write_json(folder / "split.json", split)
        print(f"{name}: labelled {len(lab)} patients ({len(train_lab)} epochs), unlabelled {len(unlab)} "
              f"({len(unlab_idx)} epochs), val {len(fold['val'])}, test {len(fold['test'])}", flush=True)
        # Teacher on the labelled patients only.
        stats4 = normalization(data, lab)
        metadata = {**split, "normalization": stats4}
        member = {**t["member"], "name": member_name, "seed_offset": 1}
        x4_gpu = x4.to(device)
        teacher, t_res = train_member(member, "ce", x4_gpu, y, train_lab, val, windows, folder / "teacher",
                                      metadata, config, stats4, folder / "status.json")
        logits = predict_teacher(teacher, x4_gpu, windows, np.arange(len(y_np)), True)
        del teacher, x4_gpu
        torch.cuda.empty_cache()
        soft = torch.softmax(logits / T, -1)
        pseudo = logits.argmax(-1)
        t_test = metrics(y_np[test], pseudo[torch.as_tensor(test, device=device)].cpu().numpy())
        t_unlab = metrics(y_np[unlab_idx], pseudo[torch.as_tensor(unlab_idx, device=device)].cpu().numpy())
        print(f"TEACHER {name} val={t_res['best_validation_macro_f1']:.4f} test F1={t_test['macro_f1']:.4f} "
              f"acc={t_test['accuracy']:.4f} | pseudo-label quality on unlabelled train patients: "
              f"acc={t_unlab['accuracy']:.4f} F1={t_unlab['macro_f1']:.4f}", flush=True)
        results = {"teacher": {**t_res, "test": t_test, "pseudo_label_quality_unlabelled": t_unlab}}
        y_mix = y.clone()
        y_mix[torch.as_tensor(unlab_idx, device=device)] = pseudo[torch.as_tensor(unlab_idx, device=device)]
        runs_lab, runs_all = runs_of(data, train_lab), runs_of(data, train_all)
        val_runs, test_runs = runs_of(data, val), runs_of(data, test)
        priors = transition_priors(y_np, runs_lab, data["subject"])
        write_json(folder / "transition_priors.json", {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                       for key, v in priors.items()})
        test_t = torch.as_tensor(test, device=device)
        for inp in args.inputs:
            chans, stats, x = student_input(inp, lab)
            x = x.to(device)
            assert torch.isfinite(x).all()
            if len(chans) == 1:
                x = x[:, 0].contiguous()
            s_config = {**config, "channels": chans, "input": inp, "seq_len": 20, "seqs_per_batch": 32,
                        "student": "SleepStudentNet (unchanged), stateful causal", "fraction": args.fraction}
            s_meta = {**fold, "normalization": stats, "channels": chans, "labelled": lab}
            ckpt = folder / inp / "checkpoints"
            for kind in args.variants:
                if kind == "ce":
                    model, res = train_variant("ce_only", x, y, runs_lab, val_runs, soft, ckpt / "ce", s_meta, s_config, device)
                elif kind == "kd":
                    model, res = train_variant("kd", x, y_mix, runs_all, val_runs, soft, ckpt / "kd", s_meta, s_config, device)
                elif kind == "tad":
                    model, res = train_variant("no_kd", x, y_mix, runs_all, val_runs, soft, ckpt / "tad", s_meta, s_config, device)
                else:
                    model, res = train_gtad(kind, priors[kind], x, y_mix, runs_all, val_runs, soft, ckpt / kind,
                                            s_meta, s_config, device)
                pr = torch.softmax(stream_predict(model, x, test_runs, device)[test_t], -1).cpu().numpy()
                results[f"{inp}_{kind}"] = {**res, "test": metrics(y_np[test], pr.argmax(-1))}
                np.save(folder / inp / f"{kind}_test_prob.npy", pr)
                s = results[f"{inp}_{kind}"]["test"]
                print(f"TEST {name} {inp:7s} {kind:8s} val={res['validation']['macro_f1']:.4f} acc={s['accuracy']:.4f} "
                      f"F1={s['macro_f1']:.4f} kappa={s['kappa']:.4f} " + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in
                                                                                  ["W", "N1", "N2", "N3", "REM"]), flush=True)
                del model
            del x
            torch.cuda.empty_cache()
        write_json(folder / "test_metrics.json", {"fold": k, "fraction": args.fraction, "test_subjects": fold["test"],
                                                  "models": results})
        lines = [f"# {title} {name}, {int(args.fraction * 100)}% labelled training patients ({len(lab)} of "
                 f"{len(fold['train'])}); test {len(fold['test'])} patients", "",
                 "| Model | Val macro F1 | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, v in results.items():
            s, vf = v["test"], (v.get("best_validation_macro_f1") or v["validation"]["macro_f1"])
            lines.append(f"| {key} | {vf:.3f} | " + " | ".join(f"{s[m]:.3f}" for m in
                         ["accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]) + " |")
        (folder / "RESULTS.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        summarize(out, args.fraction, title)


def isleeps():
    src = repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10_s1"
    config = json.loads((src / "config.json").read_text())
    folds = {f["fold"]: f for f in json.loads((src / "folds.json").read_text())}
    data = load_data(repo_paths.ISLEEPS_CACHE / "psg4_100hz", TEACHER_CHANNELS)
    strata = {r["subject"]: r["montage"] for r in data["records"]}
    x4 = torch.from_numpy(data["x"])

    def student_input(inp, lab):
        chans = INPUTS[STUDENT_INPUTS[inp]]
        sel = [TEACHER_CHANNELS.index(c) for c in chans]
        sdata = {**data, "records": [{**r, "sum": [r["sum"][i] for i in sel], "sum_squares": [r["sum_squares"][i] for i in sel]}
                                     for r in data["records"]]}
        stats = normalization(sdata, lab)
        x = x4[:, sel].sub_(torch.tensor(stats["mean_uV"]).view(1, -1, 1)).div_(torch.tensor(stats["std_uV"]).view(1, -1, 1))
        return chans, stats, x
    return data, x4, folds, strata, config, "sleepgpt4_lab", student_input, "iSLEEPS"


def naps():
    import naps_cv  # sets engine_multi.build to the masked 4-slot SleepGPT teacher used by train_member
    src = repo_paths.RESULTS / "naps" / "naps_cv10_ce"
    config = json.loads((src / "config.json").read_text())
    folds = {f["fold"]: f for f in json.loads((src / "folds.json").read_text())}
    data = naps_cv.load(repo_paths.PREPROCESSED["naps"])
    strata = {r["subject"]: r["study_type"] for r in data["records"]}
    x4 = torch.from_numpy(data.pop("x"))
    names = {"eeg": "c4", "eeg_eog": "eeg_eog", "eog": "eog"}

    def student_input(inp, lab):
        chans = naps_cv.STUDENT_INPUTS[names[inp]]
        stats_all = normalization(data, lab)
        idx = [naps_cv.STAT_CHANNELS.index(c) for c in chans]
        stats = {"mean_uV": [stats_all["mean_uV"][i] for i in idx], "std_uV": [stats_all["std_uV"][i] for i in idx]}
        x = torch.stack([naps_cv.student_channel(x4, c) for c in chans], 1)
        x.sub_(torch.tensor(stats["mean_uV"]).view(1, -1, 1)).div_(torch.tensor(stats["std_uV"]).view(1, -1, 1))
        return chans, stats, x
    return data, x4, folds, strata, config, "sleepgpt4m_lab", student_input, "NAPS"


def isruc():
    import isruc_cv  # imports naps_cv (masked 4-slot teacher); per-recording amplitude normalization, recording 40 excluded
    import naps_cv
    src = repo_paths.RESULTS / "isruc" / "isruc_cv10_ce"
    config = json.loads((src / "config.json").read_text())
    folds = {f["fold"]: f for f in json.loads((src / "folds.json").read_text())}
    data = isruc_cv.load(repo_paths.PREPROCESSED["isruc"])
    strata = {r["subject"]: r["montage"] for r in data["records"]}
    x4 = torch.from_numpy(data.pop("x"))
    names = {"eeg": "c4", "eeg_eog": "eeg_eog", "eog": "eog"}

    def student_input(inp, lab):
        chans = naps_cv.STUDENT_INPUTS[names[inp]]
        stats_all = normalization(data, lab)
        idx = [naps_cv.STAT_CHANNELS.index(c) for c in chans]
        stats = {"mean_uV": [stats_all["mean_uV"][i] for i in idx], "std_uV": [stats_all["std_uV"][i] for i in idx]}
        x = torch.stack([naps_cv.student_channel(x4, c) for c in chans], 1)
        x.sub_(torch.tensor(stats["mean_uV"]).view(1, -1, 1)).div_(torch.tensor(stats["std_uV"]).view(1, -1, 1))
        return chans, stats, x
    return data, x4, folds, strata, config, "sleepgpt4m_lab", student_input, "ISRUC-1"


def summarize(out, fraction, title="iSLEEPS"):
    per = {}
    for path in sorted(out.glob("fold_*/test_metrics.json")):
        for key, v in json.loads(path.read_text())["models"].items():
            per.setdefault(key, []).append([v["test"][k] for k in KEYS])
    rows = {key: np.array(v) for key, v in per.items()}
    write_json(out / "summary.json", {key: {"folds": len(v), **{f"{k}_mean": float(v[:, i].mean()) for i, k in enumerate(KEYS)},
                                            **{f"{k}_sd": float(v[:, i].std(ddof=1)) if len(v) > 1 else 0.0 for i, k in enumerate(KEYS)}}
                                      for key, v in rows.items()})
    lines = [f"# {title} label efficiency ({int(round(fraction * 100))}% labelled training patients): test, fold mean ± SD", "",
             "| Model | Folds | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, v in rows.items():
        m, sd = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(len(KEYS)))
        lines.append(f"| {key} | {len(v)} | " + " | ".join(f"{m[i]:.3f} ± {sd[i]:.3f}" for i in range(3)) + " | "
                     + " | ".join(f"{m[i]:.3f}" for i in range(3, len(KEYS))) + " |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
