"""Sleep-EDF-20 leave-one-subject-out (LOSO) with the iSLEEPS/NAPS/ISRUC-1 pipeline (run fold by fold with --folds).

Data: preprocessed_sleepedf20, 39 nights of 20 healthy subjects (SC cassette), channels EEG Fpz-Cz and EOG horizontal
(chin EMG is stored at 1 Hz in these recordings and is not used).
Amplitude: per-recording normalization as in ISRUC-1 (isruc_cv.py): each recording and channel is centred on its own
mean and scaled so that its median 30-s epoch SD equals a fixed reference SD (15 uV); labels are never used.
Folds: 20 folds, each holds out one subject (both nights); of the other 19, 2 validate (NumPy seed 42 + fold) and
17 train. Best epochs/checkpoints by inner-validation macro F1 only.
Per fold, two settings:
  100% labels: teacher (masked SleepGPT-BiLSTM, naps_cv.py: slots C3, C4, EMG, EOG; Fpz-Cz in the C4 slot,
               EOG in the EOG slot, C3 and EMG absent = masked) trained with CE on the 17 training subjects;
               student EEG+EOG (context-aware SleepStudentNet) with CE only.
  25% labels:  labelled subset of the training subjects (label_efficiency.labelled_subset, seed 42 + fold);
               teacher as above on the labelled subjects only; student EEG+EOG with G_cv-TAD
               (0.3 CE + 0.2 G_cv-weighted transition term, train_students_gtad.py) on labelled + unlabelled
               subjects, CE targets = scorer labels (labelled) or teacher hard pseudo-labels (unlabelled),
               G_cv from the labelled subjects' scorer transitions only.
Teacher/student recipes are those of ISRUC-S1 (isruc_cv.py) and label_efficiency.py. Shared code (src/,
naps_cv.py) is imported, never modified.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
import naps_cv  # noqa: E402  (sets engine_multi.build to the masked 4-slot SleepGPT teacher; adds the import paths)
from naps_cv import SLOTS, STAT_CHANNELS, TEACHER_CHANNELS, student_channel  # noqa: E402
from engine_multi import predict as predict_teacher, train_member  # noqa: E402
from protocol import VARIANTS, metrics, normalization, sha256, temporal_indices, write_json  # noqa: E402
from protocol_isleeps import context_windows  # noqa: E402
from sleepgpt_c4 import CHECKPOINT  # noqa: E402
from train_students_context import runs_of, stream_predict, train_variant  # noqa: E402
from train_students_gtad import train_gtad, transition_priors  # noqa: E402
from label_efficiency import labelled_subset  # noqa: E402

SOURCE = {"C4-M1": "EEG Fpz-Cz", "E1": "EOG horizontal"}   # teacher slot name -> Sleep-EDF channel; C3, EMG absent
REF_SD_UV = 15.0
STUDENT = ["EEG", "E1"]                                     # EEG+EOG student: Fpz-Cz, EOG horizontal
FRACTION = 0.25
KEYS = ["accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
MODELS = ["teacher_full", "student_ce_full", "teacher_25", "student_gtad_cv_25"]


def load(data_dir):
    with (data_dir / "records.csv").open() as stream:
        recs = sorted(csv.DictReader(stream), key=lambda r: r["recording"])
    total = sum(int(r["retained_epochs"]) for r in recs)
    x_all = np.full((total, len(TEACHER_CHANNELS), 3000), np.nan, dtype=np.float32)
    ys, subj, rec_ids, epochs, segments, records, scaling = [], [], [], [], [], [], {}
    offset = 0
    for r in recs:
        with np.load(data_dir / r["file"], allow_pickle=False) as d:
            names = d["channels"].tolist()
            assert d["fs"].item() == 100 and str(d["recording"]) == r["recording"]
            y = d["y"].astype(np.int64)
            x = x_all[offset:offset + len(y)]
            scaling[r["recording"]] = {}
            for slot, src in SOURCE.items():   # per-recording normalization, no labels
                v = d["x"][:, names.index(src)].astype(np.float32)
                assert np.isfinite(v).all()
                mean, med_sd = float(v.mean()), float(np.median(v.std(-1)))
                assert med_sd > 0, (r["recording"], src)
                x[:, TEACHER_CHANNELS.index(slot)] = (v - mean) * (REF_SD_UV / med_sd)
                scaling[r["recording"]][src] = {"mean_uV": mean, "median_epoch_sd_uV": med_sd, "gain": REF_SD_UV / med_sd}
            eeg, eog = x[:, TEACHER_CHANNELS.index("C4-M1")], x[:, TEACHER_CHANNELS.index("E1")]
            n = int(len(y) * 3000)
            # Student statistics over STAT_CHANNELS (EEG, EMG, E1); EMG is absent: placeholder mean 0, SD 1 (unused).
            ys.append(y); subj.append(np.full(len(y), r["subject"])); rec_ids.append(np.full(len(y), r["recording"]))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": r["recording"], "subject": r["subject"], "start": offset, "stop": offset + len(y),
                            "sha256": r["output_sha256"],
                            "sum": [float(eeg.sum(dtype=np.float64)), 0.0, float(eog.sum(dtype=np.float64))],
                            "sum_squares": [float(np.square(eeg, dtype=np.float64).sum()), float(n),
                                            float(np.square(eog, dtype=np.float64).sum())],
                            "sample_count": n})
            offset += len(y)
    assert offset == total
    data = {k: np.concatenate(v) for k, v in zip(["y", "subject", "recording", "epoch_index", "segment_id"],
                                                 [ys, subj, rec_ids, epochs, segments])}
    data["x"], data["records"], data["scaling"] = x_all, records, scaling
    return data


def loso_folds(subjects, n_val=2, seed=42):
    subjects = sorted(subjects)
    folds = []
    for k, test in enumerate(subjects, 1):
        rest = [s for s in subjects if s != test]
        val = sorted(np.random.default_rng(seed + k).choice(rest, n_val, replace=False).tolist())
        folds.append({"fold": k, "train": [s for s in rest if s not in val], "val": val, "test": [test]})
    return folds


def student_x(x4, stats, device):
    sel = [STAT_CHANNELS.index(c) for c in STUDENT]
    mean = torch.tensor([stats["mean_uV"][i] for i in sel], device=device).view(1, -1, 1)
    std = torch.tensor([stats["std_uV"][i] for i in sel], device=device).view(1, -1, 1)
    x = (torch.stack([student_channel(x4, c) for c in STUDENT], 1) - mean) / std
    assert torch.isfinite(x).all()
    return x, {"mean_uV": [stats["mean_uV"][i] for i in sel], "std_uV": [stats["std_uV"][i] for i in sel]}


def summarize(out):
    per, pooled = {}, {}
    for path in sorted((out / "folds").glob("fold_*/test_metrics.json")):
        for key, v in json.loads(path.read_text())["models"].items():
            per.setdefault(key, []).append([v["test"][k] for k in KEYS])
            pooled[key] = pooled.get(key, 0) + np.array(v["test"]["confusion_matrix"])
    summary = {}
    for key, rows in per.items():
        v, cm = np.array(rows), pooled[key]
        y = np.repeat(np.arange(5), cm.sum(1))
        pred = np.concatenate([np.repeat(np.arange(5), cm[i]) for i in range(5)])
        summary[key] = {"folds": len(v), **{f"{k}_mean": float(v[:, i].mean()) for i, k in enumerate(KEYS)},
                        **{f"{k}_sd": float(v[:, i].std(ddof=1)) if len(v) > 1 else 0.0 for i, k in enumerate(KEYS)},
                        "pooled": {k: m for k, m in metrics(y, pred).items() if k in KEYS}}
    write_json(out / "summary.json", summary)
    lines = ["# Sleep-EDF-20 LOSO: test, mean over held-out subjects ± SD (pooled over all test epochs)", "",
             "| Model | Folds | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in MODELS:
        if key in summary:
            s = summary[key]
            lines.append(f"| {key} | {s['folds']} | " + " | ".join(
                f"{s[k + '_mean']:.3f} ± {s[k + '_sd']:.3f} ({s['pooled'][k]:.3f})" for k in KEYS[:3]) + " | "
                + " | ".join(f"{s[k + '_mean']:.3f}" for k in KEYS[3:]) + " |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["sleepedf20"])
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "sleepedf20" / "sleepedf20_loso")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--check", action="store_true", help="load data, print folds and scaling, no training")
    args = parser.parse_args()
    data = load(args.data_dir)
    subjects = sorted(str(s) for s in set(data["subject"]))
    folds = loso_folds(subjects)
    strata = {s: "all" for s in subjects}
    print(f"{len(data['records'])} recordings, {len(subjects)} subjects, {len(data['y'])} epochs", flush=True)
    if args.check:
        for src in SOURCE.values():
            g = [v[src]["gain"] for v in data["scaling"].values()]
            print(f"{src}: gain median {np.median(g):.2f} (range {min(g):.2f}-{max(g):.2f})")
        for f in folds[:3]:
            lab = labelled_subset(f["train"], strata, FRACTION, 42 + f["fold"])
            print(f"fold {f['fold']}: test {f['test']} val {f['val']} train {len(f['train'])} labelled {lab}")
        print("normalization (all):", normalization(data, subjects)["std_uV"])
        return
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)
    member = {"name": "sleepgpt4m", "arch": "sleepgpt", "epochs": 25, "lr": 1e-4,
              "weight_decay": 0.05, "drop_path_rate": 0.2, "chunk": 64, "chunks_per_batch": 2}
    config = {
        "dataset": "Sleep-EDF-20 (SC)", "protocol": "LOSO, 2 inner-validation subjects (seed 42 + fold)",
        "amplitude_normalization": {"method": "per recording and channel: subtract mean, scale median 30-s epoch SD "
                                              "to a fixed reference; no labels", "reference_sd_uV": REF_SD_UV},
        "teacher": {"architecture": "SleepGPT-BiLSTM", "member": member, "channels": TEACHER_CHANNELS,
                    "sleepedf_channels": SOURCE, "absent_slots": ["C3", "EMG"], "sleepgpt_slots": list(SLOTS),
                    "checkpoint_sha256": sha256(CHECKPOINT), "loss": "ce",
                    "window": 21, "target": 10, "lstm_hidden": 100, "lstm_layers": 3, "dropout": 0.1,
                    "layer_decay": 0.75, "warmup_fraction": 0.1},
        "patience": 8, "min_epochs": 10, "seed": 42, "amp": True, "temperature": 2.0, "variants": VARIANTS,
        "student_epochs": 40, "student_batch": 256, "teacher_batch": 128, "seq_len": 20, "seqs_per_batch": 32,
        "fraction": FRACTION, "student_channels": STUDENT,
        "student": "SleepStudentNet (unchanged), stateful causal (context-aware)",
        "data_sha256": {r["recording"]: r["sha256"] for r in data["records"]},
        "pipeline_sha256": {n: sha256(p) for n, p in [("sleepedf_cv.py", Path(__file__)),
                            ("naps_cv.py", repo_paths.find("naps_cv.py")),
                            ("engine_multi.py", repo_paths.find("engine_multi.py")),
                            ("teachers_multi.py", repo_paths.find("teachers_multi.py")),
                            ("train_students_context.py", repo_paths.find("train_students_context.py")),
                            ("train_students_gtad.py", repo_paths.find("train_students_gtad.py"))]},
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise RuntimeError("Existing output has different settings/code/data; choose a new --output")
    else:
        write_json(out / "config.json", config)
        write_json(out / "folds.json", folds)
        write_json(out / "amplitude_scaling.json", data["scaling"])
    x4 = torch.from_numpy(data.pop("x")).to(device)
    windows = torch.as_tensor(context_windows(data, 21, 10), device=device)
    y_np = data["y"]
    y = torch.from_numpy(y_np).to(device)
    all_idx = np.arange(len(y_np))
    T = config["temperature"]
    s_config = {**config, "channels": STUDENT, "input": "eeg_eog"}
    for k in args.folds:
        fold, name = folds[k - 1], f"fold_{k:02d}"
        folder, ckpt = out / "folds" / name, out / "checkpoints" / name
        if (folder / "test_metrics.json").exists():
            print(f"Already complete: {name}", flush=True)
            continue
        train, _, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        assert len(train) + len(val) + len(test) == len(y_np)
        lab = labelled_subset(fold["train"], strata, FRACTION, 42 + k)
        train_lab, _, _ = temporal_indices(data, lab)
        unlab_idx = np.setdiff1d(train, train_lab)
        write_json(folder / "split.json", {**fold, "labelled": lab, "n_train": len(train), "n_val": len(val),
                                           "n_test": len(test), "n_labelled_epochs": len(train_lab)})
        print(f"\n{name}: test {fold['test']}, val {fold['val']}, train {len(fold['train'])} subjects "
              f"({len(train)} epochs), labelled {lab} ({len(train_lab)} epochs)", flush=True)
        tt = torch.as_tensor(test, device=device)
        test_y = y_np[test]
        train_runs, val_runs, test_runs = runs_of(data, train), runs_of(data, val), runs_of(data, test)
        results, probs = {}, {"y_true": test_y, "recording": data["recording"][test], "epoch_index": data["epoch_index"][test]}

        def teacher(key, subjects_, idx):
            stats = normalization(data, subjects_)
            meta = {**fold, "normalization": stats, "normalization_channels": STAT_CHANNELS, "labelled": subjects_}
            model, res = train_member({**member, "name": f"sleepgpt4m_{key}", "seed_offset": 1}, "ce", x4, y, idx, val,
                                      windows, ckpt / f"teacher_{key}", meta, config, stats, out / "status.json")
            z = predict_teacher(model, x4, windows, all_idx, True)
            del model
            torch.cuda.empty_cache()
            pt = torch.softmax(z[tt], -1).cpu().numpy()
            results[f"teacher_{key}"] = {**res, "test": metrics(test_y, pt.argmax(-1))}
            probs[f"teacher_{key}_prob"] = pt
            return z, stats

        def student(key, model):
            pr = torch.softmax(stream_predict(model, x, test_runs, device)[tt], -1).cpu().numpy()
            results[key]["test"] = metrics(test_y, pr.argmax(-1))
            probs[f"{key}_prob"] = pr

        # 100% labels: teacher and CE student on all training subjects.
        z, stats = teacher("full", fold["train"], train)
        x, s_stats = student_x(x4, stats, device)
        model, res = train_variant("ce_only", x, y, train_runs, val_runs, torch.softmax(z / T, -1), ckpt / "student_full",
                                   {**fold, "normalization": s_stats, "channels": STUDENT}, s_config, device)
        results["student_ce_full"] = res
        student("student_ce_full", model)
        del model, x, z
        # 25% labels: teacher on the labelled subjects; G_cv-TAD student on labelled + unlabelled subjects.
        z, stats = teacher("25", lab, train_lab)
        soft, pseudo = torch.softmax(z / T, -1), z.argmax(-1)
        u = torch.as_tensor(unlab_idx, device=device)
        results["teacher_25"]["pseudo_label_quality_unlabelled"] = metrics(y_np[unlab_idx], pseudo[u].cpu().numpy())
        y_mix = y.clone()
        y_mix[u] = pseudo[u]
        priors = transition_priors(y_np, runs_of(data, train_lab), data["subject"])
        write_json(folder / "transition_priors.json", {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                       for key, v in priors.items()})
        x, s_stats = student_x(x4, stats, device)
        model, res = train_gtad("gtad_cv", priors["gtad_cv"], x, y_mix, train_runs, val_runs, soft, ckpt / "student_25",
                                {**fold, "normalization": s_stats, "channels": STUDENT, "labelled": lab}, s_config, device)
        results["student_gtad_cv_25"] = res
        student("student_gtad_cv_25", model)
        del model, x, z, soft
        torch.cuda.empty_cache()
        np.savez_compressed(folder / "test_predictions.npz", **probs)
        write_json(folder / "test_metrics.json", {"fold": k, "test_subjects": fold["test"], "labelled": lab,
                                                  "models": results})
        for key in MODELS:
            s = results[key]["test"]
            print(f"TEST {name} {key:20s} acc={s['accuracy']:.4f} F1={s['macro_f1']:.4f} kappa={s['kappa']:.4f} "
                  + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in ["W", "N1", "N2", "N3", "REM"]), flush=True)
        summarize(out)
    write_json(out / "status.json", {"state": "requested_folds_complete", "folds": args.folds})


if __name__ == "__main__":
    main()
