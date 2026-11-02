"""ISRUC-Sleep Cohort I 10-fold patient CV with the NAPS/iSLEEPS pipeline (run fold by fold with --folds).

Data: preprocessed ISRUC-S1, 99 of 100 recordings (recording 40, the monopolar one, is excluded), scorer-1
labels. Teacher channels C3-M2, C4-M1, chin EMG, E1-M2 (all present in every kept recording).
Amplitude: per-recording normalization. Each recording and channel is centred on its own mean and scaled
so that its median 30-s epoch SD equals a fixed reference SD (REF_SD_UV, µV); labels are never used.
This removes the ~10x gain difference of the A1/A2 recordings and keeps the µV scale the SleepGPT
teacher's fixed input normalization expects.
Teacher: masked SleepGPT-BiLSTM (naps/naps_cv.py, slots C3, C4, EMG, EOG), 21-epoch context, CE, base lr
1e-4, layer decay 0.75, AdamW, cosine with warm-up, max 25 epochs, patience 8 after 10, one seed; best
epoch by inner-validation macro F1.
Students: context-aware SleepStudentNet (src/train_students_context.py) on EOG (E1-M2),
EEG (C4-M1) and EEG + EOG, CE only; best checkpoint by inner-validation macro F1.
Folds: 10 patient-disjoint folds stratified by montage (M1/M2, A1/A2), inner validation 10%.
Shared code (src/, naps_cv.py) is imported, never modified.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
import naps_cv  # noqa: E402  (sets engine_multi.build to the masked 4-slot SleepGPT teacher; adds the import paths)
from naps_cv import SLOTS, STAT_CHANNELS, STUDENT_INPUTS, TEACHER_CHANNELS, student_channel  # noqa: E402
from engine_multi import predict as predict_teacher, train_member  # noqa: E402
from protocol import VARIANTS, metrics, normalization, sha256, temporal_indices, write_json  # noqa: E402
from protocol_isleeps import context_windows, make_folds  # noqa: E402
from sleepgpt_c4 import CHECKPOINT  # noqa: E402
from train_students_context import runs_of, stream_predict, train_variant  # noqa: E402

SOURCE = {"C3-M2": "C3-M2", "C4-M1": "C4-M1", "EMG": "EMG", "E1": "E1-M2"}  # teacher slot name -> ISRUC channel
REF_SD_UV = {"C3-M2": 15.0, "C4-M1": 15.0, "EMG": 3.0, "E1": 15.0}       # fixed reference median epoch SD
EXCLUDED = {"ISRUC1_040": "monopolar recording (derivations computed); excluded as in prior ISRUC-S1 work"}
SEEDS = ["s1"]


def load(data_dir):
    with (data_dir / "records.csv").open() as stream:
        recs = {r["subject"]: r for r in csv.DictReader(stream)}
    subjects = sorted(s for s in recs if s not in EXCLUDED)
    total = sum(int(recs[s]["retained_epochs"]) for s in subjects)
    x_all = np.empty((total, len(TEACHER_CHANNELS), 3000), dtype=np.float32)
    ys, subj, epochs, segments, records, scaling = [], [], [], [], [], {}
    offset = 0
    for subject in subjects:
        path = data_dir / f"{subject}.npz"
        with np.load(path, allow_pickle=False) as d:
            names, present = d["channels"].tolist(), d["channel_present"]
            cols = [names.index(SOURCE[c]) for c in TEACHER_CHANNELS]
            assert present[cols].all() and d["fs"].item() == 100 and str(d["subject"]) == subject
            y = d["y"].astype(np.int64)
            x = x_all[offset:offset + len(y)]
            x[:] = d["x"][:, cols]
            assert np.isfinite(x).all()
            scaling[subject] = {}
            for i, c in enumerate(TEACHER_CHANNELS):   # per-recording normalization, no labels
                mean, med_sd = float(x[:, i].mean()), float(np.median(x[:, i].std(-1)))
                assert med_sd > 0, (subject, c)
                x[:, i] = (x[:, i] - mean) * (REF_SD_UV[c] / med_sd)
                scaling[subject][c] = {"mean_uV": mean, "median_epoch_sd_uV": med_sd, "gain": REF_SD_UV[c] / med_sd}
            s = x[:, [1, 2, 3]]   # student statistics: EEG (C4-M1), EMG, E1
            ys.append(y); subj.append(np.full(len(y), subject))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": subject, "subject": subject, "start": offset, "stop": offset + len(y),
                            "sha256": recs[subject]["output_sha256"], "montage": recs[subject]["montage"],
                            "sum": s.sum(axis=(0, -1), dtype=np.float64).tolist(),
                            "sum_squares": np.square(s.astype(np.float64)).sum(axis=(0, -1)).tolist(),
                            "sample_count": int(len(y) * 3000)})
            offset += len(y)
    assert offset == total
    data = {k: np.concatenate(v) for k, v in zip(["y", "subject", "epoch_index", "segment_id"],
                                                 [ys, subj, epochs, segments])}
    data["x"], data["recording"], data["records"], data["scaling"] = x_all, data["subject"], records, scaling
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["isruc"])
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "isruc" / "isruc_cv10_ce")
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    parser.add_argument("--teacher-epochs", type=int, default=25)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--student-variants", nargs="+", default=["ce_only"], choices=["ce_only", "no_kd", "full_tad"])
    parser.add_argument("--check", action="store_true", help="load data, print folds and scaling, no training")
    args = parser.parse_args()
    data = load(args.data_dir)
    strata = {r["subject"]: r["montage"] for r in data["records"]}
    folds = make_folds(np.unique(data["subject"]), strata, 10, seed=42)
    print(f"{len(strata)} recordings, {len(data['y'])} epochs; montage "
          f"{ {m: sum(v == m for v in strata.values()) for m in sorted(set(strata.values()))} }", flush=True)
    if args.check:
        for c in TEACHER_CHANNELS:
            g = {m: np.median([data["scaling"][s][c]["gain"] for s in strata if strata[s] == m]) for m in set(strata.values())}
            after = [np.median(data["x"][r["start"]:r["stop"], TEACHER_CHANNELS.index(c)].std(-1)) for r in data["records"]]
            print(f"{c}: median gain by montage {g}; median epoch SD after scaling {np.min(after):.2f}-{np.max(after):.2f} uV")
        for f in folds:
            print(f"fold {f['fold']}: train {len(f['train'])} val {len(f['val'])} test {len(f['test'])} "
                  f"(A1/A2 in test: {sum(strata[s] != 'M1/M2' for s in f['test'])})")
        return
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)
    member = {"name": "sleepgpt4m", "arch": "sleepgpt", "epochs": args.teacher_epochs, "lr": 1e-4,
              "weight_decay": 0.05, "drop_path_rate": 0.2, "chunk": 64, "chunks_per_batch": 2}
    config = {
        "dataset": "ISRUC-Sleep Cohort I", "labels": "scorer 1", "excluded": EXCLUDED, "n_folds": 10,
        "fold_stratification": "montage", "validation_fraction": 0.1,
        "amplitude_normalization": {"method": "per recording and channel: subtract mean, scale median 30-s epoch SD "
                                              "to a fixed reference; no labels", "reference_sd_uV": REF_SD_UV},
        "teacher": {"architecture": "SleepGPT-BiLSTM", "member": member, "channels": TEACHER_CHANNELS,
                    "isruc_channels": SOURCE, "sleepgpt_slots": list(SLOTS),
                    "checkpoint_sha256": sha256(CHECKPOINT), "loss": "ce", "seeds": SEEDS,
                    "teacher_selection": "single seed (s1); best epoch by inner-validation macro F1",
                    "window": 21, "target": 10, "lstm_hidden": 100, "lstm_layers": 3, "dropout": 0.1,
                    "layer_decay": 0.75, "warmup_fraction": 0.1},
        "teacher_epochs": args.teacher_epochs, "patience": 8, "min_epochs": args.min_epochs, "seed": 42, "amp": True,
        "temperature": 2.0, "variants": VARIANTS, "student_epochs": args.student_epochs, "student_batch": 256, "teacher_batch": 128,
        "seq_len": 20, "seqs_per_batch": 32, "student_inputs": STUDENT_INPUTS,
        "student_eeg": "C4-M1", "student_eog": "E1-M2", "student_variants": args.student_variants,
        "student": "SleepStudentNet (unchanged), stateful causal (context-aware)",
        "data_sha256": {r["subject"]: r["sha256"] for r in data["records"]},
        "pipeline_sha256": {n: sha256(p) for n, p in [("isruc_cv.py", Path(__file__)),
                            ("naps_cv.py", repo_paths.find("naps_cv.py")),
                            ("engine_multi.py", repo_paths.find("engine_multi.py")),
                            ("teachers_multi.py", repo_paths.find("teachers_multi.py")),
                            ("train_students_context.py", repo_paths.find("train_students_context.py"))]},
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
    y = torch.from_numpy(data["y"]).to(device)
    all_idx = np.arange(len(data["y"]))
    T = config["temperature"]
    for k in args.folds:
        fold, name = folds[k - 1], f"fold_{k:02d}"
        folder, ckpt = out / "folds" / name, out / "checkpoints" / name
        if (folder / "test_metrics.json").exists():
            print(f"Already complete: {name}"); continue
        train, _, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        assert len(train) + len(val) + len(test) == len(y)
        stats = normalization(data, fold["train"])
        metadata = {**fold, "normalization": stats, "normalization_channels": STAT_CHANNELS, "channels": TEACHER_CHANNELS,
                    "n_train": len(train), "n_val": len(val), "n_test": len(test)}
        write_json(folder / "split.json", metadata)
        print(f"\n{name}: patients {len(fold['train'])}/{len(fold['val'])}/{len(fold['test'])}, "
              f"epochs {len(train)}/{len(val)}/{len(test)}", flush=True)
        model, res = train_member({**member, "name": "sleepgpt4m_s1", "seed_offset": 1}, "ce", x4, y, train, val,
                                  windows, ckpt / "sleepgpt4m_s1_ce", metadata, config, stats, out / "status.json")
        z = predict_teacher(model, x4, windows, all_idx, True)
        prob, soft = torch.softmax(z, -1), torch.softmax(z / T, -1).detach()
        del model
        torch.cuda.empty_cache()
        vt, tt = torch.as_tensor(val, device=device), torch.as_tensor(test, device=device)
        test_y = data["y"][test]
        pt = prob[tt].cpu().numpy()
        predictions = {"y_true": test_y, "subject": data["subject"][test], "epoch_index": data["epoch_index"][test],
                       "teacher_prob": pt}
        results = {"teacher": {**res, "test": metrics(test_y, pt.argmax(-1)),
                               "validation": metrics(data["y"][val], prob[vt].argmax(-1).cpu().numpy())}}
        train_runs, val_runs, test_runs = runs_of(data, train), runs_of(data, val), runs_of(data, test)
        for inp, chans in STUDENT_INPUTS.items():
            sel = [STAT_CHANNELS.index(c) for c in chans]
            mean = torch.tensor([stats["mean_uV"][i] for i in sel], device=device).view(1, -1, 1)
            std = torch.tensor([stats["std_uV"][i] for i in sel], device=device).view(1, -1, 1)
            x = (torch.stack([student_channel(x4, c) for c in chans], 1) - mean) / std
            assert torch.isfinite(x).all()
            if len(sel) == 1:
                x = x[:, 0].contiguous()
            s_config = {**config, "channels": chans, "input": inp}
            s_meta = {**fold, "normalization": {"mean_uV": [stats["mean_uV"][i] for i in sel],
                                                 "std_uV": [stats["std_uV"][i] for i in sel]}, "channels": chans}
            for kind in args.student_variants:
                model, res = train_variant(kind, x, y, train_runs, val_runs, soft, ckpt / f"students_{inp}", s_meta, s_config, device)
                pr = torch.softmax(stream_predict(model, x, test_runs, device)[tt], -1).cpu().numpy()
                predictions[f"{inp}_{kind}_prob"] = pr
                results[f"{inp}_{kind}"] = {**res, "test": metrics(test_y, pr.argmax(-1))}
                del model
            del x
            torch.cuda.empty_cache()
        np.savez_compressed(folder / "test_predictions.npz", **predictions)
        write_json(folder / "test_metrics.json", {"fold": k, "test_subjects": fold["test"], "models": results})
        lines = [f"# ISRUC-1 {name} test results ({len(fold['test'])} patients, {len(test)} epochs)", "",
                 "| Model | Val macro F1 | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, v in results.items():
            s, vf = v["test"], v["validation"]["macro_f1"]
            lines.append(f"| {key} | {vf:.3f} | " + " | ".join(f"{s[m]:.3f}" for m in
                         ["accuracy", "macro_f1", "kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]) + " |")
        (folder / "RESULTS.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    write_json(out / "status.json", {"state": "requested_folds_complete", "folds": args.folds, "time": time.time()})


if __name__ == "__main__":
    main()
