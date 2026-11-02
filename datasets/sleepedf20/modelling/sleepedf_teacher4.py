"""Sleep-EDF-20 LOSO, 4-channel teacher only (100% labels), same folds and recipe as sleepedf_cv.py.

Teacher slots C3, C4, EMG, EOG filled with all four Sleep-EDF channels instead of masking two:
  C3 <- EEG Pz-Oz, C4 <- EEG Fpz-Cz, EMG <- EMG submental, EOG <- EOG horizontal.
The Sleep-EDF SC chin EMG is a 1-Hz envelope (high-pass 16 Hz, rectified, low-pass 0.7 Hz), not raw EMG; it is
linearly interpolated to 100 Hz. Fpz-Cz and EOG come from preprocessed_sleepedf20 (asserted equal to the raw EDF);
Pz-Oz and EMG are read from the raw EDF at the same retained epochs (epoch_index).
Amplitude: per-recording normalization of every channel as in sleepedf_cv.py (median 30-s epoch SD -> 15 uV).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyedflib
import torch

import sleepedf_cv as base   # noqa: E402  (imports naps_cv: masked 4-slot SleepGPT teacher)
from naps_cv import SLOTS, STAT_CHANNELS, TEACHER_CHANNELS  # noqa: E402
from engine_multi import predict as predict_teacher, train_member  # noqa: E402
from protocol import metrics, normalization, sha256, temporal_indices, write_json  # noqa: E402
from protocol_isleeps import context_windows  # noqa: E402
from sleepgpt_c4 import CHECKPOINT  # noqa: E402
import repo_paths  # noqa: E402

HERE = base.HERE
SOURCE = {"C3-M2": "EEG Pz-Oz", "C4-M1": "EEG Fpz-Cz", "EMG": "EMG submental", "E1": "EOG horizontal"}
RAW = ("EEG Pz-Oz", "EMG submental")
MODEL = "teacher_full_4ch"


def raw_channels(psg_path, epoch_index):
    """Pz-Oz (100 Hz) and EMG envelope (1 Hz -> 100 Hz) at the retained epochs, plus Fpz-Cz/EOG for the check."""
    out = {}
    with pyedflib.EdfReader(str(psg_path)) as f:
        labels = [s.strip() for s in f.getSignalLabels()]
        for name in ("EEG Fpz-Cz", "EOG horizontal") + RAW:
            i = labels.index(name)
            fs = f.getSampleFrequency(i)
            v = f.readSignal(i)
            if fs != 100:
                assert name == "EMG submental" and fs == 1
                v = np.interp(np.arange(len(v) * 100) / 100, np.arange(len(v)), v)
            out[name] = v.reshape(-1, 3000)[epoch_index].astype(np.float32)
    return out


def load(data_dir, raw_dir):
    with (data_dir / "records.csv").open() as stream:
        recs = sorted(csv.DictReader(stream), key=lambda r: r["recording"])
    total = sum(int(r["retained_epochs"]) for r in recs)
    x_all = np.full((total, len(TEACHER_CHANNELS), 3000), np.nan, dtype=np.float32)
    ys, subj, rec_ids, epochs, segments, records, scaling = [], [], [], [], [], [], {}
    offset = 0
    for r in recs:
        with np.load(data_dir / r["file"], allow_pickle=False) as d:
            names = d["channels"].tolist()
            y = d["y"].astype(np.int64)
            raw = raw_channels(raw_dir / str(d["source_psg"]), d["epoch_index"])
            for name in names:   # preprocessed channels equal the raw EDF at the same epochs
                assert np.array_equal(raw[name], d["x"][:, names.index(name)]), (r["recording"], name)
            x = x_all[offset:offset + len(y)]
            scaling[r["recording"]] = {}
            for slot, src in SOURCE.items():
                v = raw[src]
                assert np.isfinite(v).all()
                mean, med_sd = float(v.mean()), float(np.median(v.std(-1)))
                assert med_sd > 0, (r["recording"], src)
                x[:, TEACHER_CHANNELS.index(slot)] = (v - mean) * (base.REF_SD_UV / med_sd)
                scaling[r["recording"]][src] = {"mean_uV": mean, "median_epoch_sd_uV": med_sd,
                                                "gain": base.REF_SD_UV / med_sd}
            s = x[:, [TEACHER_CHANNELS.index(c) for c in ("C4-M1", "EMG", "E1")]].astype(np.float64)
            ys.append(y); subj.append(np.full(len(y), r["subject"])); rec_ids.append(np.full(len(y), r["recording"]))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": r["recording"], "subject": r["subject"], "start": offset, "stop": offset + len(y),
                            "sha256": r["output_sha256"], "sum": s.sum((0, 2)).tolist(),
                            "sum_squares": np.square(s).sum((0, 2)).tolist(), "sample_count": int(len(y) * 3000)})
            offset += len(y)
        print(f"loaded {r['recording']}", flush=True)
    assert offset == total
    data = {k: np.concatenate(v) for k, v in zip(["y", "subject", "recording", "epoch_index", "segment_id"],
                                                 [ys, subj, rec_ids, epochs, segments])}
    data["x"], data["records"], data["scaling"] = x_all, records, scaling
    return data


def summarize(out):
    rows, cm = [], 0
    for path in sorted((out / "folds").glob("fold_*/test_metrics.json")):
        v = json.loads(path.read_text())["models"][MODEL]["test"]
        rows.append([v[k] for k in base.KEYS])
        cm = cm + np.array(v["confusion_matrix"])
    v = np.array(rows)
    y = np.repeat(np.arange(5), cm.sum(1))
    pred = np.concatenate([np.repeat(np.arange(5), cm[i]) for i in range(5)])
    pooled = {k: m for k, m in metrics(y, pred).items() if k in base.KEYS}
    summary = {MODEL: {"folds": len(v), **{f"{k}_mean": float(v[:, i].mean()) for i, k in enumerate(base.KEYS)},
                       **{f"{k}_sd": float(v[:, i].std(ddof=1)) if len(v) > 1 else 0.0 for i, k in enumerate(base.KEYS)},
                       "pooled": pooled}}
    write_json(out / "summary.json", summary)
    s = summary[MODEL]
    (out / "SUMMARY.md").write_text(
        "# Sleep-EDF-20 LOSO, 4-channel teacher: test, mean over held-out subjects ± SD (pooled)\n\n"
        "| Model | Folds | Accuracy | Macro F1 | Kappa | W | N1 | N2 | N3 | REM |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {MODEL} | {s['folds']} | " + " | ".join(f"{s[k + '_mean']:.3f} ± {s[k + '_sd']:.3f} ({pooled[k]:.3f})"
                                                 for k in base.KEYS[:3])
        + " | " + " | ".join(f"{pooled[k]:.3f}" for k in base.KEYS[3:]) + " |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["sleepedf20"])
    parser.add_argument("--raw-dir", type=Path, default=repo_paths.RAW["sleepedf20"])
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "sleepedf20" / "sleepedf20_loso_teacher4")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--check", action="store_true", help="load data, print scaling, no training")
    args = parser.parse_args()
    data = load(args.data_dir, args.raw_dir)
    subjects = sorted(str(s) for s in set(data["subject"]))
    folds = base.loso_folds(subjects)
    print(f"{len(data['records'])} recordings, {len(subjects)} subjects, {len(data['y'])} epochs", flush=True)
    for src in SOURCE.values():
        g = [v[src]["gain"] for v in data["scaling"].values()]
        print(f"{src}: gain median {np.median(g):.3g} (range {min(g):.3g}-{max(g):.3g})", flush=True)
    if args.check:
        return
    previous = json.loads((repo_paths.RESULTS / "sleepedf20" / "sleepedf20_loso" / "folds.json").read_text())
    assert previous == folds, "folds differ from sleepedf20_loso"
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)
    member = {"name": "sleepgpt4m", "arch": "sleepgpt", "epochs": 25, "lr": 1e-4,
              "weight_decay": 0.05, "drop_path_rate": 0.2, "chunk": 64, "chunks_per_batch": 2}
    config = {
        "dataset": "Sleep-EDF-20 (SC)", "protocol": "LOSO, 2 inner-validation subjects (seed 42 + fold), as sleepedf20_loso",
        "amplitude_normalization": {"method": "per recording and channel: subtract mean, scale median 30-s epoch SD "
                                              "to a fixed reference; no labels", "reference_sd_uV": base.REF_SD_UV},
        "teacher": {"architecture": "SleepGPT-BiLSTM", "member": member, "channels": TEACHER_CHANNELS,
                    "sleepedf_channels": SOURCE, "emg": "1-Hz envelope linearly interpolated to 100 Hz",
                    "sleepgpt_slots": list(SLOTS), "checkpoint_sha256": sha256(CHECKPOINT), "loss": "ce",
                    "window": 21, "target": 10, "lstm_hidden": 100, "lstm_layers": 3, "dropout": 0.1,
                    "layer_decay": 0.75, "warmup_fraction": 0.1},
        "patience": 8, "min_epochs": 10, "seed": 42, "amp": True, "teacher_batch": 128,
        "data_sha256": {r["recording"]: r["sha256"] for r in data["records"]},
        "pipeline_sha256": {n: sha256(p) for n, p in [("sleepedf_teacher4.py", Path(__file__)),
                            ("sleepedf_cv.py", HERE / "sleepedf_cv.py"),
                            ("naps_cv.py", repo_paths.find("naps_cv.py")),
                            ("engine_multi.py", repo_paths.find("engine_multi.py")),
                            ("teachers_multi.py", repo_paths.find("teachers_multi.py"))]},
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
        print(f"\n{name}: test {fold['test']}, val {fold['val']}, train {len(fold['train'])} subjects "
              f"({len(train)} epochs)", flush=True)
        stats = normalization(data, fold["train"])
        meta = {**fold, "normalization": stats, "normalization_channels": STAT_CHANNELS, "labelled": fold["train"]}
        model, res = train_member({**member, "name": "sleepgpt4m_full4ch", "seed_offset": 1}, "ce", x4, y, train, val,
                                  windows, ckpt / "teacher_full_4ch", meta, config, stats, out / "status.json")
        z = predict_teacher(model, x4, windows, all_idx, True)
        del model
        torch.cuda.empty_cache()
        test_y = y_np[test]
        pt = torch.softmax(z[torch.as_tensor(test, device=device)], -1).cpu().numpy()
        res = {**res, "test": metrics(test_y, pt.argmax(-1))}
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "test_predictions.npz", y_true=test_y, recording=data["recording"][test],
                            epoch_index=data["epoch_index"][test], **{f"{MODEL}_prob": pt})
        write_json(folder / "test_metrics.json", {"fold": k, "test_subjects": fold["test"], "models": {MODEL: res}})
        s = res["test"]
        print(f"TEST {name} {MODEL} acc={s['accuracy']:.4f} F1={s['macro_f1']:.4f} kappa={s['kappa']:.4f} "
              + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in ["W", "N1", "N2", "N3", "REM"]), flush=True)
        summarize(out)
    write_json(out / "status.json", {"state": "requested_folds_complete", "folds": args.folds})


if __name__ == "__main__":
    main()
