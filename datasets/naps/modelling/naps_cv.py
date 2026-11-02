"""NAPS 10-fold patient CV with the iSLEEPS pipeline (run fold by fold with --folds).

Data: naps/preprocessed, recordings with C3-M2 and/or C4-M1, chin EMG and E1 (LOC) EOG: 133 of 138.
Teacher: SleepGPT-BiLSTM fine-tuned per fold on SleepGPT slots C3, C4, EMG, EOG. Each recording uses
the central EEGs it has (both, or only C3 or only C4, each in its own slot); an absent channel's
tokens are masked as attention keys and from pooling, which equals SleepGPT's actual_channels mode
for the channels present. 21-epoch context, all targets each epoch, CE loss, base lr 1e-4, layer
decay 0.75, AdamW, cosine with warm-up, max 25 epochs, patience 8 after 10, one seed (the first
iSLEEPS seed); best epoch by inner-validation macro F1. Same settings as datasets/isleeps/modelling/run_sleepgpt4_cv.py.
Students: context-aware SleepStudentNet (src/train_students_context.py) on EOG only,
EEG only (C4-M1, else C3-M2) and EEG + EOG, as CE only and/or TAD only (No KD) (--student-variants); best checkpoint by
inner-validation macro F1.
Folds: 10 patient-disjoint folds stratified by study type (diagnostic / CPAP / MAD / split / unknown).
Shared code (src/) is imported, never modified.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
import engine_multi  # noqa: E402
from engine_multi import predict as predict_teacher, train_member  # noqa: E402
from protocol import VARIANTS, metrics, normalization, sha256, temporal_indices, write_json  # noqa: E402
from protocol_isleeps import context_windows, make_folds  # noqa: E402
from sleepgpt_c4 import CHECKPOINT, PATCHES  # noqa: E402
from teachers_multi import SleepGPTMultiEncoder, SleepGPTMultiTeacher  # noqa: E402
from train_students_context import runs_of, stream_predict, train_variant  # noqa: E402

TEACHER_CHANNELS = ["C3-M2", "C4-M1", "EMG", "E1"]   # NaN where a recording lacks C3 or C4
SLOTS = ("C3", "C4", "EMG", "EOG")
STAT_CHANNELS = ["EEG", "EMG", "E1"]                 # student EEG = C4-M1, else C3-M2
STUDENT_INPUTS = {"eog": ["E1"], "c4": ["EEG"], "eeg_eog": ["EEG", "E1"]}
STUDENT_VARIANTS = ["ce_only", "no_kd"]
SEEDS = ["s1"]  # one teacher seed (seed offset 1, as the first iSLEEPS seed); no ensemble


def masked_attention(attn, x, keys):
    B, N, C = x.shape
    bias = torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
    q, k, v = F.linear(x, attn.qkv.weight, bias).reshape(B, N, 3, attn.heads, -1).permute(2, 0, 3, 1, 4)
    return attn.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=keys).transpose(1, 2).reshape(B, N, C))


class MaskedSleepGPTEncoder(SleepGPTMultiEncoder):
    """SleepGPTMultiEncoder with a per-epoch channel subset: a NaN channel is absent, its tokens are
    never attended to and are left out of the pooled embedding."""
    def forward(self, x):
        present = ~torch.isnan(x[:, :, 0])
        x = torch.nan_to_num(x, nan=0.0)
        M, C, _ = x.shape
        x = (x.float() - self.mean) / self.std
        t = self.proj(x).reshape(M, C, -1, PATCHES).permute(0, 1, 3, 2).reshape(M, C * PATCHES, -1)
        f = self.fft_proj(self.spectrogram(x).to(t.dtype)).squeeze(-1)
        f = f.reshape(M, C, -1, PATCHES).permute(0, 1, 3, 2).reshape(M, C * PATCHES, -1)
        pos = self.pos_embed.repeat(1, C, 1) + self.channel_embed.repeat_interleave(PATCHES, dim=1)
        cls = self.cls_token_pos_embed
        t = torch.cat([self.cls_token.expand(M, -1, -1) + cls, t + pos], 1) + self.token_type_embeddings.weight[0]
        f = torch.cat([self.fft_cls_token.expand(M, -1, -1) + cls, f + pos], 1) + self.token_type_embeddings.weight[1]
        h = torch.cat([t, f], dim=1)
        keep = present.repeat_interleave(PATCHES, dim=1)
        one = present.new_ones(M, 1)
        keys = torch.cat([one, keep, one, keep], 1)[:, None, None, :]   # time CLS, time, FFT CLS, FFT
        for b in self.blocks:
            h = h + b.drop_path(b.gamma_1 * masked_attention(b.attn, b.norm1(h), keys))
            if b.shared:
                h = h + b.drop_path(b.gamma_2 * b.mlp_tf(b.norm2_tf(h)))
            else:
                ht, hf = h[:, :1 + b.time_len], h[:, 1 + b.time_len:]
                ht = ht + b.drop_path(b.gamma_2 * b.mlp_time(b.norm2_time(ht)))
                hf = hf + b.drop_path(b.gamma_2 * b.mlp_fft(b.norm2_fft(hf)))
                h = torch.cat([ht, hf], dim=1)
        h = self.norm(h)
        n, w = PATCHES * C, keep[..., None].to(h.dtype)
        return torch.cat([(h[:, 1:1 + n] * w).sum(1) / w.sum(1), (h[:, 2 + n:] * w).sum(1) / w.sum(1)], dim=-1)


def build(spec, config, stats):
    t = config["teacher"]
    model = SleepGPTMultiTeacher(t["window"], t["target"], slots=SLOTS, drop_path_rate=spec["drop_path_rate"],
                                 pretrained=True, lstm_hidden=t["lstm_hidden"], lstm_layers=t["lstm_layers"],
                                 dropout=t["dropout"])
    model.encoder.__class__ = MaskedSleepGPTEncoder
    return model


engine_multi.build = build  # train_member builds the masked 4-slot teacher


def student_channel(x4, name):
    if name == "EEG":
        return torch.where(torch.isnan(x4[:, 1]), x4[:, 0], x4[:, 1])
    return x4[:, TEACHER_CHANNELS.index(name)]


def load(data_dir):
    with (data_dir / "records.csv").open() as stream:
        recs = {r["subject"]: r for r in csv.DictReader(stream)}
    eligible = []
    for subject in sorted(recs, key=lambda s: int(s.split("_")[1])):
        with np.load(data_dir / f"{subject}.npz", allow_pickle=False) as d:
            have = dict(zip(d["channels"].tolist(), d["channel_present"].tolist()))
        if (have["C3-M2"] or have["C4-M1"]) and have["EMG"] and have["E1"]:
            eligible.append(subject)
    total = sum(int(recs[s]["retained_epochs"]) for s in eligible)
    x_all = np.full((total, len(TEACHER_CHANNELS), 3000), np.nan, dtype=np.float32)  # filled in place (~6 GB)
    ys, subjects, epochs, segments, records, eeg_source = [], [], [], [], [], {}
    offset = 0
    for subject in eligible:
        with np.load(data_dir / f"{subject}.npz", allow_pickle=False) as d:
            names, present = d["channels"].tolist(), d["channel_present"]
            cols = [i for i, c in enumerate(TEACHER_CHANNELS) if present[names.index(c)]]
            y = d["y"].astype(np.int64)
            x = x_all[offset:offset + len(y)]
            x[:, cols] = d["x"][:, [names.index(TEACHER_CHANNELS[i]) for i in cols]]
            assert np.isfinite(x[:, cols]).all() and d["fs"].item() == 100
            eeg = "C4-M1" if 1 in cols else "C3-M2"
            s = np.stack([x[:, TEACHER_CHANNELS.index(eeg)], x[:, 2], x[:, 3]], 1)  # student statistics
            eeg_source[subject] = {"teacher": [TEACHER_CHANNELS[i] for i in cols if i < 2], "student": eeg}
            ys.append(y); subjects.append(np.full(len(y), subject))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": subject, "subject": subject, "start": offset, "stop": offset + len(y),
                            "sha256": sha256(data_dir / f"{subject}.npz"), "study_type": recs[subject]["study_type"] or "unknown",
                            "alignment_flag": recs[subject].get("alignment_flag", ""),
                            "sum": s.sum(axis=(0, -1), dtype=np.float64).tolist(),
                            "sum_squares": np.square(s.astype(np.float64)).sum(axis=(0, -1)).tolist(),
                            "sample_count": int(len(y) * 3000)})
            offset += len(y)
    assert offset == total
    data = {k: np.concatenate(v) for k, v in zip(["y", "subject", "epoch_index", "segment_id"],
                                                 [ys, subjects, epochs, segments])}
    data["x"], data["recording"] = x_all, data["subject"]
    data["records"], data["eeg_source"] = records, eeg_source
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["naps"])
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "naps" / "naps_cv10")
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    parser.add_argument("--teacher-epochs", type=int, default=25)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--student-variants", nargs="+", default=STUDENT_VARIANTS, choices=["ce_only", "no_kd", "full_tad"])
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)
    data = load(args.data_dir)
    strata = {r["subject"]: r["study_type"] for r in data["records"]}
    folds = make_folds(np.unique(data["subject"]), strata, 10, seed=42)
    teacher_eeg = [len(v["teacher"]) == 2 for v in data["eeg_source"].values()]
    print(f"{len(strata)} recordings, {len(data['y'])} epochs; teacher C3+C4 in {sum(teacher_eeg)}, one central "
          f"EEG in {len(teacher_eeg) - sum(teacher_eeg)}; student EEG from C3 in "
          f"{sum(v['student'] == 'C3-M2' for v in data['eeg_source'].values())}", flush=True)
    member = {"name": "sleepgpt4m", "arch": "sleepgpt", "epochs": args.teacher_epochs, "lr": 1e-4,
              "weight_decay": 0.05, "drop_path_rate": 0.2, "chunk": 64, "chunks_per_batch": 2}
    config = {
        "dataset": "NSRR NAPS", "n_folds": 10, "fold_stratification": "study type", "validation_fraction": 0.1,
        "teacher": {"architecture": "SleepGPT-BiLSTM", "member": member, "channels": TEACHER_CHANNELS,
                    "sleepgpt_slots": list(SLOTS),
                    "eeg": "C3-M2 and C4-M1 where both exist, else the one present; absent slot masked "
                           "(attention keys and pooling)",
                    "checkpoint_sha256": sha256(CHECKPOINT), "loss": "ce", "seeds": SEEDS,
                    "teacher_selection": "single seed (s1); best epoch by inner-validation macro F1",
                    "window": 21, "target": 10, "lstm_hidden": 100, "lstm_layers": 3, "dropout": 0.1,
                    "layer_decay": 0.75, "warmup_fraction": 0.1},
        "teacher_epochs": args.teacher_epochs, "patience": 8, "min_epochs": args.min_epochs, "seed": 42, "amp": True,
        "temperature": 2.0, "variants": VARIANTS, "student_epochs": args.student_epochs, "student_batch": 256, "teacher_batch": 128,
        "seq_len": 20, "seqs_per_batch": 32, "student_inputs": STUDENT_INPUTS,
        "student_eeg": "C4-M1, or C3-M2 when C4 is absent", "student_variants": args.student_variants,
        "student": "SleepStudentNet (unchanged), stateful causal (context-aware)",
        "data_sha256": {r["subject"]: r["sha256"] for r in data["records"]},
        "pipeline_sha256": {n: sha256(p) for n, p in [("naps_cv.py", Path(__file__)),
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
        write_json(out / "eeg_source.json", data["eeg_source"])
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
        train, pairs, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        assert len(train) + len(val) + len(test) == len(y)
        stats = normalization(data, fold["train"])
        metadata = {**fold, "normalization": stats, "normalization_channels": STAT_CHANNELS, "channels": TEACHER_CHANNELS,
                    "n_train": len(train), "n_val": len(val), "n_test": len(test)}
        write_json(folder / "split.json", metadata)
        print(f"\n{name}: patients {len(fold['train'])}/{len(fold['val'])}/{len(fold['test'])}, "
              f"epochs {len(train)}/{len(val)}/{len(test)}", flush=True)
        status = out / "status.json"
        probs, softs, seed_results = {}, {}, {}
        for j, s in enumerate(SEEDS, 1):
            model, res = train_member({**member, "name": f"sleepgpt4m_{s}", "seed_offset": j}, "ce", x4, y, train, val,
                                      windows, ckpt / f"sleepgpt4m_{s}_ce", metadata, config, stats, status)
            z = predict_teacher(model, x4, windows, all_idx, True)
            probs[s], softs[s], seed_results[s] = torch.softmax(z, -1), torch.softmax(z / T, -1), res
            del model
            torch.cuda.empty_cache()
        if len(SEEDS) > 1:
            probs = {"ensemble": torch.stack(list(probs.values())).mean(0), **probs}
            softs = {"ensemble": torch.stack(list(softs.values())).mean(0), **softs}
        vt = torch.as_tensor(val, device=device)
        val_scores = {c: metrics(data["y"][val], p[vt].argmax(-1).cpu().numpy()) for c, p in probs.items()}
        choice = max(val_scores, key=lambda c: val_scores[c]["macro_f1"])
        soft = softs[choice].detach()
        write_json(ckpt / "teacher_selection.json", {"choice": choice, "seeds": seed_results,
                   "validation_macro_f1": {c: v["macro_f1"] for c, v in val_scores.items()}})
        tt = torch.as_tensor(test, device=device)
        test_y = data["y"][test]
        predictions = {"y_true": test_y, "subject": data["subject"][test], "epoch_index": data["epoch_index"][test]}
        results = {}
        for c, p in probs.items():
            pc = p[tt].cpu().numpy()
            predictions[f"teacher_{c}_prob"] = pc
            results[f"teacher_{c}"] = {"test": metrics(test_y, pc.argmax(-1)), "validation": val_scores[c]}
        results["teacher"] = {**results[f"teacher_{choice}"], "choice": choice}
        print(f"{name} teacher choice={choice} val " + ", ".join(f"{c}:{v['macro_f1']:.4f}" for c, v in val_scores.items()), flush=True)
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
        lines = [f"# NAPS {name} test results ({len(fold['test'])} patients, {len(test)} epochs)", "",
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
