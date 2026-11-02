"""Context-aware (stateful, causal) SleepStudentNet distilled from the 4-channel SleepGPT teacher.

The student modules are unchanged (CNN -> within-epoch GRU(64) -> linear). Context comes from
carrying the GRU state from epoch t to epoch t+1, in training and inference:
- Training: sequences of `--seq-len` consecutive epochs inside continuous runs (random start offset
  each training epoch, so every epoch is used), state carried and back-propagated through the
  sequence. Per epoch: alpha*CE + beta*KD(teacher_t) + gamma*TAD, where TAD for t >= 1 is
  conf(teacher_{t-1}) * KL(teacher_t || student_t) with student_t computed from the carried state
  (temperature 2, softmax once), i.e. the repository's transition term on the real recurrent path.
- Validation/test: streaming inference through each continuous run (past epochs only), state reset at
  the start of each run. Checkpoints selected on inner-validation macro F1.
Also re-scores the existing per-epoch students (results/sleepgpt4_cv10 and *_students_eeg_eog) with
streaming inference, without retraining. Folds, teacher (recomputed from its checkpoints), student
optimizer and schedule (Adam 1e-3, wd 1e-4, cosine over 40 epochs, patience 8 after 10) as before.
"""
import repo_paths
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
from engine import atomic_save, predict as predict_student, train_model as train_student  # noqa: E402
from models import SleepStudentNet  # noqa: E402
from protocol_isleeps import VARIANTS, context_windows, load_data, metrics, normalization, temporal_indices, write_json  # noqa: E402

from teachers_multi import SleepGPTMultiTeacher  # noqa: E402
from train_students_eeg_eog import teacher_logits  # noqa: E402

TEACHER_CHANNELS = ["C3-M2", "C4-M1", "EMG", "E1-M2"]
INPUTS = {"c4": ["C4-M1"], "eeg_eog": ["C4-M1", "E1-M2"], "eog": ["E1-M2"]}


def runs_of(data, indices):
    """Split sorted global indices into continuous runs (same recording, consecutive epoch_index)."""
    idx = np.sort(np.asarray(indices))
    breaks = np.flatnonzero((np.diff(data["epoch_index"][idx]) != 1) |
                            (data["recording"][idx][1:] != data["recording"][idx][:-1])) + 1
    return np.split(idx, breaks)


def step(model, x_t, h):
    """One epoch through the unchanged student modules; returns logits and the carried state."""
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(1)
    feat = model.dropout(model.cnn(x_t).permute(0, 2, 1))
    out, h = model.gru(feat, h)
    return model.fc(out[:, -1, :]), h


@torch.inference_mode()
def stream_predict(model, x, runs, device):
    """Causal streaming logits for every epoch in `runs` (state reset at each run start)."""
    model.eval()
    logits = torch.zeros((len(x), 5), device=device)
    runs = sorted(runs, key=len, reverse=True)
    lengths = torch.tensor([len(r) for r in runs], device=device)
    table = torch.full((len(runs), int(lengths[0])), -1, dtype=torch.long, device=device)
    for i, r in enumerate(runs):
        table[i, :len(r)] = torch.as_tensor(r, device=device)
    h = torch.zeros((1, len(runs), model.gru.hidden_size), device=device)
    for t in range(table.shape[1]):
        active = int((lengths > t).sum())  # runs sorted by length: the active ones are a prefix
        idx = table[:active, t]
        with torch.autocast("cuda", dtype=torch.float16):
            out, h_new = step(model, x[idx], h[:, :active].contiguous())
        logits[idx] = out.float()
        h = torch.cat([h_new.float(), h[:, active:]], dim=1)
    return logits


def train_sequences(train_runs, seq_len, rng):
    """Non-overlapping full-length sequences with a random offset per run."""
    seqs = []
    for r in train_runs:
        if len(r) < seq_len:
            continue
        off = int(rng.integers(0, seq_len))
        off = min(off, len(r) - seq_len)
        for s in range(off, len(r) - seq_len + 1, seq_len):
            seqs.append(r[s:s + seq_len])
    return np.stack(seqs)


def train_variant(kind, x, y, train_runs, val_runs, soft, folder, metadata, config, device):
    folder.mkdir(parents=True, exist_ok=True)
    best_path, done_path, history_path = folder / f"{kind}.pt", folder / f"{kind}_complete.json", folder / f"{kind}_history.json"
    w = VARIANTS[kind]
    seed = config["seed"] + metadata["fold"] * 1000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SleepStudentNet(n_channels=len(config["channels"])).to(device)
    if done_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"])
        return model, json.loads(done_path.read_text())
    epochs, T = config["student_epochs"], config["temperature"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda")
    val = np.concatenate(val_runs)
    y_val = y[torch.as_tensor(val, device=device)].cpu().numpy()
    best_f1, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        if stale >= config["patience"] and epoch > config["min_epochs"]:
            break
        started = time.monotonic()
        model.train()
        rng = np.random.default_rng(seed + epoch)
        seqs = train_sequences(train_runs, config["seq_len"], rng)
        seqs = torch.as_tensor(seqs[rng.permutation(len(seqs))], device=device)
        sums, seen = torch.zeros(4, device=device), 0
        for b in range(0, len(seqs), config["seqs_per_batch"]):
            batch = seqs[b:b + config["seqs_per_batch"]]                      # (S, L)
            optimizer.zero_grad(set_to_none=True)
            h = None
            ce = kd = tad = torch.zeros((), device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                for t in range(batch.shape[1]):
                    idx = batch[:, t]
                    logits, h = step(model, x[idx], h)
                    logp = F.log_softmax(logits.float() / T, -1)
                    ce = ce + F.cross_entropy(logits.float(), y[idx])
                    if w["beta"]:
                        kd = kd + F.kl_div(logp, soft[idx], reduction="batchmean") * T ** 2
                    if w["gamma"] and t > 0:
                        conf = soft[batch[:, t - 1]].amax(-1)
                        tad = tad + (conf * F.kl_div(logp, soft[idx], reduction="none").sum(-1)).mean() * T ** 2
            L = batch.shape[1]
            ce, kd, tad = ce / L, kd / L, tad / max(1, L - 1)
            loss = w["alpha"] * ce + w["beta"] * kd + w["gamma"] * tad
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {kind} epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = batch.numel()
            sums += torch.stack([loss.detach(), ce.detach(), kd.detach(), tad.detach()]) * n
            seen += n
        scheduler.step()
        logits = stream_predict(model, x, val_runs, device)
        score = metrics(y_val, logits[torch.as_tensor(val, device=device)].argmax(-1).cpu().numpy())
        history.append({"epoch": epoch, "train_loss": (sums / seen).tolist(), "validation": score,
                        "seconds": time.monotonic() - started})
        improved = score["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = score["macro_f1"], epoch, 0
            atomic_save(best_path, {"model_state_dict": model.state_dict(), "model_name": kind, "epoch": epoch,
                                   "validation": score, "fold_metadata": metadata, "config": config,
                                   "selection": "inner_validation_macro_f1", "inference": "causal streaming"})
        else:
            stale += 1
        write_json(history_path, history)
        print(f"fold {metadata['fold']:02d} {config['input']:7s} {kind:8s} ep {epoch:02d}/{epochs} val acc={score['accuracy']:.4f} "
              f"F1={score['macro_f1']:.4f} kappa={score['kappa']:.4f} {history[-1]['seconds']:.1f}s{' BEST' if improved else ''}",
              flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"])
    result = {"best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"], "epochs_run": len(history),
              "training_seconds": sum(h["seconds"] for h in history), "checkpoint": str(best_path)}
    write_json(done_path, result)
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10")
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10_students_context")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--inputs", nargs="+", default=["c4", "eeg_eog"], choices=list(INPUTS))
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--seqs-per-batch", type=int, default=32)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--skip-perepoch", action="store_true", help="Train/score only the context students")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    src_config = json.loads((args.source / "config.json").read_text())
    folds = {f["fold"]: f for f in json.loads((args.source / "folds.json").read_text())}
    data = load_data(repo_paths.ISLEEPS_CACHE / "psg4_100hz", TEACHER_CHANNELS)
    t = src_config["teacher"]
    windows = torch.as_tensor(context_windows(data, t["window"], t["target"]))
    x4 = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"]).to(device)
    T = src_config["temperature"]
    out = args.output.resolve()
    summary = []
    for k in args.folds:
        fold, name = folds[k], f"fold_{k:02d}"
        src_ckpt = args.source / "checkpoints" / name
        train, pairs, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        train_runs, val_runs, test_runs = runs_of(data, train), runs_of(data, val), runs_of(data, test)
        selection = json.loads((src_ckpt / "teacher_selection.json").read_text())
        softs = {}
        for seed in selection["seeds"]:
            ck = torch.load(src_ckpt / f"sleepgpt4_{seed}_{selection['loss']}" / "teacher.pt", map_location="cpu", weights_only=True)
            tm = SleepGPTMultiTeacher(t["window"], t["target"], pretrained=False).to(device)
            tm.load_state_dict(ck["model_state_dict"])
            softs[seed] = torch.softmax(teacher_logits(tm, x4, windows, device) / T, -1)
            del tm
            torch.cuda.empty_cache()
        soft = (torch.stack(list(softs.values())).mean(0) if selection["choice"] == "ensemble"
                else softs[selection["choice"]]).to(device)
        test_tensor = torch.as_tensor(test, device=device)
        test_y = data["y"][test]
        for inp in args.inputs:
            chans = INPUTS[inp]
            sel = [TEACHER_CHANNELS.index(c) for c in chans]
            sdata = {**data, "x": None, "records": [{**r, "sum": [r["sum"][i] for i in sel],
                                                      "sum_squares": [r["sum_squares"][i] for i in sel]} for r in data["records"]]}
            stats = normalization(sdata, fold["train"])
            mean = torch.tensor(stats["mean_uV"], dtype=torch.float32).view(1, -1, 1)
            std = torch.tensor(stats["std_uV"], dtype=torch.float32).view(1, -1, 1)
            x = ((x4[:, sel] - mean) / std).to(device)
            if len(chans) == 1:
                x = x[:, 0]
            config = {**src_config, "channels": chans, "input": inp, "seq_len": args.seq_len,
                      "seqs_per_batch": args.seqs_per_batch, "student": "SleepStudentNet (unchanged), stateful causal",
                      "student_normalization": "training-patient per-channel mean/std"}
            metadata = {**fold, "normalization": stats, "channels": chans}
            folder = out / inp / name
            write_json(folder / "split.json", metadata)
            results = {}
            predictions = {"y_true": test_y, "subject": data["subject"][test], "epoch_index": data["epoch_index"][test]}
            for kind in args.variants:
                model, result = train_variant(kind, x, y, train_runs, val_runs, soft, out / inp / "checkpoints" / name,
                                              metadata, config, device)
                logits = stream_predict(model, x, test_runs, device)
                prob = torch.softmax(logits[test_tensor], -1).cpu().numpy()
                predictions[f"context_{kind}_prob"], predictions[f"context_{kind}_pred"] = prob, prob.argmax(-1)
                results[f"context_{kind}"] = {**result, "test": metrics(test_y, prob.argmax(-1))}
                del model
            np.savez_compressed(folder / "test_predictions.npz", **predictions)
            # Per-epoch students: reuse the existing runs (C4, C4+EOG) or train them here with the unchanged
            # engine.train_model (other inputs); score per-epoch (as trained) and with streaming inference.
            old_root = {"c4": args.source, "eeg_eog": repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10_students_eeg_eog"}.get(inp)
            for kind in ([] if args.skip_perepoch else args.variants):
                if old_root is not None:
                    m = SleepStudentNet(n_channels=len(chans)).to(device)
                    m.load_state_dict(torch.load(old_root / "checkpoints" / name / f"{kind}.pt", map_location=device,
                                                 weights_only=True)["model_state_dict"])
                    prev = json.loads((old_root / "folds" / name / "test_metrics.json").read_text())["models"][kind]["test"]
                else:
                    pe_config = {**config, "student": "SleepStudentNet (unchanged), per-epoch"}
                    m, _ = train_student(kind, x, y, train, val, None, None, pairs, soft, out / inp / "perepoch_checkpoints" / name,
                                         metadata, pe_config, out / inp / "status.json")
                    logits = predict_student(m, kind, x, test, None, 256, 128, True)
                    prev = metrics(test_y, logits[test_tensor].argmax(-1).cpu().numpy())
                results[f"perepoch_{kind}_streamed"] = {"test": metrics(test_y, stream_predict(m, x, test_runs, device)[test_tensor].argmax(-1).cpu().numpy())}
                results[f"perepoch_{kind}"] = {"test": prev}
                del m
            write_json(folder / "test_metrics.json", {"fold": k, "input": inp, "test_subjects": fold["test"], "models": results})
            for key, v in results.items():
                s = v["test"]
                summary.append({"fold": k, "input": inp, "model": key, **{c: s[c] for c in ["accuracy", "macro_f1", "kappa"] + [f"f1_{c}" for c in ["W", "N1", "N2", "N3", "REM"]]}})
                print(f"TEST {name} {inp:7s} {key:28s} acc={s['accuracy']:.4f} F1={s['macro_f1']:.4f} kappa={s['kappa']:.4f} "
                      + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in ["W", "N1", "N2", "N3", "REM"]), flush=True)
            del x
            torch.cuda.empty_cache()
    if out.name == "sleepgpt4_cv10_students_context":
        write_json(out / ("summary.json" if args.inputs == ["c4", "eeg_eog"] else f"summary_{'_'.join(args.inputs)}.json"), summary)
    else:
        write_json(out / f"summary_folds_{'_'.join(map(str, args.folds))}.json", summary)


if __name__ == "__main__":
    main()
