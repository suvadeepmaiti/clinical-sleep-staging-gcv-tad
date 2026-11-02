"""Context-aware students with a transition-matrix-weighted TAD term (ablation of the plain TAD term).

Same student, sequences, optimizer, schedule, streaming inference and checkpoint rule as
train_students_context.py; loss alpha*CE + gamma*L_trans with the "No KD" weights (0.3, 0.2), T = 2.
Plain TAD (train_students_context.py):  conf_{t-1} * KL(p_t || q_t).
Here, with s = teacher argmax at t-1, conf_{t-1} = max_k p_{t-1}[k] and G a fold-local 5x5 prior:
    L_trans = T^2 * conf_{t-1} * sum_k G[s, k] * (p_t[k] log(p_t[k] / q_t[k]) - p_t[k] + q_t[k])
(per-class generalized KL terms, each >= 0, so the weighted sum stays a divergence with minimum at q = p).
  gtad    : G = 1 - M, M = row-normalized transition matrix of all training-patient pairs (y_t -> y_t+1).
  gtad_cv : G = (1 - M_mean) / (1 + s_sens * CV), CV = M_std / (M_mean + eps), M_mean/M_std over the
            training patients' own row-normalized matrices (rows a patient never visits are skipped);
            s_sens = 0.5, eps = 1e-6.
--variants ce_only / no_kd train the unchanged baselines (train_students_context.train_variant) in the same
setting. Teachers are the existing fold checkpoints (iSLEEPS: 1-seed SleepGPT4; NAPS: masked SleepGPT); test
patients never enter M, G, training or selection.
"""
import repo_paths
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from train_students_context import (INPUTS, TEACHER_CHANNELS, SleepStudentNet, atomic_save, context_windows,  # noqa: E402
                                    load_data, metrics, normalization, runs_of, step, stream_predict,
                                    teacher_logits, temporal_indices, train_sequences, write_json)
from teachers_multi import SleepGPTMultiTeacher  # noqa: E402
from train_students_context import train_variant  # noqa: E402

GAMMA, ALPHA, S_SENS, EPS = 0.2, 0.3, 0.5, 1e-6


def transition_priors(y, train_runs, subject):
    pooled, per = np.zeros((5, 5)), {}
    for r in train_runs:
        c = per.setdefault(subject[r[0]], np.zeros((5, 5)))
        np.add.at(c, (y[r[:-1]], y[r[1:]]), 1)
        np.add.at(pooled, (y[r[:-1]], y[r[1:]]), 1)
    M = pooled / np.maximum(pooled.sum(1, keepdims=True), 1)
    rows = np.stack([np.where(c.sum(1, keepdims=True) > 0, c / np.maximum(c.sum(1, keepdims=True), 1), np.nan)
                     for c in per.values()])
    mean, std = np.nanmean(rows, 0), np.nanstd(rows, 0)
    cv = std / (mean + EPS)
    return {"M": M, "M_mean": mean, "M_std": std, "gtad": 1 - M, "gtad_cv": (1 - mean) / (1 + S_SENS * cv),
            "pairs": int(pooled.sum()), "patients": len(per)}


def train_gtad(kind, G, x, y, train_runs, val_runs, soft, folder, metadata, config, device):
    folder.mkdir(parents=True, exist_ok=True)
    best_path, done_path, history_path = folder / f"{kind}.pt", folder / f"{kind}_complete.json", folder / f"{kind}_history.json"
    seed = config["seed"] + metadata["fold"] * 1000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SleepStudentNet(n_channels=len(config["channels"])).to(device)
    if done_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"])
        return model, json.loads(done_path.read_text())
    epochs, T = config["student_epochs"], config["temperature"]
    G = torch.as_tensor(G, dtype=torch.float32, device=device)
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
        sums, seen = torch.zeros(3, device=device), 0
        for b in range(0, len(seqs), config["seqs_per_batch"]):
            batch = seqs[b:b + config["seqs_per_batch"]]
            optimizer.zero_grad(set_to_none=True)
            h = None
            ce = tr = torch.zeros((), device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                for t in range(batch.shape[1]):
                    idx = batch[:, t]
                    logits, h = step(model, x[idx], h)
                    ce = ce + torch.nn.functional.cross_entropy(logits.float(), y[idx])
                    if t > 0:
                        prev = soft[batch[:, t - 1]]
                        logq = torch.log_softmax(logits.float() / T, -1)
                        p = soft[idx].clamp_min(1e-8)
                        term = p * (p.log() - logq) - p + logq.exp()
                        tr = tr + (prev.amax(-1) * (G[prev.argmax(-1)] * term).sum(-1)).mean() * T ** 2
            L = batch.shape[1]
            ce, tr = ce / L, tr / max(1, L - 1)
            loss = ALPHA * ce + GAMMA * tr
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {kind} epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = batch.numel()
            sums += torch.stack([loss.detach(), ce.detach(), tr.detach()]) * n
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


def isleeps(fold_id, device):
    src = repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10_s1"
    config = json.loads((src / "config.json").read_text())
    fold = {f["fold"]: f for f in json.loads((src / "folds.json").read_text())}[fold_id]
    data = load_data(repo_paths.ISLEEPS_CACHE / "psg4_100hz", TEACHER_CHANNELS)
    t = config["teacher"]
    windows = torch.as_tensor(context_windows(data, t["window"], t["target"]))
    x4 = torch.from_numpy(data["x"])
    loss = json.loads((src / "loss_selection.json").read_text())["selected_loss"]
    member = src / "checkpoints" / f"fold_{fold_id:02d}" / f"sleepgpt4_s1_{loss}"
    assert (member / "teacher_complete.json").exists(), f"seed-1 teacher of fold {fold_id} not finished"
    tm = SleepGPTMultiTeacher(t["window"], t["target"], pretrained=False).to(device)
    tm.load_state_dict(torch.load(member / "teacher.pt",
                                  map_location="cpu", weights_only=True)["model_state_dict"])
    logits = teacher_logits(tm, x4, windows, device)
    del tm

    def inputs(inp, train_subjects):
        chans = INPUTS[inp]
        idx = [TEACHER_CHANNELS.index(c) for c in chans]
        sdata = {**data, "records": [{**r, "sum": [r["sum"][i] for i in idx], "sum_squares": [r["sum_squares"][i] for i in idx]}
                                     for r in data["records"]]}
        stats = normalization(sdata, train_subjects)
        x = x4[:, idx].sub_(torch.tensor(stats["mean_uV"]).view(1, -1, 1)).div_(torch.tensor(stats["std_uV"]).view(1, -1, 1))
        return chans, stats, x
    return data, fold, config, logits, inputs, ["eog", "c4", "eeg_eog"]


def naps(fold_id, device):
    import naps_cv
    src = repo_paths.RESULTS / "naps" / "naps_cv10"
    config = json.loads((src / "config.json").read_text())
    fold = json.loads((src / "folds.json").read_text())[fold_id - 1]
    data = naps_cv.load(repo_paths.PREPROCESSED["naps"])
    x4 = torch.from_numpy(data.pop("x"))
    windows = torch.as_tensor(context_windows(data, 21, 10))
    tm = naps_cv.build(config["teacher"]["member"], config, None).to(device)
    tm.load_state_dict(torch.load(src / "checkpoints" / f"fold_{fold_id:02d}" / "sleepgpt4m_s1_ce" / "teacher.pt",
                                  map_location="cpu", weights_only=True)["model_state_dict"])
    logits = teacher_logits(tm, x4, windows, device)
    del tm

    def inputs(inp, train_subjects):
        chans = naps_cv.STUDENT_INPUTS[inp]
        stats_all = normalization(data, train_subjects)
        idx = [naps_cv.STAT_CHANNELS.index(c) for c in chans]
        stats = {"mean_uV": [stats_all["mean_uV"][i] for i in idx], "std_uV": [stats_all["std_uV"][i] for i in idx]}
        x = torch.stack([naps_cv.student_channel(x4, c) for c in chans], 1)
        x.sub_(torch.tensor(stats["mean_uV"]).view(1, -1, 1)).div_(torch.tensor(stats["std_uV"]).view(1, -1, 1))
        return chans, stats, x
    return data, fold, config, logits, inputs, list(naps_cv.STUDENT_INPUTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["isleeps", "naps"], required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["gtad", "gtad_cv"],
                        choices=["ce_only", "no_kd", "gtad", "gtad_cv"], help="ce_only/no_kd = existing baselines")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out = (repo_paths.RESULTS / "isleeps" / "gtad_students_isleeps_s1" if args.dataset == "isleeps"
           else repo_paths.RESULTS / "naps" / "gtad_students")
    for k in args.folds:
        data, fold, src_config, logits, inputs, input_names = (isleeps if args.dataset == "isleeps" else naps)(k, device)
        T = src_config["temperature"]
        soft = torch.softmax(logits / T, -1).to(device)
        y_np = data["y"]
        y = torch.from_numpy(y_np).to(device)
        train, _, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        train_runs, val_runs, test_runs = runs_of(data, train), runs_of(data, val), runs_of(data, test)
        priors = transition_priors(y_np, train_runs, data["subject"])
        name = f"fold_{k:02d}"
        write_json(out / name / "transition_priors.json", {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                          for key, v in priors.items()})
        test_t = torch.as_tensor(test, device=device)
        for inp in input_names:
            chans, stats, x_cpu = inputs(inp, fold["train"])
            x = x_cpu.to(device)
            if len(chans) == 1:
                x = x[:, 0].contiguous()
            assert torch.isfinite(x).all()
            config = {**src_config, "channels": chans, "input": inp, "seq_len": 20, "seqs_per_batch": 32,
                      "student": "SleepStudentNet (unchanged), stateful causal", "alpha": ALPHA, "gamma": GAMMA,
                      "s_sens": S_SENS}
            metadata = {**fold, "normalization": stats, "channels": chans}
            results = {}
            for kind in args.variants:
                if kind in ("ce_only", "no_kd"):
                    model, res = train_variant(kind, x, y, train_runs, val_runs, soft, out / name / inp / "checkpoints",
                                               metadata, config, device)
                else:
                    model, res = train_gtad(kind, priors[kind], x, y, train_runs, val_runs, soft,
                                            out / name / inp / "checkpoints", metadata, config, device)
                pr = torch.softmax(stream_predict(model, x, test_runs, device)[test_t], -1).cpu().numpy()
                results[kind] = {**res, "test": metrics(y_np[test], pr.argmax(-1))}
                np.save(out / name / inp / f"{kind}_test_prob.npy", pr)
                s = results[kind]["test"]
                print(f"TEST {args.dataset} {name} {inp:7s} {kind:8s} val={res['validation']['macro_f1']:.4f} acc={s['accuracy']:.4f} "
                      f"F1={s['macro_f1']:.4f} kappa={s['kappa']:.4f} " + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in
                                                                                  ["W", "N1", "N2", "N3", "REM"]), flush=True)
                del model
            write_json(out / name / inp / "test_metrics.json", {"fold": k, "input": inp, "test_subjects": fold["test"],
                                                                "models": results})
            del x, x_cpu
            torch.cuda.empty_cache()
        del soft, logits, data


if __name__ == "__main__":
    main()
