"""Train one multichannel context-teacher member; inner-validation selection, exact resume.

Each epoch uses every training target once (no subsampling), in shuffled chunks of `chunk`
consecutive targets; a batch embeds every epoch its windows need once and shares it across the
overlapping context windows. AdamW with cosine decay and linear warm-up (per step), bfloat16,
gradient clip 1.0, early stopping on inner-validation macro F1. Loss: 'ce', 'cb' (class-balanced
cross-entropy with effective-number weights from the training labels, beta 0.9999) or 'focal'
(gamma 2). Test patients are never used here.
"""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import atomic_save  # noqa: E402
from protocol import metrics, write_json  # noqa: E402

from teachers_multi import SEResNetMultiTeacher, SleepGPTMultiTeacher  # noqa: E402


def build(spec, config, stats):
    t = config["teacher"]
    head = {"lstm_hidden": t["lstm_hidden"], "lstm_layers": t["lstm_layers"], "dropout": t["dropout"]}
    if spec["arch"] == "sleepgpt":
        return SleepGPTMultiTeacher(t["window"], t["target"], drop_path_rate=spec["drop_path_rate"],
                                    pretrained=True, **head)
    if spec["arch"] == "seresnet":
        return SEResNetMultiTeacher(stats["mean_uV"], stats["std_uV"], t["window"], t["target"], **head)
    raise ValueError(spec["arch"])


def class_weights(labels, beta=0.9999):
    counts = np.bincount(labels, minlength=5).astype(np.float64)
    w = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1)))
    return (w / w.sum() * 5).tolist()


def loss_fn(logits, target, kind, weights):
    logits = logits.float()
    if kind == "ce":
        return F.cross_entropy(logits, target)
    if kind == "cb":
        return F.cross_entropy(logits, target, weight=weights)
    if kind == "focal":
        logp = F.log_softmax(logits, -1).gather(1, target[:, None]).squeeze(1)
        return (-(1 - logp.exp()) ** 2 * logp).mean()
    raise ValueError(kind)


def windowed_logits(model, x, windows, idx, amp, embed_batch=None):
    needed, inverse = torch.unique(windows[idx], return_inverse=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if embed_batch is None:
            emb = model.embed(x[needed])
        else:
            emb = torch.cat([model.embed(x[needed[s:s + embed_batch]]) for s in range(0, len(needed), embed_batch)])
        return model.forward_embeddings(emb.float()[inverse])


@torch.inference_mode()
def predict(model, x, windows, indices, amp=True, targets_per_batch=1024, embed_batch=256):
    model.eval()
    indices = torch.as_tensor(indices, device=x.device, dtype=torch.long)
    logits = torch.zeros((len(x), 5), device=x.device, dtype=torch.float32)
    for start in range(0, len(indices), targets_per_batch):
        idx = indices[start:start + targets_per_batch]
        logits[idx] = windowed_logits(model, x, windows, idx, amp, embed_batch).float()
    if not torch.isfinite(logits[indices]).all():
        raise FloatingPointError("Nonfinite teacher logits")
    return logits


def train_member(spec, loss_kind, x, y, train, val, windows, folder, metadata, config, stats, status_path):
    """Returns (model with best weights, result dict). `folder` is unique to (member, loss)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t = config["teacher"]
    seed = config["seed"] + metadata["fold"] * 1000 + spec["seed_offset"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build(spec, config, stats).to(x.device)
    best_path, last_path = folder / "teacher.pt", folder / "teacher_last.pt"
    history_path, done_path = folder / "teacher_history.json", folder / "teacher_complete.json"
    if done_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=x.device, weights_only=True)["model_state_dict"])
        return model, json.loads(done_path.read_text())
    epochs, chunk = spec["epochs"], spec["chunk"]
    train = np.asarray(train)
    chunks = [train[i:i + chunk] for i in range(0, len(train), chunk)]
    steps_per_epoch = math.ceil(len(chunks) / spec["chunks_per_batch"])
    total, warmup = epochs * steps_per_epoch, int(t["warmup_fraction"] * epochs * steps_per_epoch)
    groups = (model.parameter_groups(spec["lr"], t["layer_decay"], spec["weight_decay"]) if spec["arch"] == "sleepgpt"
              else model.parameter_groups(spec["lr"], spec["weight_decay"]))
    optimizer = torch.optim.AdamW(groups)
    schedule = lambda step: (step + 1) / max(1, warmup) if step < warmup else \
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    weights = torch.tensor(class_weights(y[torch.as_tensor(train, device=x.device)].cpu().numpy()),
                           device=x.device, dtype=torch.float32)
    first_epoch, best_f1, best_epoch, stale, history = 1, -1., 0, 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location=x.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng_cuda"]])
        first_epoch = state["epoch"] + 1
        best_f1, best_epoch, stale, history = state["best_f1"], state["best_epoch"], state["stale"], state["history"]
        print(f"Resuming {spec['name']} ({loss_kind}) at epoch {first_epoch}", flush=True)
    y_val = y[torch.as_tensor(val, device=x.device)].cpu().numpy()
    for epoch in range(first_epoch, epochs + 1):
        if stale >= config["patience"] and epoch > config["min_epochs"]:
            break
        started = time.monotonic()
        write_json(status_path, {"state": "training", "fold": metadata["fold"], "test_subjects": metadata["test"],
                                "model": f"teacher member {spec['name']} (loss {loss_kind})", "epoch": epoch,
                                "max_epochs": epochs, "best_validation_macro_f1": best_f1})
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(chunks))
        sums, seen = torch.zeros(1, device=x.device), 0
        for start in range(0, len(order), spec["chunks_per_batch"]):
            idx = torch.as_tensor(np.concatenate([chunks[i] for i in order[start:start + spec["chunks_per_batch"]]]),
                                  device=x.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(windowed_logits(model, x, windows, idx, config["amp"]), y[idx], loss_kind, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {spec['name']} {loss_kind}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sums += loss.detach() * idx.numel()
            seen += idx.numel()
        val_logits = predict(model, x, windows, val, config["amp"])
        score = metrics(y_val, val_logits[torch.as_tensor(val, device=x.device)].argmax(-1).cpu().numpy())
        elapsed = time.monotonic() - started
        tl = float((sums / seen).item())
        history.append({"epoch": epoch, "train_loss": [tl, tl, 0.0, 0.0], "train_targets": int(seen),
                        "validation": score, "seconds": elapsed})
        improved = score["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = score["macro_f1"], epoch, 0
            atomic_save(best_path, {"model_state_dict": model.state_dict(), "model_name": "teacher",
                                   "architecture": spec["name"], "loss": loss_kind, "epoch": epoch,
                                   "validation": score, "fold_metadata": metadata, "config": config,
                                   "selection": "inner_validation_macro_f1", "parameters": model.count_params()})
        else:
            stale += 1
        atomic_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "epoch": epoch, "best_f1": best_f1,
                               "best_epoch": best_epoch, "stale": stale, "history": history,
                               "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()})
        write_json(history_path, history)
        print(f"fold {metadata['fold']:02d} {spec['name']:<12s} {loss_kind:<5s} ep {epoch:02d}/{epochs} "
              f"loss={tl:.4f} val acc={score['accuracy']:.4f} F1={score['macro_f1']:.4f} "
              f"kappa={score['kappa']:.4f} {elapsed:.1f}s{' BEST' if improved else ''}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=x.device, weights_only=True)["model_state_dict"])
    result = {"member": spec["name"], "loss": loss_kind, "best_epoch": best_epoch,
              "best_validation_macro_f1": best_f1, "validation": history[best_epoch - 1]["validation"],
              "epochs_run": len(history), "training_seconds": sum(h["seconds"] for h in history),
              "checkpoint": str(best_path), "parameters": model.count_params(), "class_weights": weights.tolist()}
    write_json(done_path, result)
    last_path.unlink(missing_ok=True)
    return model, result
