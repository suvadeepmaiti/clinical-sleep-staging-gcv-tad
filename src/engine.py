"""CUDA training with inner-validation checkpoint selection and epoch resume."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import SleepStudentNet, TinySleepNet
from protocol import VARIANTS, metrics, write_json


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def teacher_batches(groups, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    batches = []
    for values in groups.values():
        order = rng.permutation(len(values)) if seed is not None else np.arange(len(values))
        batches.extend(values[order[start:start + batch_size]] for start in range(0, len(order), batch_size))
    if seed is not None:
        rng.shuffle(batches)
    return batches


@torch.inference_mode()
def predict(model, kind, x, indices, groups, batch_size=256, teacher_batch=15, amp=True):
    model.eval()
    logits = torch.zeros((len(x), 5), device=x.device, dtype=torch.float32)
    batches = teacher_batches(groups, teacher_batch) if kind == "teacher" else [
        indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    for batch in batches:
        idx = torch.as_tensor(batch, device=x.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out, _ = model(x[idx])
        logits[idx.reshape(-1)] = out.reshape(-1, 5).float()
    if not torch.isfinite(logits[torch.as_tensor(indices, device=x.device)]).all():
        raise FloatingPointError("Nonfinite inference logits")
    return logits


def student_loss(model, x, y, idx, soft, next_index, valid_pair, weights, temperature):
    """Teacher targets are already softmax(logits/T); do NOT soften twice."""
    logits, hidden = model(x[idx])
    ce = F.cross_entropy(logits.float(), y[idx])
    kd = logits.new_zeros((), dtype=torch.float32)
    trans = logits.new_zeros((), dtype=torch.float32)
    if weights["beta"]:
        kd = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                      soft[idx], reduction="batchmean") * temperature ** 2
    mask = valid_pair[idx]
    if weights["gamma"] and bool(mask.any()):
        following, _ = model(x[next_index[idx[mask]]], hidden[:, mask, :].detach().contiguous())
        kl = F.kl_div(F.log_softmax(following.float() / temperature, dim=-1),
                      soft[next_index[idx[mask]]], reduction="none").sum(-1)
        confidence = soft[idx[mask]].amax(-1)
        trans = (confidence * kl).mean() * temperature ** 2
    loss = weights["alpha"] * ce + weights["beta"] * kd + weights["gamma"] * trans
    return loss, torch.stack([ce.detach(), kd.detach(), trans.detach()])


def train_model(kind, x, y, train, val, train_groups, val_groups, pairs, soft,
                folder, metadata, config, status_path):
    """Return best model; only train/validation indices are used in this routine."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed = config["seed"] + metadata["fold"] * 1000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_channels = len(config.get("channels", ["EEG Fpz-Cz"]))
    model = (TinySleepNet(n_channels=n_channels) if kind == "teacher" else
             SleepStudentNet(n_channels=n_channels)).to(x.device)
    best_path, last_path = folder / f"{kind}.pt", folder / f"{kind}_last.pt"
    history_path, done_path = folder / f"{kind}_history.json", folder / f"{kind}_complete.json"
    if done_path.exists():
        best = torch.load(best_path, map_location=x.device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        return model, json.loads(done_path.read_text())
    epochs = config["teacher_epochs"] if kind == "teacher" else config["student_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, 10, .5) if kind == "teacher" else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    first_epoch, best_f1, best_epoch, stale, history = 1, -1., 0, 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location=x.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng_cuda"]])
        first_epoch = state["epoch"] + 1
        best_f1, best_epoch, stale, history = state["best_f1"], state["best_epoch"], state["stale"], state["history"]
        print(f"Resuming {kind} at epoch {first_epoch}", flush=True)
    valid_pair = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    valid_pair[torch.as_tensor(pairs, device=x.device)] = True
    next_index = torch.arange(len(x), device=x.device)
    next_index[valid_pair] += 1
    y_val = y[torch.as_tensor(val, device=x.device)].cpu().numpy()
    for epoch in range(first_epoch, epochs + 1):
        if stale >= config["patience"] and epoch > config["min_epochs"]:
            break
        started = time.monotonic()
        write_json(status_path, {"state": "training", "fold": metadata["fold"],
                                "test_subject": metadata["test"], "model": kind,
                                "epoch": epoch, "max_epochs": epochs,
                                "best_validation_macro_f1": best_f1})
        model.train()
        if kind == "teacher":
            batches = teacher_batches(train_groups, config["teacher_batch"], seed + epoch)
        else:
            order = np.random.default_rng(seed + epoch).permutation(train)
            batches = [order[i:i + config["student_batch"]] for i in range(0, len(order), config["student_batch"])]
        sums = torch.zeros(4, device=x.device)
        seen = 0
        for batch in batches:
            idx = torch.as_tensor(batch, device=x.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                if kind == "teacher":
                    logits, _ = model(x[idx])
                    loss = F.cross_entropy(logits.float().reshape(-1, 5), y[idx].reshape(-1))
                    components = torch.stack([loss.detach(), loss.detach() * 0, loss.detach() * 0])
                else:
                    loss, components = student_loss(model, x, y, idx, soft, next_index, valid_pair,
                                                     VARIANTS[kind], config["temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {kind}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = idx.numel()
            sums += torch.cat([loss.detach().reshape(1), components]) * n
            seen += n
        scheduler.step()
        val_logits = predict(model, kind, x, val, val_groups, config["student_batch"],
                             config["teacher_batch"], config["amp"])
        val_pred = val_logits[torch.as_tensor(val, device=x.device)].argmax(-1).cpu().numpy()
        score = metrics(y_val, val_pred)
        elapsed = time.monotonic() - started
        history.append({"epoch": epoch, "train_loss": (sums / seen).cpu().tolist(),
                        "validation": score, "seconds": elapsed})
        improved = score["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = score["macro_f1"], epoch, 0
            atomic_save(best_path, {"model_state_dict": model.state_dict(), "model_name": kind,
                                   "epoch": epoch, "validation": score, "fold_metadata": metadata,
                                   "config": config, "selection": "inner_validation_macro_f1",
                                   "parameters": model.count_params()})
        else:
            stale += 1
        atomic_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                               "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                               "stale": stale, "history": history,
                               "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()})
        write_json(history_path, history)
        print(f"fold {metadata['fold']:02d} {kind:8s} ep {epoch:02d}/{epochs} "
              f"val acc={score['accuracy']:.4f} F1={score['macro_f1']:.4f} "
              f"kappa={score['kappa']:.4f} {elapsed:.1f}s{' BEST' if improved else ''}", flush=True)
    best = torch.load(best_path, map_location=x.device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    result = {"best_epoch": best_epoch, "validation": best["validation"], "epochs_run": len(history),
              "training_seconds": sum(h["seconds"] for h in history),
              "checkpoint": str(best_path), "parameters": model.count_params()}
    write_json(done_path, result)
    return model, result
