"""Run EEG or EEG+EOG teacher/CE/No-KD/Full-TAD LOSO on the local NVIDIA GPU."""
import repo_paths
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from engine import predict, train_model
from protocol import CHANNEL_SETS, VARIANTS, load_data, make_folds, metrics, normalization, sha256, temporal_indices, write_json
from report import aggregate

HERE = Path(__file__).resolve().parent


def save_transitions(path, soft, pairs):
    """Repository step-3 teacher transition statistics, using training pairs only."""
    p = soft.detach().cpu().numpy()
    states = p[pairs].argmax(-1)
    mean, std, counts = np.eye(5, dtype=np.float32), np.zeros((5, 5), dtype=np.float32), np.zeros(5, dtype=np.int64)
    for stage in range(5):
        following = p[pairs[states == stage] + 1]
        counts[stage] = len(following)
        if len(following):
            mean[stage] = following.mean(0)
        if len(following) > 1:
            std[stage] = following.std(0, ddof=1)
    np.savez_compressed(path, mean=mean, std=std, counts=counts,
                        pair_start_global_index=pairs, p_t=p[pairs], p_t1=p[pairs + 1],
                        confidence=p[pairs].max(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["sleepedf20"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--channels", choices=CHANNEL_SETS, default="eeg")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--teacher-batch", type=int, default=15)
    parser.add_argument("--student-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(f < 1 or f > 20 for f in args.folds):
        parser.error("--folds must contain distinct values between 1 and 20")
    if any(getattr(args, key) <= 0 for key in ("teacher_epochs", "student_epochs", "patience", "min_epochs", "teacher_batch", "student_batch")):
        parser.error("Epoch counts, patience and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run outside the restricted sandbox using the CUDA training environment.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    channels = CHANNEL_SETS[args.channels]
    print(f"Reading signals: {channels}", flush=True)
    data = load_data(args.data_dir, channels)
    folds = make_folds(np.unique(data["subject"]), args.seed)
    config = {
        "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs,
        "teacher_batch": args.teacher_batch, "student_batch": args.student_batch,
        "patience": args.patience, "min_epochs": args.min_epochs, "seed": args.seed,
        "amp": not args.no_amp, "temperature": 2.0, "variants": VARIANTS,
        "channels": channels, "modality": args.channels, "sequence_length": 20, "validation_subjects": 2,
        "checkpoint_selection": "inner validation macro F1", "optimizer": "Adam lr=0.001 weight_decay=0.0001",
        "teacher_scheduler": "StepLR step=10 gamma=0.5", "student_scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0, "transition_weight": "max teacher probability at t, at T=2",
        "student_training": "CE/KD on every training epoch; TAD only on valid consecutive within-recording pairs",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "pipeline_sha256": {p.name: sha256(p) for p in [HERE / n for n in
                              ["models.py", "protocol.py", "engine.py", "run_loso.py", "report.py", "audit_results.py"]]},
        "data_sha256": {r["recording"]: r["sha256"] for r in data["records"]},
    }
    out = (args.output or repo_paths.RESULTS / "sleepedf20" / f"{args.channels}_loso20").resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_stream = (out / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process is already training into {out}") from error
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise RuntimeError("Existing output has different settings/code/data. Choose a new --output folder.")
    else:
        if any(p.name != ".run.lock" for p in out.iterdir()):
            raise RuntimeError("New output directory must be empty")
        write_json(out / "config.json", config)
        write_json(out / "folds.json", folds)
    write_json(out / "runtime.json", {"gpu": torch.cuda.get_device_name(0), "pid": os.getpid(),
               "python": sys.executable, "cuda": torch.version.cuda, "requested_folds": args.folds,
               "started_unix": time.time()})
    status_path = out / "status.json"
    x_raw = torch.from_numpy(data["x"]).to(device)
    y = torch.from_numpy(data["y"]).to(device)
    all_indices, _, all_groups = temporal_indices(data, np.unique(data["subject"]))
    try:
        for fold in folds:
            if fold["fold"] not in args.folds:
                continue
            name = f"fold_{fold['fold']:02d}_{fold['test'][0]}"
            folder = out / "folds" / name
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "test_metrics.json").exists():
                print(f"Already complete: {name}", flush=True)
                continue
            train, pairs, train_groups = temporal_indices(data, fold["train"])
            val, _, val_groups = temporal_indices(data, fold["val"])
            test, _, test_groups = temporal_indices(data, fold["test"])
            assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
            assert len(train) + len(val) + len(test) == len(y)
            stats = normalization(data, fold["train"])
            metadata = {**fold, "normalization": stats, "n_train": len(train), "n_val": len(val),
                        "n_test": len(test), "n_training_pairs": len(pairs), "channels": channels}
            write_json(folder / "split.json", metadata)
            print(f"\n{name}: train={len(train)} val={len(val)} test={len(test)}; validation={fold['val']}", flush=True)
            mean = torch.as_tensor(stats["mean_uV"], device=device, dtype=x_raw.dtype)
            std = torch.as_tensor(stats["std_uV"], device=device, dtype=x_raw.dtype)
            if len(channels) > 1:
                mean, std = mean.reshape(1, -1, 1), std.reshape(1, -1, 1)
            x = (x_raw - mean) / std
            ckpt = out / "checkpoints" / name
            teacher, teacher_result = train_model("teacher", x, y, train, val, train_groups, val_groups,
                pairs, None, ckpt, metadata, config, status_path)
            teacher_logits = predict(teacher, "teacher", x, all_indices, all_groups,
                                     args.student_batch, args.teacher_batch, not args.no_amp)
            soft = torch.softmax(teacher_logits / config["temperature"], dim=-1).detach()
            save_transitions(folder / "training_transitions.npz", soft, pairs)
            test_tensor = torch.as_tensor(test, device=device)
            test_y = data["y"][test]
            predictions = {"y_true": test_y, "subject": data["subject"][test],
                           "recording": data["recording"][test], "epoch_index": data["epoch_index"][test]}
            teacher_prob = torch.softmax(teacher_logits[test_tensor], dim=-1).cpu().numpy()
            predictions["teacher_prob"] = teacher_prob
            predictions["teacher_pred"] = teacher_prob.argmax(-1)
            results = {"teacher": {**teacher_result, "test": metrics(test_y, predictions["teacher_pred"])}}
            del teacher, teacher_logits
            for kind in VARIANTS:
                model, result = train_model(kind, x, y, train, val, train_groups, val_groups,
                    pairs, soft, ckpt, metadata, config, status_path)
                logits = predict(model, kind, x, test, test_groups, args.student_batch,
                                 args.teacher_batch, not args.no_amp)
                probabilities = torch.softmax(logits[test_tensor], dim=-1).cpu().numpy()
                predictions[f"{kind}_prob"] = probabilities
                predictions[f"{kind}_pred"] = probabilities.argmax(-1)
                results[kind] = {**result, "test": metrics(test_y, predictions[f"{kind}_pred"])}
                del model, logits
            np.savez_compressed(folder / "test_predictions.npz", **predictions)
            write_json(folder / "test_metrics.json", {"fold": fold["fold"], "test_subject": fold["test"][0],
                       "models": results})
            for kind, value in results.items():
                score = value["test"]
                print(f"TEST {name} {kind}: acc={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}, "
                      f"kappa={score['kappa']:.4f}", flush=True)
            aggregate(out)
            del x, soft
            torch.cuda.empty_cache()
        aggregate(out, plots=True)
        completed = len(list((out / "folds").glob("*/test_metrics.json")))
        if completed == 20:
            with np.load(out / "out_of_fold_predictions.npz", allow_pickle=False) as predictions:
                assert len(predictions["y_true"]) == len(y)
                assert len(np.unique(predictions["subject"])) == 20
            from audit_results import audit
            audit(out, args.data_dir)
        write_json(status_path, {"state": "complete" if completed == 20 else "requested_folds_complete",
                   "folds_completed": completed, "folds_expected": 20})
        print(f"Finished {completed}/20 folds. Results: {out}", flush=True)
    except BaseException as error:
        write_json(status_path, {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                "error": repr(error), "resume": "Rerun the identical command."})
        raise


if __name__ == "__main__":
    main()
