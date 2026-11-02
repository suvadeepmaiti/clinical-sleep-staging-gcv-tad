"""iSLEEPS 10-fold patient CV: multichannel (C3, C4, chin EMG, E1 EOG) SleepGPT-BiLSTM teacher
with 21-epoch context, distilled into the unchanged single-channel C4 SleepStudentNet.

All selection uses inner-validation patients only:
1. Loss: in fold 1, SleepGPT4 (seed 1) is trained with CE, class-balanced CE and focal loss; the
   loss with the best fold-1 inner-validation macro F1 is used in every fold (loss_selection.json).
2. Seeds: in every fold, `--seeds` SleepGPT4 models are fine-tuned with that loss (fold 1 reuses
   its seed-1 loss-selection model).
3. Teacher: per fold, the candidate with the best inner-validation macro F1 among the mean of all
   seeds' probabilities and each seed alone. Soft targets: the chosen candidate's
   softmax(logits / T) (averaged over seeds for the ensemble).
4. CE-only, No-KD and Full-TAD students (C4 at 100 Hz) exactly as in the earlier experiments.
"""
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from engine import predict as predict_student, train_model as train_student  # noqa: E402
from run_loso import save_transitions  # noqa: E402
from protocol_isleeps import (VARIANTS, context_windows, load_data, make_folds, metrics, normalization,  # noqa: E402
                              sha256, temporal_indices, write_json)
from report_isleeps import aggregate  # noqa: E402

from engine_multi import predict as predict_teacher, train_member  # noqa: E402
from sleepgpt_c4 import CHECKPOINT  # noqa: E402

TEACHER_CHANNELS = ["C3-M2", "C4-M1", "EMG", "E1-M2"]
LOSSES = ["ce", "cb", "focal"]
LOSS_FOLD = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.ISLEEPS_CACHE / "psg4_100hz")
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--folds", type=int, nargs="+", help="Run only these folds (default: all)")
    parser.add_argument("--seeds", type=int, default=2, help="SleepGPT4 seeds per fold (ensemble candidates)")
    parser.add_argument("--teacher-epochs", type=int, default=25)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--student-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    folds_requested = args.folds or list(range(1, args.n_folds + 1))
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    data = load_data(args.data_dir, TEACHER_CHANNELS)
    c4 = TEACHER_CHANNELS.index("C4-M1")
    strata = {r["subject"]: r["montage"] for r in data["records"]}
    folds = make_folds(np.unique(data["subject"]), strata, args.n_folds, seed=args.seed)
    previous = repo_paths.SPLITS / "isleeps_folds.json"  # folds of the published runs
    if previous.exists():
        assert json.loads(previous.read_text()) == folds, "Folds differ from the earlier iSLEEPS runs"
    member = {"name": "sleepgpt4", "arch": "sleepgpt", "epochs": args.teacher_epochs, "lr": 1e-4,
              "weight_decay": 0.05, "drop_path_rate": 0.2, "chunk": 64, "chunks_per_batch": 2}
    teacher_spec = {
        "architecture": "SleepGPT-BiLSTM", "member": member, "seeds_per_fold": args.seeds,
        "channels": TEACHER_CHANNELS, "sleepgpt_slots": ["C3", "C4", "EMG", "EOG"],
        "sleepgpt_checkpoint_sha256": sha256(CHECKPOINT),
        "lr_note": "base lr 1e-4 = chosen on inner validation in 4 of 6 folds of results/sleepgpt_c4_cv10",
        "loss_selection": f"fold {LOSS_FOLD} inner validation, seed 1, candidates {LOSSES}; applied to all folds",
        "teacher_selection": "per fold, best inner-validation macro F1 among the seed ensemble and each seed",
        "window": 21, "target": 10, "lstm_hidden": 100, "lstm_layers": 3, "dropout": 0.1,
        "layer_decay": 0.75, "warmup_fraction": 0.1, "scheduler": "cosine with linear warm-up, per step",
        "optimizer": "AdamW", "train_targets_per_epoch": "all training epochs", "mixed_precision": "bfloat16",
        "losses": {"ce": "cross-entropy", "cb": "class-balanced CE, effective-number weights (beta 0.9999)",
                   "focal": "focal loss, gamma 2"},
        "context": "21 consecutive epochs centred on the target; edges repeat the nearest epoch; "
                   "windows never cross removed epochs",
        "input": "uV with SleepGPT's fixed per-slot normalization"}
    config = {
        "dataset": "iSLEEPS", "n_folds": args.n_folds, "fold_stratification": "montage (M1/M2 vs A1/A2 labels)",
        "validation_fraction": 0.1, "teacher": teacher_spec, "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs, "teacher_batch": 128, "student_batch": args.student_batch,
        "patience": args.patience, "min_epochs": args.min_epochs, "seed": args.seed,
        "amp": not args.no_amp, "temperature": 2.0, "variants": VARIANTS,
        "channels": ["C4-M1"], "student": "SleepStudentNet (unchanged), C4 at 100 Hz",
        "checkpoint_selection": "inner validation macro F1",
        "student_optimizer": "Adam lr=0.001 weight_decay=0.0001", "student_scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0, "student_normalization": "training-patient C4 mean/std",
        "transition_weight": "max teacher probability at t, at T=2",
        "student_training": "CE/KD on every training epoch; TAD only on valid consecutive within-recording pairs",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "pipeline_sha256": {**{p.name: sha256(p) for p in [repo_paths.find(n) for n in [
            "teachers_multi.py", "engine_multi.py", "run_sleepgpt4_cv.py", "audit_sleepgpt4.py", "sleepgpt_c4.py",
            "check_sleepgpt_multi.py", "make_psg4_cache.py"]]},
            **{f"distillation/{n}": sha256(repo_paths.find(n)) for n in ["protocol_isleeps.py", "report_isleeps.py", "models_isleeps.py"]},
            **{f"sleep_distillation_loso/{n}": sha256(repo_paths.find(n)) for n in ["models.py", "protocol.py", "engine.py", "run_loso.py"]}},
        "data_sha256": {r["subject"]: r["sha256"] for r in data["records"]},
    }
    out = args.output.resolve()
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
               "python": sys.executable, "cuda": torch.version.cuda, "requested_folds": folds_requested,
               "started_unix": time.time()})
    status_path = out / "status.json"
    x4 = torch.from_numpy(data["x"]).to(device)                       # (N, 4, 3000) raw uV
    windows = torch.as_tensor(context_windows(data, teacher_spec["window"], teacher_spec["target"]), device=device)
    y = torch.from_numpy(data["y"]).to(device)
    all_indices = np.arange(len(data["y"]))
    student_data = {**data, "x": np.ascontiguousarray(data["x"][:, c4]), "channels": ["C4-M1"],
                    "records": [{**r, "sum": r["sum"][c4], "sum_squares": r["sum_squares"][c4]} for r in data["records"]]}
    T = config["temperature"]
    loss_path = out / "loss_selection.json"
    try:
        for fold in folds:
            if fold["fold"] not in folds_requested:
                continue
            name = f"fold_{fold['fold']:02d}"
            folder = out / "folds" / name
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "test_metrics.json").exists():
                print(f"Already complete: {name}", flush=True)
                continue
            train, pairs, _ = temporal_indices(data, fold["train"])
            val, _, _ = temporal_indices(data, fold["val"])
            test, _, _ = temporal_indices(data, fold["test"])
            assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
            assert len(train) + len(val) + len(test) == len(y)
            stats4 = normalization(data, fold["train"])
            stats = normalization(student_data, fold["train"])
            metadata = {**fold, "normalization": stats, "teacher_normalization": stats4, "n_train": len(train),
                        "n_val": len(val), "n_test": len(test), "n_training_pairs": len(pairs), "channels": ["C4-M1"]}
            write_json(folder / "split.json", metadata)
            print(f"\n{name}: train={len(train)} val={len(val)} test={len(test)} epochs; "
                  f"patients {len(fold['train'])}/{len(fold['val'])}/{len(fold['test'])}", flush=True)
            ckpt = out / "checkpoints" / name
            seed_spec = lambda k: {**member, "name": f"sleepgpt4_s{k}", "seed_offset": k}
            # 1. Loss selection (fold 1 only), seed 1.
            if fold["fold"] == LOSS_FOLD and not loss_path.exists():
                trials = []
                for loss in LOSSES:
                    model, result = train_member(seed_spec(1), loss, x4, y, train, val, windows,
                                                 ckpt / f"sleepgpt4_s1_{loss}", metadata, config, stats4, status_path)
                    trials.append(result)
                    del model
                    torch.cuda.empty_cache()
                chosen = max(trials, key=lambda r: r["best_validation_macro_f1"])["loss"]
                write_json(loss_path, {"fold": LOSS_FOLD, "rule": "highest best-epoch inner-validation macro F1",
                                       "selected_loss": chosen, "candidates": trials})
                print(f"selected loss={chosen} on fold {LOSS_FOLD} inner validation (" + ", ".join(
                    f"{r['loss']}:{r['best_validation_macro_f1']:.4f}" for r in trials) + ")", flush=True)
            if not loss_path.exists():
                raise RuntimeError(f"Run fold {LOSS_FOLD} first: it selects the loss for all folds")
            loss = json.loads(loss_path.read_text())["selected_loss"]
            # 2. Seeds with the chosen loss (fold 1 seed 1 is reused from loss selection).
            logits, seed_results = {}, {}
            for k in range(1, args.seeds + 1):
                model, result = train_member(seed_spec(k), loss, x4, y, train, val, windows,
                                             ckpt / f"sleepgpt4_s{k}_{loss}", metadata, config, stats4, status_path)
                logits[f"s{k}"] = predict_teacher(model, x4, windows, all_indices, not args.no_amp)
                seed_results[f"s{k}"] = result
                del model
                torch.cuda.empty_cache()
            # 3. Teacher choice on inner validation.
            prob = {k: torch.softmax(z, -1) for k, z in logits.items()}
            soft_of = {k: torch.softmax(z / T, -1) for k, z in logits.items()}
            if len(logits) > 1:
                prob = {"ensemble": torch.stack(list(prob.values())).mean(0), **prob}
                soft_of = {"ensemble": torch.stack(list(soft_of.values())).mean(0), **soft_of}
            val_tensor = torch.as_tensor(val, device=device)
            y_val = data["y"][val]
            val_scores = {k: metrics(y_val, p[val_tensor].argmax(-1).cpu().numpy()) for k, p in prob.items()}
            choice = max(val_scores, key=lambda k: val_scores[k]["macro_f1"])  # ties keep the ensemble
            soft = soft_of[choice].detach()
            write_json(ckpt / "teacher_selection.json", {
                "loss": loss, "rule": "highest inner-validation macro F1 (ensemble listed first, wins ties)",
                "choice": choice, "validation_macro_f1": {k: v["macro_f1"] for k, v in val_scores.items()},
                "seeds": seed_results})
            print(f"fold {fold['fold']:02d} teacher choice={choice} (" + ", ".join(
                f"{k}:{v['macro_f1']:.4f}" for k, v in val_scores.items()) + ")", flush=True)
            save_transitions(folder / "training_transitions.npz", soft, pairs)
            test_tensor = torch.as_tensor(test, device=device)
            test_y = data["y"][test]
            predictions = {"y_true": test_y, "subject": data["subject"][test], "epoch_index": data["epoch_index"][test]}
            candidate_test = {}
            for k, p in prob.items():
                pk = p[test_tensor].cpu().numpy()
                predictions[f"candidate_{k}_prob"] = pk
                candidate_test[k] = metrics(test_y, pk.argmax(-1))
            predictions["teacher_prob"] = predictions[f"candidate_{choice}_prob"]
            predictions["teacher_pred"] = predictions["teacher_prob"].argmax(-1)
            results = {"teacher": {"best_epoch": 0, "validation": val_scores[choice], "choice": choice, "loss": loss,
                                   "seeds": seed_results, "candidate_validation": val_scores,
                                   "candidate_test": candidate_test,
                                   "test": metrics(test_y, predictions["teacher_pred"])}}
            del logits, prob, soft_of
            # 4. Students.
            x = (torch.from_numpy(student_data["x"]).to(device) - float(stats["mean_uV"])) / float(stats["std_uV"])
            for kind in VARIANTS:
                model, result = train_student(kind, x, y, train, val, None, None, pairs, soft, ckpt,
                                              metadata, config, status_path)
                out_logits = predict_student(model, kind, x, test, None, args.student_batch, 128, not args.no_amp)
                probabilities = torch.softmax(out_logits[test_tensor], dim=-1).cpu().numpy()
                predictions[f"{kind}_prob"] = probabilities
                predictions[f"{kind}_pred"] = probabilities.argmax(-1)
                results[kind] = {**result, "test": metrics(test_y, predictions[f"{kind}_pred"])}
                del model, out_logits
            np.savez_compressed(folder / "test_predictions.npz", **predictions)
            write_json(folder / "test_metrics.json", {"fold": fold["fold"], "test_subjects": fold["test"],
                       "models": results})
            for kind, value in results.items():
                score = value["test"]
                print(f"TEST {name} {kind}: acc={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}, "
                      f"kappa={score['kappa']:.4f}", flush=True)
            for k, score in candidate_test.items():
                print(f"TEST {name} teacher candidate {k}: acc={score['accuracy']:.4f}, F1={score['macro_f1']:.4f}, "
                      f"kappa={score['kappa']:.4f}", flush=True)
            report(out)
            del x, soft
            torch.cuda.empty_cache()
        report(out, plots=True)
        completed = len(list((out / "folds").glob("*/test_metrics.json")))
        if completed == args.n_folds:
            from audit_sleepgpt4 import audit
            audit(out, args.data_dir)
        write_json(status_path, {"state": "complete" if completed == args.n_folds else "requested_folds_complete",
                   "folds_completed": completed, "folds_expected": args.n_folds})
        print(f"Finished {completed}/{args.n_folds} folds. Results: {out}", flush=True)
    except BaseException as error:
        write_json(status_path, {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                "error": repr(error), "resume": "Rerun the identical command."})
        raise


def report(out, plots=False):
    aggregate(out, plots)
    path = out / "RESULTS.md"
    if path.exists():
        path.write_text(path.read_text().replace(
            "Teacher: SE-ResNet18-BiLSTM (9-epoch window).",
            "Teacher: SleepGPT-BiLSTM on C3, C4, chin EMG and E1 EOG with 21-epoch context (loss chosen on fold-1 "
            "inner validation; seed ensemble or single seed chosen per fold on inner validation). Students: C4 only."))


if __name__ == "__main__":
    main()
