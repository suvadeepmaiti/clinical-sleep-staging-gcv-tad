"""EEG+EOG students (C4-M1 + E1-M2) distilled from the 4-channel SleepGPT teacher of results/sleepgpt4_cv10.

For each requested completed fold: rebuild the fold's chosen teacher (seed ensemble or single seed,
as recorded in teacher_selection.json) from its checkpoints, recompute soft targets for all epochs
(checked against the run's saved teacher test probabilities), and train the unchanged two-channel
SleepStudentNet as CE-only, No KD and Full TAD with the same losses and settings as the C4 students.
Folds, inner-validation checkpoint selection and normalization (training patients only, per channel)
are identical to the source run. Teacher data stay in CPU memory and are streamed to the GPU, so
this can run beside the training job.
"""
import repo_paths
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from engine import predict as predict_student, train_model as train_student  # noqa: E402
from protocol_isleeps import VARIANTS, context_windows, load_data, metrics, normalization, write_json  # noqa: E402
from report_isleeps import aggregate  # noqa: E402

from teachers_multi import SleepGPTMultiTeacher  # noqa: E402

TEACHER_CHANNELS = ["C3-M2", "C4-M1", "EMG", "E1-M2"]
STUDENT_CHANNELS = ["C4-M1", "E1-M2"]


@torch.inference_mode()
def teacher_logits(model, x_cpu, windows, device, targets_per_batch=512, embed_batch=256):
    """Exact eval-mode logits for all epochs; x stays on the CPU, each batch's epochs are moved."""
    model.eval()
    out = torch.zeros((len(x_cpu), 5), dtype=torch.float32)
    for start in range(0, len(x_cpu), targets_per_batch):
        idx = torch.arange(start, min(start + targets_per_batch, len(x_cpu)))
        needed, inverse = torch.unique(windows[idx], return_inverse=True)
        xb = x_cpu[needed].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = torch.cat([model.embed(xb[s:s + embed_batch]) for s in range(0, len(xb), embed_batch)])
            out[idx] = model.forward_embeddings(emb.float()[inverse.to(device)]).float().cpu()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10")
    parser.add_argument("--output", type=Path, default=repo_paths.RESULTS / "isleeps" / "sleepgpt4_cv10_students_eeg_eog")
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    src_config = json.loads((args.source / "config.json").read_text())
    folds = {f["fold"]: f for f in json.loads((args.source / "folds.json").read_text())}
    data = load_data(repo_paths.ISLEEPS_CACHE / "psg4_100hz", TEACHER_CHANNELS)
    t = src_config["teacher"]
    windows = torch.as_tensor(context_windows(data, t["window"], t["target"]))
    x4 = torch.from_numpy(data["x"])
    sel = [TEACHER_CHANNELS.index(c) for c in STUDENT_CHANNELS]
    student_data = {**data, "x": np.ascontiguousarray(data["x"][:, sel]), "channels": STUDENT_CHANNELS,
                    "records": [{**r, "sum": [r["sum"][i] for i in sel], "sum_squares": [r["sum_squares"][i] for i in sel]}
                                for r in data["records"]]}
    config = {**src_config, "channels": STUDENT_CHANNELS,
              "student": "SleepStudentNet (unchanged architecture, 2 input channels), C4-M1 + E1-M2 at 100 Hz",
              "student_normalization": "training-patient per-channel mean/std",
              "teacher_source": str(args.source)}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_text()) == config, "Different settings: choose a new --output"
    else:
        write_json(out / "config.json", config)
        write_json(out / "folds.json", list(folds.values()))
    y = torch.from_numpy(data["y"]).to(device)
    T = config["temperature"]
    for k in args.folds:
        fold = folds[k]
        name = f"fold_{k:02d}"
        src_ckpt = args.source / "checkpoints" / name
        if not (args.source / "folds" / name / "test_metrics.json").exists():
            raise RuntimeError(f"{name} of the source run is not complete yet")
        folder = out / "folds" / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "test_metrics.json").exists():
            print(f"Already complete: {name}", flush=True)
            continue
        from protocol_isleeps import temporal_indices
        train, pairs, _ = temporal_indices(data, fold["train"])
        val, _, _ = temporal_indices(data, fold["val"])
        test, _, _ = temporal_indices(data, fold["test"])
        selection = json.loads((src_ckpt / "teacher_selection.json").read_text())
        probs, softs = [], []
        for seed in selection["seeds"]:
            ck = torch.load(src_ckpt / f"sleepgpt4_{seed}_{selection['loss']}" / "teacher.pt",
                            map_location="cpu", weights_only=True)
            model = SleepGPTMultiTeacher(t["window"], t["target"], pretrained=False).to(device)
            model.load_state_dict(ck["model_state_dict"])
            z = teacher_logits(model, x4, windows, device)
            probs.append((seed, torch.softmax(z, -1))); softs.append((seed, torch.softmax(z / T, -1)))
            del model
            torch.cuda.empty_cache()
        if selection["choice"] == "ensemble":
            prob = torch.stack([p for _, p in probs]).mean(0); soft = torch.stack([s for _, s in softs]).mean(0)
        else:
            prob = dict(probs)[selection["choice"]]; soft = dict(softs)[selection["choice"]]
        with np.load(args.source / "folds" / name / "test_predictions.npz") as d:
            agree = float((d["teacher_prob"].argmax(-1) == prob[test].argmax(-1).numpy()).mean())
            maxdiff = float(np.abs(d["teacher_prob"] - prob[test].numpy()).max())
        print(f"{name}: teacher={selection['choice']} recomputed vs saved test predictions: argmax agreement "
              f"{agree:.4%}, max |prob diff| {maxdiff:.2e}", flush=True)
        assert agree > 0.995, "Recomputed teacher differs from the saved teacher"
        stats = normalization(student_data, fold["train"])
        metadata = {**fold, "normalization": stats, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                    "n_training_pairs": len(pairs), "channels": STUDENT_CHANNELS}
        write_json(folder / "split.json", metadata)
        mean = torch.tensor(stats["mean_uV"], dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(stats["std_uV"], dtype=torch.float32).view(1, -1, 1)
        x = ((torch.from_numpy(student_data["x"]) - mean) / std).to(device)
        soft = soft.to(device)
        test_tensor = torch.as_tensor(test, device=device)
        test_y = data["y"][test]
        predictions = {"y_true": test_y, "subject": data["subject"][test], "epoch_index": data["epoch_index"][test],
                       "teacher_prob": prob[test].numpy(), "teacher_pred": prob[test].numpy().argmax(-1)}
        results = {"teacher": {"best_epoch": 0, "choice": selection["choice"],
                               "test": metrics(test_y, predictions["teacher_pred"])}}
        for kind in VARIANTS:
            model, result = train_student(kind, x, y, train, val, None, None, pairs, soft, out / "checkpoints" / name,
                                          metadata, config, out / "status.json")
            logits = predict_student(model, kind, x, test, None, 256, 128, True)
            p = torch.softmax(logits[test_tensor], -1).cpu().numpy()
            predictions[f"{kind}_prob"], predictions[f"{kind}_pred"] = p, p.argmax(-1)
            results[kind] = {**result, "test": metrics(test_y, predictions[f"{kind}_pred"])}
            del model, logits
        np.savez_compressed(folder / "test_predictions.npz", **predictions)
        write_json(folder / "test_metrics.json", {"fold": k, "test_subjects": fold["test"], "models": results})
        for kind, value in results.items():
            s = value["test"]
            print(f"TEST {name} {kind} (EEG+EOG students): acc={s['accuracy']:.4f} F1={s['macro_f1']:.4f} "
                  f"kappa={s['kappa']:.4f} " + " ".join(f"{c}={s['f1_' + c]:.3f}" for c in ["W", "N1", "N2", "N3", "REM"]),
                  flush=True)
        del x, soft
        torch.cuda.empty_cache()
    aggregate(out)


if __name__ == "__main__":
    main()
