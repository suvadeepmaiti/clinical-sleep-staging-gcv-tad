"""Aggregate held-out predictions; never select models on test results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol import STAGES, metrics, write_json

MODELS = ["teacher", "ce_only", "no_kd", "full_tad"]
METRICS = ["accuracy", "macro_f1", "kappa"] + [f"f1_{s}" for s in STAGES]


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def aggregate(output, plots=False):
    output = Path(output)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    channels = config.get("channels", ["EEG Fpz-Cz"])
    modality = "EEG+EOG" if len(channels) == 2 else "EEG-only"
    completed = sorted((output / "folds").glob("*/test_metrics.json"))
    rows, pooled_rows, summary_rows, report = [], [], [], {}
    pooled_predictions = {}
    for kind in MODELS:
        truths, preds, subjects, recordings, epochs = [], [], [], [], []
        model_rows = []
        for path in completed:
            item = json.loads(path.read_text())
            score = item["models"][kind]["test"]
            row = {"model": kind, "fold": item["fold"], "test_subject": item["test_subject"],
                   "best_epoch": item["models"][kind]["best_epoch"], "n_epochs": score["n_epochs"],
                   **{key: score[key] for key in METRICS}}
            model_rows.append(row); rows.append(row)
            with np.load(path.parent / "test_predictions.npz", allow_pickle=False) as d:
                truths.append(d["y_true"]); preds.append(d[f"{kind}_pred"])
                subjects.append(d["subject"]); recordings.append(d["recording"]); epochs.append(d["epoch_index"])
        if not model_rows:
            continue
        means = {key: float(np.mean([r[key] for r in model_rows])) for key in METRICS}
        stds = {key: float(np.std([r[key] for r in model_rows], ddof=1)) if len(model_rows) > 1 else 0.
                for key in METRICS}
        y, pred = np.concatenate(truths), np.concatenate(preds)
        pooled = metrics(y, pred)
        all_subjects, all_recordings, all_epochs = map(np.concatenate, [subjects, recordings, epochs])
        identities = list(zip(all_recordings.tolist(), all_epochs.tolist()))
        assert len(set(identities)) == len(identities), "Duplicate out-of-fold predictions"
        pooled_predictions[f"{kind}_pred"] = pred
        report[kind] = {"fold_mean": means, "fold_std": stds, "pooled": pooled}
        summary_rows.append({"model": kind, "folds_completed": len(model_rows),
                             **{f"{key}_{stat}": values[key] for key in METRICS
                                for stat, values in [("mean", means), ("std", stds)]}})
        pooled_rows.append({"model": kind, "folds_completed": len(model_rows), "n_epochs": len(y),
                            **{key: pooled[key] for key in METRICS}})
    if not report:
        return
    pooled_predictions.update(y_true=y, subject=all_subjects, recording=all_recordings, epoch_index=all_epochs)
    np.savez_compressed(output / "out_of_fold_predictions.npz", **pooled_predictions)
    complete = len(completed) == 20
    write_json(output / "results.json", {"complete": complete, "folds_completed": len(completed),
               "channels": channels,
               "folds_expected": 20, "aggregation": "Unweighted fold mean; sample SD (ddof=1); pooled OOF also reported",
               "models": report})
    write_csv(output / "per_fold_metrics.csv", rows)
    write_csv(output / "summary_mean_std.csv", summary_rows)
    write_csv(output / "pooled_metrics.csv", pooled_rows)
    text = [f"# {modality} LOSO: {len(completed)}/20 folds complete", "",
            "Best epochs selected by inner-validation macro F1. Test subjects never select checkpoints.", "",
            "## Unweighted fold mean ± sample standard deviation", "",
            "| Model | " + " | ".join(METRICS) + " |",
            "|---|" + "---|" * len(METRICS)]
    for kind, values in report.items():
        text.append("| " + kind + " | " + " | ".join(
            f"{values['fold_mean'][key]:.4f} ± {values['fold_std'][key]:.4f}" for key in METRICS) + " |")
    text += ["", "## Pooled out-of-fold metrics", "", "| Model | " + " | ".join(METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for kind, values in report.items():
        text.append("| " + kind + " | " + " | ".join(f"{values['pooled'][key]:.4f}" for key in METRICS) + " |")
    text += ["", "All scores are fractions (0–1), except kappa which can be negative.",
             "No KD omits the current-epoch KD term but retains teacher-based transition supervision.",
             "Models are the supplied repository's implementations; see protocol.md for corrections and settings."]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
        for ax, (kind, value) in zip(axes, report.items()):
            cm = np.array(value["pooled"]["confusion_matrix"])
            ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] / max(cm[i].sum(), 1) > .5 else "black")
            ax.set(title=kind, xlabel="Predicted", ylabel="True", xticks=range(5), yticks=range(5),
                   xticklabels=STAGES, yticklabels=STAGES)
        fig.savefig(output / "confusion_matrices.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        positions = np.arange(5)
        for i, (kind, value) in enumerate(report.items()):
            ax.bar(positions + (i - 1.5) * .2, [value["pooled"][f"f1_{s}"] for s in STAGES], .2, label=kind)
        ax.set(xticks=positions, xticklabels=STAGES, ylabel="Pooled out-of-fold F1", ylim=(0, 1),
               title=f"{modality} LOSO ({len(completed)}/20 folds)")
        ax.legend(); fig.savefig(output / "per_class_f1.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    aggregate(args.output, args.plots)
