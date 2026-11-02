"""Aggregate held-out iSLEEPS predictions by fold, by patient and pooled; never select on test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol_isleeps import STAGES, metrics, write_json

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


def mean_std(rows):
    return ({k: float(np.mean([r[k] for r in rows])) for k in METRICS},
            {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0. for k in METRICS})


def aggregate(output, plots=False):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    n_folds = config["n_folds"]
    completed = sorted((output / "folds").glob("*/test_metrics.json"))
    fold_rows, subject_rows, pooled_rows, summary_rows, report = [], [], [], [], {}
    pooled_predictions = {}
    for kind in MODELS:
        truths, preds, subjects, epochs, rows_fold, rows_subject = [], [], [], [], [], []
        for path in completed:
            item = json.loads(path.read_text())
            score = item["models"][kind]["test"]
            rows_fold.append({"model": kind, "fold": item["fold"], "test_subjects": len(item["test_subjects"]),
                              "best_epoch": item["models"][kind]["best_epoch"], "n_epochs": score["n_epochs"],
                              **{k: score[k] for k in METRICS}})
            with np.load(path.parent / "test_predictions.npz", allow_pickle=False) as d:
                truths.append(d["y_true"]); preds.append(d[f"{kind}_pred"])
                subjects.append(d["subject"]); epochs.append(d["epoch_index"])
                for subject in sorted(set(d["subject"].tolist()), key=lambda s: int(s[2:])):
                    m = d["subject"] == subject
                    s = metrics(d["y_true"][m], d[f"{kind}_pred"][m])
                    rows_subject.append({"model": kind, "fold": item["fold"], "subject": subject,
                                         "n_epochs": s["n_epochs"], **{k: s[k] for k in METRICS}})
        if not rows_fold:
            continue
        y, pred = np.concatenate(truths), np.concatenate(preds)
        all_subjects, all_epochs = np.concatenate(subjects), np.concatenate(epochs)
        identities = list(zip(all_subjects.tolist(), all_epochs.tolist()))
        assert len(set(identities)) == len(identities), "Duplicate out-of-fold predictions"
        pooled = metrics(y, pred)
        fold_mean, fold_std = mean_std(rows_fold)
        subject_mean, subject_std = mean_std(rows_subject)
        report[kind] = {"fold_mean": fold_mean, "fold_std": fold_std, "subject_mean": subject_mean,
                        "subject_std": subject_std, "pooled": pooled}
        pooled_predictions[f"{kind}_pred"] = pred
        fold_rows += rows_fold; subject_rows += rows_subject
        summary_rows.append({"model": kind, "folds_completed": len(rows_fold), "subjects": len(rows_subject),
                             **{f"{k}_{name}": values[k] for k in METRICS for name, values in
                                [("fold_mean", fold_mean), ("fold_std", fold_std),
                                 ("subject_mean", subject_mean), ("subject_std", subject_std)]}})
        pooled_rows.append({"model": kind, "folds_completed": len(rows_fold), "n_epochs": len(y),
                            **{k: pooled[k] for k in METRICS}})
    if not report:
        return
    pooled_predictions.update(y_true=y, subject=all_subjects, epoch_index=all_epochs)
    np.savez_compressed(output / "out_of_fold_predictions.npz", **pooled_predictions)
    write_json(output / "results.json", {
        "complete": len(completed) == n_folds, "folds_completed": len(completed), "folds_expected": n_folds,
        "channels": config["channels"],
        "aggregation": "Unweighted fold mean and patient mean with sample SD (ddof=1); pooled OOF also reported",
        "models": report})
    write_csv(output / "per_fold_metrics.csv", fold_rows)
    write_csv(output / "per_subject_metrics.csv", subject_rows)
    write_csv(output / "summary_mean_std.csv", summary_rows)
    write_csv(output / "pooled_metrics.csv", pooled_rows)
    table = lambda key_mean, key_std: [
        "| " + kind + " | " + " | ".join(f"{v[key_mean][k]:.4f} ± {v[key_std][k]:.4f}" for k in METRICS) + " |"
        for kind, v in report.items()]
    header = ["| Model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    text = [f"# iSLEEPS {'+'.join(config['channels'])} {n_folds}-fold patient CV: "
            f"{len(completed)}/{n_folds} folds complete", "",
            "Best epochs selected by inner-validation macro F1. Test patients never select checkpoints.", "",
            "## Patient mean ± sample SD", "", *header, *table("subject_mean", "subject_std"), "",
            "## Fold mean ± sample SD", "", *header, *table("fold_mean", "fold_std"), "",
            "## Pooled out-of-fold metrics", "", *header,
            *["| " + kind + " | " + " | ".join(f"{v['pooled'][k]:.4f}" for k in METRICS) + " |"
              for kind, v in report.items()], "",
            "All scores are fractions (0–1), except kappa which can be negative.",
            "Teacher: SE-ResNet18-BiLSTM (9-epoch window). Students: unchanged SleepStudentNet."]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(report), figsize=(17, 4), constrained_layout=True)
        for ax, (kind, value) in zip(np.atleast_1d(axes), report.items()):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    aggregate(args.output, args.plots)
