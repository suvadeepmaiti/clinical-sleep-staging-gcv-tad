"""Independently audit complete LOSO predictions, metrics, and model selection."""
import repo_paths
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

from protocol import STAGES, write_json
from report import METRICS, MODELS


def independent_metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=range(5), average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred, labels=range(5)),
            **{f"f1_{s}": v for s, v in zip(STAGES, f1_score(y, pred, labels=range(5), average=None, zero_division=0))}}


def audit(output, data_dir, expected_folds=20):
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    report = json.loads((output / "results.json").read_text())
    assert 1 <= expected_folds <= 20
    assert report["folds_completed"] >= expected_folds
    if expected_folds == 20:
        assert report["complete"] and report["folds_completed"] == 20
    config = json.loads((output / "config.json").read_text())
    fold_specs = json.loads((output / "folds.json").read_text())[:expected_folds]
    test_subjects = {s for f in fold_specs for s in f["test"]}
    channels = config.get("channels", ["EEG Fpz-Cz"])
    expected = {}
    for path in sorted(data_dir.glob("SC4*.npz")):
        with np.load(path, allow_pickle=False) as d:
            if d["subject"].item() not in test_subjects:
                continue
            for i, label in zip(d["epoch_index"], d["y"]):
                expected[(path.stem, int(i))] = int(label)
    seen = set()
    per_model = {kind: [] for kind in MODELS}
    for spec in fold_specs:
        folder = output / "folds" / f"fold_{spec['fold']:02d}_{spec['test'][0]}"
        split = json.loads((folder / "split.json").read_text())
        result = json.loads((folder / "test_metrics.json").read_text())
        assert len(split["train"]) == 17 and len(split["val"]) == 2 and len(split["test"]) == 1
        assert not set(split["train"]) & set(split["val"])
        assert not set(split["test"]) & set(split["train"] + split["val"])
        assert sorted(split["normalization"]["fit_subjects"]) == sorted(split["train"])
        assert all(split[k] == spec[k] for k in ("fold", "train", "val", "test"))
        assert split["channels"] == channels
        assert np.asarray(split["normalization"]["mean_uV"]).size == len(channels)
        assert np.asarray(split["normalization"]["std_uV"]).size == len(channels)
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as d:
            assert set(d["subject"].tolist()) == set(split["test"])
            keys = [(r, int(i)) for r, i in zip(d["recording"].tolist(), d["epoch_index"])]
            assert not seen & set(keys) and len(set(keys)) == len(keys)
            seen.update(keys)
            np.testing.assert_array_equal(d["y_true"], [expected[k] for k in keys])
            for kind in MODELS:
                probabilities, pred = d[f"{kind}_prob"], d[f"{kind}_pred"]
                assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
                np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-6)
                np.testing.assert_array_equal(pred, probabilities.argmax(-1))
                scores = independent_metrics(d["y_true"], pred)
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, result["models"][kind]["test"][metric], atol=1e-12)
                np.testing.assert_array_equal(confusion_matrix(d["y_true"], pred, labels=range(5)),
                                              result["models"][kind]["test"]["confusion_matrix"])
                per_model[kind].append(scores)
                checkpoint_dir = output / "checkpoints" / folder.name
                history = json.loads((checkpoint_dir / f"{kind}_history.json").read_text())
                checkpoint = torch.load(checkpoint_dir / f"{kind}.pt", weights_only=True, map_location="cpu")
                best = max(history, key=lambda row: row["validation"]["macro_f1"])
                assert checkpoint["epoch"] == best["epoch"] == result["models"][kind]["best_epoch"]
                assert checkpoint["validation"] == best["validation"]
                assert checkpoint["fold_metadata"] == split
                assert checkpoint["selection"] == "inner_validation_macro_f1"
                assert checkpoint["config"] == config
                first_conv = "cnn_large.0.weight" if kind == "teacher" else "cnn.0.weight"
                assert checkpoint["model_state_dict"][first_conv].shape[1] == len(channels)
                del checkpoint
    assert seen == set(expected), "Every retained epoch must be tested exactly once"
    if expected_folds < 20:
        audit_result = {"passed": True, "folds": expected_folds, "channels": channels,
                        "best_checkpoints": expected_folds * 4,
                        "unique_test_epochs_per_model": len(expected),
                        "checks": ["Disjoint splits and correct channels", "Valid prediction probabilities",
                                   "Source labels and sklearn metrics agree", "Validation-selected checkpoints"]}
        write_json(output / f"audit_first_{expected_folds:02d}_folds.json", audit_result)
        print(json.dumps(audit_result, indent=2))
        return
    with np.load(output / "out_of_fold_predictions.npz", allow_pickle=False) as d:
        assert len(d["y_true"]) == len(expected)
        for kind in MODELS:
            pooled = independent_metrics(d["y_true"], d[f"{kind}_pred"])
            for metric in METRICS:
                values = [row[metric] for row in per_model[kind]]
                np.testing.assert_allclose(report["models"][kind]["fold_mean"][metric], np.mean(values), atol=1e-12)
                np.testing.assert_allclose(report["models"][kind]["fold_std"][metric], np.std(values, ddof=1), atol=1e-12)
                np.testing.assert_allclose(report["models"][kind]["pooled"][metric], pooled[metric], atol=1e-12)
    audit_result = {"passed": True, "folds": 20, "best_checkpoints": 80, "channels": channels,
                    "unique_test_epochs_per_model": len(expected),
                    "checks": ["All subject splits disjoint", "Training-only normalization membership",
                               "Every epoch tested exactly once", "All prediction probabilities valid",
                               "All metrics agree with scikit-learn", "Fold means and SD verified",
                               "Every best checkpoint matches maximum inner-validation macro F1"]}
    write_json(output / "audit.json", audit_result)
    print(json.dumps(audit_result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["sleepedf20"])
    parser.add_argument("--folds", type=int, default=20, help="Audit the first N folds; default all 20")
    args = parser.parse_args()
    audit(args.output, args.data_dir, args.folds)
