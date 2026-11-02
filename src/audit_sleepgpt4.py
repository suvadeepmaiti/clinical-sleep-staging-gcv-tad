"""Independently audit multichannel SleepGPT-teacher iSLEEPS CV predictions, metrics, splits and checkpoint selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

import repo_paths
from protocol_isleeps import STAGES, write_json  # noqa: E402
from report_isleeps import METRICS, MODELS  # noqa: E402

HERE = Path(__file__).resolve().parent


def independent_metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=range(5), average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred, labels=range(5)),
            **{f"f1_{s}": v for s, v in zip(STAGES, f1_score(y, pred, labels=range(5), average=None,
                                                              zero_division=0))}}


def audit(output, data_dir):
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    config = json.loads((output / "config.json").read_text())
    report = json.loads((output / "results.json").read_text())
    folds = json.loads((output / "folds.json").read_text())
    n_folds = config["n_folds"]
    assert report["complete"] and report["folds_completed"] == n_folds == len(folds)
    expected = {}
    for path in sorted(data_dir.glob("SN*.npz")):
        with np.load(path, allow_pickle=False) as d:
            for i, label in zip(d["epoch_index"], d["y"]):
                expected[(path.stem, int(i))] = int(label)
    all_subjects = {s for s, _ in expected}
    assert sorted(s for f in folds for s in f["test"]) == sorted(all_subjects)
    seen, per_fold = set(), {kind: [] for kind in MODELS}
    for spec in folds:
        folder = output / "folds" / f"fold_{spec['fold']:02d}"
        split = json.loads((folder / "split.json").read_text())
        result = json.loads((folder / "test_metrics.json").read_text())
        assert all(split[k] == spec[k] for k in ("fold", "train", "val", "test"))
        assert set(split["train"]) | set(split["val"]) | set(split["test"]) == all_subjects
        assert not set(split["train"]) & set(split["val"])
        assert not set(split["test"]) & set(split["train"] + split["val"])
        assert sorted(split["normalization"]["fit_subjects"]) == sorted(split["train"])
        assert split["channels"] == config["channels"]
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as d:
            assert set(d["subject"].tolist()) == set(split["test"])
            keys = [(s, int(i)) for s, i in zip(d["subject"].tolist(), d["epoch_index"])]
            assert not seen & set(keys) and len(set(keys)) == len(keys)
            seen.update(keys)
            np.testing.assert_array_equal(d["y_true"], [expected[k] for k in keys])
            for kind in MODELS:
                probabilities, pred = d[f"{kind}_prob"], d[f"{kind}_pred"]
                assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
                np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-5)
                np.testing.assert_array_equal(pred, probabilities.argmax(-1))
                scores = independent_metrics(d["y_true"], pred)
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, result["models"][kind]["test"][metric], atol=1e-12)
                np.testing.assert_array_equal(confusion_matrix(d["y_true"], pred, labels=range(5)),
                                              result["models"][kind]["test"]["confusion_matrix"])
                per_fold[kind].append(scores)
                ckpt_dir = output / "checkpoints" / folder.name
                if kind == "teacher":
                    sel = json.loads((ckpt_dir / "teacher_selection.json").read_text())
                    loss_sel = json.loads((output / "loss_selection.json").read_text())
                    assert loss_sel["selected_loss"] == max(loss_sel["candidates"],
                                                            key=lambda c: c["best_validation_macro_f1"])["loss"]
                    assert sel["loss"] == loss_sel["selected_loss"]
                    scores = sel["validation_macro_f1"]
                    assert sel["choice"] == max(scores, key=scores.get)
                    subs = [f"sleepgpt4_{k}_{sel['loss']}" for k in sel["seeds"]]
                    if spec["fold"] == loss_sel["fold"]:
                        subs += [f"sleepgpt4_s1_{c['loss']}" for c in loss_sel["candidates"]]
                    for sub in sorted(set(subs)):
                        history = json.loads((ckpt_dir / sub / "teacher_history.json").read_text())
                        info = json.loads((ckpt_dir / sub / "teacher_complete.json").read_text())
                        checkpoint = torch.load(ckpt_dir / sub / "teacher.pt", weights_only=True, map_location="cpu")
                        best = max(history, key=lambda row: row["validation"]["macro_f1"])
                        assert checkpoint["epoch"] == best["epoch"] == info["best_epoch"]
                        assert checkpoint["validation"] == best["validation"]
                        assert checkpoint["fold_metadata"] == split and checkpoint["config"] == config
                        assert checkpoint["selection"] == "inner_validation_macro_f1"
                        del checkpoint
                    seed_probs = [d[f"candidate_{k}_prob"] for k in sel["seeds"]]
                    if "ensemble" in scores:
                        np.testing.assert_allclose(d["candidate_ensemble_prob"], np.mean(seed_probs, axis=0), atol=1e-6)
                    np.testing.assert_array_equal(probabilities, d[f"candidate_{sel['choice']}_prob"])
                    continue
                history = json.loads((ckpt_dir / f"{kind}_history.json").read_text())
                checkpoint = torch.load(ckpt_dir / f"{kind}.pt", weights_only=True, map_location="cpu")
                best = max(history, key=lambda row: row["validation"]["macro_f1"])
                assert checkpoint["epoch"] == best["epoch"] == result["models"][kind]["best_epoch"]
                assert checkpoint["validation"] == best["validation"]
                assert checkpoint["fold_metadata"] == split and checkpoint["config"] == config
                assert checkpoint["selection"] == "inner_validation_macro_f1"
                assert checkpoint["model_state_dict"]["cnn.0.weight"].shape[1] == len(config["channels"])
                del checkpoint
    assert seen == set(expected), "Every retained epoch must be tested exactly once"
    with np.load(output / "out_of_fold_predictions.npz", allow_pickle=False) as d:
        assert len(d["y_true"]) == len(expected)
        for kind in MODELS:
            pooled = independent_metrics(d["y_true"], d[f"{kind}_pred"])
            for metric in METRICS:
                values = [row[metric] for row in per_fold[kind]]
                np.testing.assert_allclose(report["models"][kind]["fold_mean"][metric], np.mean(values), atol=1e-12)
                np.testing.assert_allclose(report["models"][kind]["fold_std"][metric], np.std(values, ddof=1), atol=1e-12)
                np.testing.assert_allclose(report["models"][kind]["pooled"][metric], pooled[metric], atol=1e-12)
    result = {"passed": True, "folds": n_folds, "best_checkpoints": n_folds * len(MODELS),
              "channels": config["channels"], "subjects": len(all_subjects),
              "unique_test_epochs_per_model": len(expected),
              "checks": ["Patient-disjoint train/validation/test in every fold", "Every patient tested once",
                         "Training-only normalization membership", "Every epoch tested exactly once",
                         "All prediction probabilities valid", "All metrics agree with scikit-learn",
                         "Fold means and SD verified",
                         "Every best checkpoint (students, all teacher members) matches maximum inner-validation macro F1",
                         "Loss and teacher candidate are the inner-validation best", "Ensemble equals the seed mean"]}
    write_json(output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.ISLEEPS_CACHE / "psg4_100hz")
    args = parser.parse_args()
    audit(args.output, args.data_dir)
