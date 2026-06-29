import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_kfold import evaluate


def parse_fold_epoch(value):
    try:
        fold_text, epoch_text = value.split(":", maxsplit=1)
        fold = int(fold_text)
        epoch = int(epoch_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected FOLD:EPOCH, got '{value}'. Example: 1:8"
        ) from exc

    if fold < 1:
        raise argparse.ArgumentTypeError("Fold numbers must be >= 1.")
    if epoch < 1:
        raise argparse.ArgumentTypeError("Epoch numbers must be >= 1.")

    return fold, epoch


def parse_ratio(value):
    try:
        positive_text, negative_text = value.split(":", maxsplit=1)
        positive = float(positive_text)
        negative = float(negative_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected POSITIVE:NEGATIVE ratio, got '{value}'. Example: 1:100"
        ) from exc

    if positive <= 0 or negative <= 0:
        raise argparse.ArgumentTypeError("Ratio values must be > 0.")

    return positive, negative


def logit(values):
    eps = np.finfo(float).eps
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_scores_to_ratio(y_true, y_scores, target_ratio):
    positive, negative = target_ratio
    source_prevalence = float(np.mean(y_true))
    target_prevalence = positive / (positive + negative)

    if source_prevalence <= 0.0 or source_prevalence >= 1.0:
        raise ValueError(
            "Cannot calibrate logits because observed OOF labels contain only one class."
        )

    offset = logit(target_prevalence) - logit(source_prevalence)
    calibrated_scores = sigmoid(logit(y_scores) + offset)
    return calibrated_scores, {
        "method": "prior_logit_shift",
        "target_ratio": f"{positive:g}:{negative:g}",
        "source_prevalence": source_prevalence,
        "target_prevalence": target_prevalence,
        "logit_offset": float(offset),
    }


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def flatten_metrics(metrics, prefix=""):
    flat = {}
    for key, value in metrics.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def format_selection_label(fold_epochs):
    return "custom_oof_" + "_".join(
        f"f{fold}_e{epoch}" for fold, epoch in fold_epochs
    )


def load_custom_oof(predictions_dir, fold_epochs):
    y_true_sets = []
    y_score_sets = []
    fold_ids = []
    source_files = []

    seen_folds = set()
    for fold, epoch in fold_epochs:
        if fold in seen_folds:
            raise ValueError(f"Fold {fold} was provided more than once.")
        seen_folds.add(fold)

        path = predictions_dir / f"fold_{fold}" / f"epoch_{epoch}_val_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Prediction file does not exist: {path}")

        data = np.load(path)
        y_true = np.asarray(data["y_true"]).reshape(-1).astype(int)
        y_scores = np.asarray(data["y_scores"]).reshape(-1).astype(float)

        if y_true.shape != y_scores.shape:
            raise ValueError(f"Shape mismatch in {path}: y_true and y_scores differ.")

        y_true_sets.append(y_true)
        y_score_sets.append(y_scores)
        fold_ids.append(np.full(shape=y_true.shape, fill_value=fold))
        source_files.append(path)

    return (
        np.concatenate(y_true_sets),
        np.concatenate(y_score_sets),
        np.concatenate(fold_ids),
        source_files,
    )


def save_results(
    output_dir,
    label,
    fold_epochs,
    y_true,
    y_scores,
    folds,
    metrics,
    source_files,
    calibration,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"{label}_predictions.npz"
    np.savez_compressed(
        predictions_path,
        y_true=y_true,
        y_scores=y_scores,
        folds=folds,
        fold_epochs=np.array(fold_epochs),
        source_files=np.array([str(path) for path in source_files]),
        calibration=json.dumps(calibration) if calibration is not None else "",
    )

    payload = {
        "fold_epochs": [
            {"fold": fold, "epoch": epoch} for fold, epoch in fold_epochs
        ],
        "input_prediction_files": [str(path) for path in source_files],
        "calibration": calibration,
        "metrics": metrics,
    }

    metrics_json_path = output_dir / f"{label}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)

    metrics_csv_path = output_dir / f"{label}_metrics.csv"
    flat_metrics = flatten_metrics(to_serializable(metrics))
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
        writer.writeheader()
        writer.writerow(flat_metrics)

    print(f"Saved custom OOF predictions to: {predictions_path}")
    print(f"Saved custom OOF metrics JSON to: {metrics_json_path}")
    print(f"Saved custom OOF metrics CSV to: {metrics_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine selected fold/epoch validation predictions into one OOF evaluation."
    )
    parser.add_argument(
        "--predictions_dir",
        "--predictions-dir",
        required=True,
        type=Path,
        help="Experiment directory containing fold_<N>/epoch_<M>_val_predictions.npz files.",
    )
    parser.add_argument(
        "--fold_epoch",
        "--fold-epoch",
        nargs="+",
        required=True,
        type=parse_fold_epoch,
        help="Fold/epoch pairs as FOLD:EPOCH. Example: --fold_epoch 1:8 2:10 3:6 4:10 5:8",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files. Defaults to --predictions_dir.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename prefix. Defaults to a prefix based on selected folds/epochs.",
    )
    parser.add_argument(
        "--n_bootstrap",
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for metric evaluation.",
    )
    parser.add_argument(
        "--calibration_ratio",
        "--calibration-ratio",
        type=parse_ratio,
        default=None,
        help=(
            "Apply prior logit-shift calibration to a known positive:negative "
            "ratio before evaluation. Example: 1:100"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions_dir = args.predictions_dir
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {predictions_dir}")

    fold_epochs = sorted(args.fold_epoch, key=lambda item: item[0])
    label = args.name or format_selection_label(fold_epochs)
    output_dir = args.output_dir or predictions_dir

    y_true, y_scores, folds, source_files = load_custom_oof(
        predictions_dir,
        fold_epochs,
    )

    print("Custom OOF Predictions:")
    for fold, epoch in fold_epochs:
        print(f"- fold_{fold}: epoch_{epoch}")

    calibration = None
    if args.calibration_ratio is not None:
        y_scores, calibration = calibrate_scores_to_ratio(
            y_true,
            y_scores,
            args.calibration_ratio,
        )
        print(
            "Applied prior logit-shift calibration: "
            f"target_ratio={calibration['target_ratio']}, "
            f"logit_offset={calibration['logit_offset']:.6f}"
        )

    metrics = evaluate(
        y_true=y_true,
        y_scores=y_scores,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print("Evaluation Metrics:")
    print("Base metrics:")
    for metric_name, metric_value in metrics["base_metrics"].items():
        print(f"- {metric_name}: {metric_value:.4f}")
    print("Bootstrap metrics GC:")
    for metric_name, metric_value in metrics["bootstrap_metrics_v2"].items():
        print(f"- {metric_name}: {metric_value:.4f}")

    save_results(
        output_dir=output_dir,
        label=label,
        fold_epochs=fold_epochs,
        y_true=y_true,
        y_scores=y_scores,
        folds=folds,
        metrics=metrics,
        source_files=source_files,
        calibration=calibration,
    )


if __name__ == "__main__":
    main()
