# Load an .npz file
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_kfold import evaluate


def load_npz_file(file_path):
    data = np.load(file_path)
    y_true = data['y_true']
    y_scores = data['y_scores']
    folds = data["folds"] if "folds" in data else None
    return y_true, y_scores, folds


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


def validate_epochs(predictions_dirs, epochs):
    if len(predictions_dirs) == 1 and len(epochs) == 1:
        return epochs

    if len(predictions_dirs) > 1 and len(epochs) != len(predictions_dirs):
        raise ValueError(
            "When passing multiple prediction directories, provide exactly one "
            "--epoch value per directory."
        )

    if len(predictions_dirs) == 1 and len(epochs) != 1:
        raise ValueError("A single prediction directory requires exactly one epoch.")

    return epochs


def format_epoch_label(epochs):
    if len(epochs) == 1:
        return f"epoch_{epochs[0]}"
    return "epochs_" + "_".join(str(epoch) for epoch in epochs)


def load_prediction_ensemble(predictions_dirs, epochs):
    y_true_reference = None
    folds_reference = None
    score_sets = []
    loaded_paths = []

    for predictions_dir, epoch in zip(predictions_dirs, epochs):
        predictions_dir = Path(predictions_dir)
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Directory {predictions_dir} does not exist.")

        file_path = predictions_dir / f"epoch_{epoch}_full_predictions.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file {file_path} does not exist.")

        y_true, y_scores, folds = load_npz_file(file_path)
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_scores = np.asarray(y_scores).reshape(-1).astype(float)
        folds = None if folds is None else np.asarray(folds).reshape(-1)

        if y_true_reference is None:
            y_true_reference = y_true
            folds_reference = folds
        else:
            if y_true.shape != y_true_reference.shape or not np.array_equal(
                y_true, y_true_reference
            ):
                raise ValueError(
                    f"Labels/order in {file_path} do not match the first input."
                )
            if folds_reference is not None and folds is not None:
                if folds.shape != folds_reference.shape or not np.array_equal(
                    folds, folds_reference
                ):
                    raise ValueError(
                        f"Fold ids in {file_path} do not match the first input."
                    )

        score_sets.append(y_scores)
        loaded_paths.append(file_path)

    y_scores_ensemble = np.mean(np.stack(score_sets, axis=0), axis=0)
    return y_true_reference, y_scores_ensemble, folds_reference, loaded_paths


def load_best_prediction_ensemble(predictions_dirs):
    y_true_reference = None
    folds_reference = None
    score_sets = []
    loaded_paths = []

    for predictions_dir in predictions_dirs:
        predictions_dir = Path(predictions_dir)
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Directory {predictions_dir} does not exist.")

        file_path = predictions_dir / "best_full_predictions.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file {file_path} does not exist.")

        y_true, y_scores, folds = load_npz_file(file_path)
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_scores = np.asarray(y_scores).reshape(-1).astype(float)
        folds = None if folds is None else np.asarray(folds).reshape(-1)

        if y_true_reference is None:
            y_true_reference = y_true
            folds_reference = folds
        else:
            if y_true.shape != y_true_reference.shape or not np.array_equal(
                y_true, y_true_reference
            ):
                raise ValueError(
                    f"Labels/order in {file_path} do not match the first input."
                )
            if folds_reference is not None and folds is not None:
                if folds.shape != folds_reference.shape or not np.array_equal(
                    folds, folds_reference
                ):
                    raise ValueError(
                        f"Fold ids in {file_path} do not match the first input."
                    )

        score_sets.append(y_scores)
        loaded_paths.append(file_path)

    y_scores_ensemble = np.mean(np.stack(score_sets, axis=0), axis=0)
    return y_true_reference, y_scores_ensemble, folds_reference, loaded_paths


def save_results(output_dir, epochs, y_true, y_scores, folds, metrics, input_paths, eval_model):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_label = "best" if eval_model == "best" else format_epoch_label(epochs)
    predictions_path = output_dir / f"{epoch_label}_ensemble_predictions.npz"
    if folds is None:
        np.savez_compressed(predictions_path, y_true=y_true, y_scores=y_scores)
    else:
        np.savez_compressed(
            predictions_path,
            y_true=y_true,
            y_scores=y_scores,
            folds=folds,
        )

    metrics_payload = {
        "epochs": epochs,
        "eval_model": eval_model,
        "input_prediction_files": [str(path) for path in input_paths],
        "metrics": metrics,
    }
    metrics_json_path = output_dir / f"{epoch_label}_ensemble_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(metrics_payload), f, indent=2)

    metrics_csv_path = output_dir / f"{epoch_label}_ensemble_metrics.csv"
    flat_metrics = flatten_metrics(to_serializable(metrics))
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
        writer.writeheader()
        writer.writerow(flat_metrics)

    print(f"Saved ensemble predictions to: {predictions_path}")
    print(f"Saved ensemble metrics JSON to: {metrics_json_path}")
    print(f"Saved ensemble metrics CSV to: {metrics_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_dir",
        "--predictions-dir",
        dest="predictions_dirs",
        nargs="+",
        required=True,
        help=(
            "One or more directories containing epoch_<N>_full_predictions.npz. "
            "When multiple directories are given, scores are averaged."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        nargs="+",
        default=[15],
        help=(
            "Epoch number(s) to evaluate. Provide one value for single-dir use, "
            "or one epoch per predictions directory for ensembling."
        ),
    )
    parser.add_argument("--eval_model", type=str, default="last", help="Evaluation model to use (e.g., 'last', 'best')")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="Directory to export ensemble predictions and metrics.",
    )
    args = parser.parse_args()

    if args.eval_model not in ["last", "best"]:
        print(f"Error: Invalid eval_model '{args.eval_model}'. Must be 'last' or 'best'. Using default: 'last'.")
        args.eval_model = "last"

    predictions_dirs = [Path(path) for path in args.predictions_dirs]

    # Config in output dir
    config_path = predictions_dirs[0] / "config.txt"
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml.safe_load(f)

    try:
        if args.eval_model == "best":
            epochs = args.epoch
            y_true, y_scores, folds, input_paths = load_best_prediction_ensemble(
                predictions_dirs,
            )
        else:
            epochs = validate_epochs(predictions_dirs, args.epoch)
            y_true, y_scores, folds, input_paths = load_prediction_ensemble(
                predictions_dirs,
                epochs,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if len(predictions_dirs) == 1:
        print("OOF Predictions:")
    else:
        print(f"OOF Ensemble Predictions ({len(predictions_dirs)} runs, mean aggregation):")
    y_true_oof = np.asarray(y_true).reshape(-1).astype(int)
    y_scores_oof = np.asarray(y_scores).reshape(-1).astype(float)
    metrics = evaluate(y_true=y_true_oof, y_scores=y_scores_oof, n_bootstrap=1000, seed=42)
    print("Evaluation Metrics:")
    print("Base metrics:")
    for metric_name, metric_value in metrics["base_metrics"].items():
        print(f"- {metric_name}: {metric_value:.4f}")
    print("Bootstrap metrics:")
    for metric_name, metric_set in metrics["bootstrap_metrics"].items():
        for metric_name_inner, metric_value in metric_set.items():
            print(f"- {metric_name} - {metric_name_inner}: {metric_value:.4f}")
    print("Bootstrap metrics GC:")
    for metric_name, metric_value in metrics["bootstrap_metrics_v2"].items():
        print(f"- {metric_name}: {metric_value:.4f}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif len(predictions_dirs) == 1:
        output_dir = predictions_dirs[0]
    else:
        output_dir = predictions_dirs[0].parent / f"ensemble_{format_epoch_label(epochs)}"

    save_results(
        output_dir=output_dir,
        epochs=epochs,
        y_true=y_true_oof,
        y_scores=y_scores_oof,
        folds=folds,
        metrics=metrics,
        input_paths=input_paths,
        eval_model=args.eval_model,
    )
