import torch
import csv
from torch import nn
from pathlib import Path
import pandas as pd
from dataclasses import asdict
import json
import numpy as np


def save_checkpoint(
    save_dir: Path,
    *,
    epoch: int,
    batch_idx: int,
    global_step: int,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "criterion": criterion.state_dict(),
        "config": asdict(config),
    }
    checkpoint_path = save_dir / f"checkpoint_step{global_step}.pt"
    tmp_checkpoint_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, tmp_checkpoint_path)
    tmp_checkpoint_path.replace(checkpoint_path)

    inference = {
        "model": teacher.state_dict(),
        "model_name": config.model,
        "out_dim": config.out_dim,
        "image_size": config.image_size,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target": config.lora_target,
        "global_step": global_step,
        "epoch": epoch,
    }
    inference_path = save_dir / f"inference_step{global_step}.pt"
    tmp_inference_path = inference_path.with_suffix(inference_path.suffix + ".tmp")
    torch.save(inference, tmp_inference_path)
    tmp_inference_path.replace(inference_path)

    print(f"---Saved inference checkpoint: {save_dir / f'inference_step{global_step}.pt'}")
    print(f"---Saved training checkpoint:  {save_dir / f'checkpoint_step{global_step}.pt'}")


def save_val_metrics(save_dir: Path, val_metrics: dict, global_step: int) -> None:
    row = {"global_step": global_step}
    for metric_name, stats in val_metrics.items():
        for stat_name, value in stats.items():
            row[f"{metric_name}_{stat_name}"] = value

    csv_path = save_dir / f"val_metrics_step{global_step}.csv"
    tmp_csv_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df = pd.DataFrame([row])

    df.to_csv(tmp_csv_path, index=False)
    tmp_csv_path.replace(csv_path)

    print(f"---Saved validation metrics CSV: {csv_path}")



def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def save_metrics_json(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(metrics), f, indent=2)


def save_predictions_npz(y_true, y_scores, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        y_true=np.asarray(y_true).reshape(-1),
        y_scores=np.asarray(y_scores).reshape(-1),
    )


def save_full_predictions_npz(output_dir, epoch):
    """Combine validation predictions from all folds for one epoch."""
    output_dir = Path(output_dir)

    prediction_files = sorted(
        output_dir.glob(f"fold_*/epoch_{epoch}_val_predictions.npz")
    )

    if not prediction_files:
        print(f"No prediction files found for epoch {epoch}")
        return None

    all_y_true = []
    all_y_scores = []
    fold_ids = []

    for path in prediction_files:
        data = np.load(path)

        all_y_true.append(data["y_true"])
        all_y_scores.append(data["y_scores"])

        fold_name = path.parent.name  # e.g. fold_1
        fold_idx = int(fold_name.split("_")[1])
        fold_ids.append(
            np.full(shape=data["y_true"].shape, fill_value=fold_idx)
        )

    y_true = np.concatenate(all_y_true)
    y_scores = np.concatenate(all_y_scores)
    folds = np.concatenate(fold_ids)

    save_path = output_dir / f"epoch_{epoch}_full_predictions.npz"

    np.savez_compressed(
        save_path,
        y_true=y_true,
        y_scores=y_scores,
        folds=folds,
    )

    print(f"Saved combined predictions to: {save_path}")
    return save_path


def save_best_full_predictions_npz(output_dir, best_epochs_by_fold):
    """Combine validation predictions from each fold's best epoch."""
    output_dir = Path(output_dir)

    all_y_true = []
    all_y_scores = []
    fold_ids = []
    source_files = []

    for fold_idx in sorted(best_epochs_by_fold):
        epoch = best_epochs_by_fold[fold_idx]
        path = output_dir / f"fold_{fold_idx}" / f"epoch_{epoch}_val_predictions.npz"

        if not path.exists():
            print(f"Missing best prediction file for fold {fold_idx}: {path}")
            return None

        data = np.load(path)
        all_y_true.append(data["y_true"])
        all_y_scores.append(data["y_scores"])
        fold_ids.append(
            np.full(shape=data["y_true"].shape, fill_value=fold_idx)
        )
        source_files.append(str(path))

    y_true = np.concatenate(all_y_true)
    y_scores = np.concatenate(all_y_scores)
    folds = np.concatenate(fold_ids)

    save_path = output_dir / "best_full_predictions.npz"

    np.savez_compressed(
        save_path,
        y_true=y_true,
        y_scores=y_scores,
        folds=folds,
        best_epochs_by_fold=np.array(
            [best_epochs_by_fold[fold_idx] for fold_idx in sorted(best_epochs_by_fold)]
        ),
        source_files=np.array(source_files),
    )

    print(f"Saved best combined predictions to: {save_path}")
    return save_path


def flatten_metrics(metrics):
    flat = {}

    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in flatten_metrics(value).items():
                flat[f"{key}.{subkey}"] = subvalue
        else:
            flat[key] = value

    return flat


def save_summary_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat_rows = [flatten_metrics(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def save_config(args, output_dir):
    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")