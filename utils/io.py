import torch
from torch import nn
from pathlib import Path
import pandas as pd
from dataclasses import asdict

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
