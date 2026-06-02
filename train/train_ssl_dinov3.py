#!/usr/bin/env python3
"""
train_dinov3_gastronet5m_ssl_lora_multicrop.py

Runnable DINOv3-inspired self-supervised domain adaptation for GastroNet5M.

This version keeps the complete training/checkpoint structure from script 1 and
adds the stronger DINO ideas from script 2:
  - multi-crop augmentation
  - centered teacher distribution
  - teacher predicts only global crops
  - student predicts global + local crops
  - EMA teacher update
  - frozen DINOv3 backbone with trainable LoRA adapters + DINO projection head
"""

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seed import seed_everything
from utils.module import freeze_module, count_parameters, ema_update
from utils.schedule import cosine_ema_momentum, linear_warmup_value
from utils.split import split_kfold_dataset
from utils.batch import move_crops_to_device
from utils.io import save_checkpoint

from dataset import GastroNetZipDataset, RareDataset

from transforms import GastroDINOv3MultiCropTransform, ValidationRare26Transform

from models import DINOStudentTeacherModel

from criterion import DINOCenteredLoss

from evaluate_ssl_probe import evaluate_ssl_model


@dataclass
class TrainConfig:
    data: str
    model: str
    image_size: int
    epochs: int
    batch_size: int
    accum_steps: int
    lr: float
    weight_decay: float
    out_dim: int
    workers: int
    save_dir: str
    save_every: int
    val_data: Optional[str]
    val_every_steps: int
    seed: int
    amp: bool
    max_images: Optional[int]
    grad_clip: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target: List[str]
    local_crops_number: int
    global_crop_scale: Tuple[float, float]
    local_crop_scale: Tuple[float, float]
    strong_color_jitter: bool
    student_temp: float
    teacher_temp: float
    teacher_temp_warmup: float
    teacher_temp_warmup_epochs: int
    center_momentum: float
    ema_base: float
    k_folds: int
    probe_epochs: int
    probe_lr: float
    probe_weight_decay: float
    probe_batch_size: int


def train(args) -> None:
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        data=args.data,
        model=args.model,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dim=args.out_dim,
        workers=args.workers,
        save_dir=args.save_dir,
        save_every=args.save_every,
        val_data=args.val_data,
        val_every_steps=args.val_every_steps,
        seed=args.seed,
        amp=use_amp,
        max_images=args.max_images,
        grad_clip=args.grad_clip,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        local_crops_number=args.local_crops_number,
        global_crop_scale=tuple(args.global_crop_scale),
        local_crop_scale=tuple(args.local_crop_scale),
        strong_color_jitter=args.strong_color_jitter,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        teacher_temp_warmup=args.teacher_temp_warmup,
        teacher_temp_warmup_epochs=args.teacher_temp_warmup_epochs,
        center_momentum=args.center_momentum,
        ema_base=args.ema_base,
        k_folds=getattr(args, "k_folds", 5),
        probe_epochs=getattr(args, "probe_epochs", 5),
        probe_lr=getattr(args, "probe_lr", 1e-3),
        probe_weight_decay=getattr(args, "probe_weight_decay", 0.0),
        probe_batch_size=getattr(args, "probe_batch_size", args.batch_size),
    )

    # Store configuration in the save directory for reproducibility.
    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    train_transform = GastroDINOv3MultiCropTransform(
        image_size=args.image_size,
        local_crops_number=args.local_crops_number,
        global_crop_scale=tuple(args.global_crop_scale),
        local_crop_scale=tuple(args.local_crop_scale),
        strong_color_jitter=args.strong_color_jitter,
    )

    train_dataset = GastroNetZipDataset(root=args.data, transform=train_transform, max_images=args.max_images)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )

    print(f"---Loaded training dataset with {len(train_dataset):,} indexed images.")
    print(f"---Batches per epoch: {len(train_loader):,}")
    print(f"---Crops per image: {2 + args.local_crops_number} = 2 global + {args.local_crops_number} local")
    print(f"---Device: {device}")
    print(f"---AMP enabled: {use_amp}")

    student = DINOStudentTeacherModel(
        model_name=args.model,
        out_dim=args.out_dim,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_target,
    ).to(device)

    teacher = DINOStudentTeacherModel(
        model_name=args.model,
        out_dim=args.out_dim,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_target,
    ).to(device)
    
    # Initialize teacher with student weights, then freeze the teacher. 
    # The teacher will be updated as an EMA of the student during training. 
    # Does not need its own optimizer or gradients.
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    freeze_module(teacher)

    criterion = DINOCenteredLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp_warmup if args.teacher_temp_warmup_epochs > 0 else args.teacher_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    total, trainable = count_parameters(student)
    print(f"---LoRA layers found: {len(student.lora_layers)}")
    print(f"---First LoRA layers: {student.lora_layers[:10]}")
    print(f"---Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_optimizer_steps = args.epochs * steps_per_epoch
    warmup_steps = args.teacher_temp_warmup_epochs * steps_per_epoch

    global_step = 0
    last_val_step = 0

    # Initialize validation dataset.
    val_transform = ValidationRare26Transform(image_size=args.image_size)
    dataset = RareDataset(root_dir=args.val_data, transform=val_transform)
    print(f"---Loaded validation dataset with {len(dataset):,} images.")

    # Create K-fold splits for the validation dataset to evaluate the SSL probe more robustly.
    folds = split_kfold_dataset(dataset, k=getattr(args, "k_folds", 5), seed=args.seed)
    print(f"---Created {len(folds)} folds for cross-validation.")
    
    # Evaluate the pretrained teacher model on the validation set before training to get a baseline.
    dinov2_val_metrics = evaluate_ssl_model(
        folds,
        student=teacher,  # Use the teacher for evaluation since it's the EMA of the student
        device=device,
        args=args,
        global_step=global_step,
    )
    print("---SSL probe validation metrics:")
    for metric_name, stats in dinov2_val_metrics.items():
        print(
            f"---  {metric_name:<18} "
            f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
        )


    # Train for the specified number of epochs, evaluating on the validation set every val_every_steps.
    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch_idx, crops in enumerate(pbar, start=1):
            crops = move_crops_to_device(crops, device)

            teacher_temp = linear_warmup_value(
                step=global_step,
                warmup_steps=warmup_steps,
                start=args.teacher_temp_warmup,
                end=args.teacher_temp,
            )
            criterion.set_teacher_temp(teacher_temp)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Student sees all crops. Teacher sees only the two global crops.
                student_outputs = [student(crop) for crop in crops]
                with torch.no_grad():
                    teacher_outputs = [teacher(crop) for crop in crops[:2]]

                loss = criterion(student_outputs, teacher_outputs)
                loss_for_backward = loss / args.accum_steps

            scaler.scale(loss_for_backward).backward()
            running_loss += loss.detach().item()

            # Update weights and teacher EMA every accum_steps or at the end of the epoch.
            do_optimizer_step = (batch_idx % args.accum_steps == 0 or batch_idx == len(train_loader))
            if do_optimizer_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                momentum = cosine_ema_momentum(
                    step=global_step,
                    total_steps=total_optimizer_steps,
                    base_momentum=args.ema_base,
                    final_momentum=1.0,
                )
                ema_update(student, teacher, momentum)
                global_step += 1

                pbar.set_postfix(
                    loss=f"{running_loss / batch_idx:.4f}",
                    ema=f"{momentum:.5f}",
                    t_temp=f"{teacher_temp:.4f}",
                    step=global_step,
                )

            # Evaluate on the validation set every val_every_steps.
            if (
                args.val_every_steps > 0
                and global_step > 0
                and global_step % args.val_every_steps == 0
                and global_step != last_val_step
            ):  
                dinov2_val_metrics = evaluate_ssl_model(
                    folds,
                    student=teacher,  # Use the teacher for evaluation since it's the EMA of the student
                    device=device,
                    args=args,
                    global_step=global_step,
                )
                print("---SSL probe validation metrics:")
                for metric_name, stats in dinov2_val_metrics.items():
                    print(
                        f"---  {metric_name:<18} "
                        f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
                    )
                save_checkpoint(
                    save_dir,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scaler=scaler,
                    criterion=criterion,
                    config=config,
                )
                print(f"---Saved latest checkpoint at step {global_step}: {save_dir / f'checkpoint_step{global_step}.pt'}")
                last_val_step = global_step


    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 -> GastroNet5M SSL LoRA adaptation from a YAML config")
    parser.add_argument("--config", default="configs/dinov3_ssl_gastronet5m.yaml", help="Path to YAML config file.")
    cli_args = parser.parse_args()

    with open(Path(cli_args.config), "r", encoding="utf-8") as f:
        args = SimpleNamespace(**yaml.safe_load(f))
    return args


if __name__ == "__main__":
    train(parse_args())
