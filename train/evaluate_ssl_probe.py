import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm

from metrics import compute_metrics

from models import ClassificationHead
from utils.io import save_val_metrics


def bootstrap_evaluation(
    y_true,
    y_pred,
    n_bootstrap=1000,
    min_neoplasia=1000,
    ndbe_multiplier=100,
    output_dir=None,
    prefix="bootstrap",
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    results = []

    neoplasia_indices = np.where(y_true == 1)[0]
    ndbe_indices = np.where(y_true == 0)[0]

    for _ in range(n_bootstrap):
        neoplasia_sample = np.random.choice(
            neoplasia_indices, size=min_neoplasia, replace=True
        )
        ndbe_sample = np.random.choice(
            ndbe_indices, size=min_neoplasia * ndbe_multiplier, replace=True
        )

        sample_indices = np.concatenate([neoplasia_sample, ndbe_sample])

        metrics = compute_metrics(y_true[sample_indices], y_pred[sample_indices])
        results.append(metrics)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)

    # Compute median and 95% CI
    summary_df = metrics_df.describe(percentiles=[0.025, 0.5, 0.975]).loc[
        ["2.5%", "50%", "97.5%"]
    ]

    # Save if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_dir / f"{prefix}_bootstrap_raw.csv", index=False)
        summary_df.to_csv(output_dir / f"{prefix}_bootstrap_summary.csv")

        print(
            f"Saved raw bootstrap metrics to: {output_dir / f'{prefix}_bootstrap_raw.csv'}"
        )
        print(
            f"Saved summary statistics to: {output_dir / f'{prefix}_bootstrap_summary.csv'}"
        )

    return summary_df


def evaluate_ssl_model(
    folds,
    student,
    device,
    args,
    global_step: int = 0,
):
    avg_val_metrics = {}

    probe_batch_size = getattr(args, "probe_batch_size", args.batch_size)
    total_batches = 0

    for train_subset, val_subset in folds:
        total_batches += math.ceil(len(train_subset) / probe_batch_size)
        total_batches += math.ceil(len(val_subset) / probe_batch_size)

    student.eval()

    with tqdm(total=total_batches, desc="SSL probe eval", unit="batch") as pbar:
        for fold_idx, (train_subset, val_subset) in enumerate(folds, start=1):

            # Set a unique seed for the probe training to ensure reproducibility across folds.
            probe_seed = getattr(args, "probe_seed", args.seed) + fold_idx
            probe_generator = torch.Generator()
            probe_generator.manual_seed(probe_seed)

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(probe_seed)
                fc = ClassificationHead(in_dim=student.hidden_size).to(device)

            fc_optimizer = torch.optim.AdamW(
                fc.parameters(),
                lr=getattr(args, "probe_lr", args.lr),
                weight_decay=args.probe_weight_decay,
            )
            fc_criterion = nn.BCEWithLogitsLoss()

            fc_train_loader = DataLoader(
                train_subset,
                batch_size=probe_batch_size,
                shuffle=True,
                generator=probe_generator,
            )
            fc_val_loader = DataLoader(
                val_subset,
                batch_size=probe_batch_size,
                shuffle=False,
            )

            fc.train()
            for image, label in fc_train_loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).float().unsqueeze(1)

                with torch.no_grad():
                    features = student.extract_features(image)

                fc_optimizer.zero_grad(set_to_none=True)
                logits = fc(features)
                loss = fc_criterion(logits, label)
                loss.backward()
                fc_optimizer.step()

                pbar.update(1)
                pbar.set_postfix(fold=fold_idx, phase="train")

            fc.eval()
            y_true, y_pred = [], []

            with torch.no_grad():
                for image, label in fc_val_loader:
                    image = image.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True).float().unsqueeze(1)

                    features = student.extract_features(image)
                    logits = fc(features)

                    y_true.extend(label.cpu().numpy().ravel())
                    y_pred.extend(torch.sigmoid(logits).cpu().numpy().ravel())

                    pbar.update(1)
                    pbar.set_postfix(fold=fold_idx, phase="val")

            metrics = compute_metrics(np.array(y_true), np.array(y_pred))
            avg_val_metrics[fold_idx] = metrics

    aggregated_metrics = {}
    for metric_name in avg_val_metrics[1].keys():
        metric_values = [avg_val_metrics[fold][metric_name] for fold in avg_val_metrics]
        aggregated_metrics[metric_name] = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }

    if getattr(args, "save_dir", None):
        save_val_metrics(Path(args.save_dir), aggregated_metrics, global_step)

    return aggregated_metrics
