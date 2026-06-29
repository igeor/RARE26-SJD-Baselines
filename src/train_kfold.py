import os
import csv
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import RareDataset, get_class_counts
from utils.split import split_kfold_dataset
from utils.dataloader import create_dataloaders
from utils.seed import seed_everything
from utils.io import (
    save_config, 
    save_metrics_json, 
    save_predictions_npz, 
    save_full_predictions_npz, 
    save_best_full_predictions_npz
)
from utils.config import get_early_stopping_config
from models import build_model
from optimizer import build_optimizer
from evaluate_kfold import evaluate
from transforms import TrainRare26Transform, ValidationRare26Transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=total_loss / (pbar.n + 1),
                accuracy=correct / total,
            )

    return total_loss / len(dataloader), correct / total


def get_n_bootstrap(args):
    metrics = getattr(args, "metrics", None)
    if metrics is not None:
        return metrics.get("n_bootstrap", 1000)

    validate = getattr(args, "validate", None)
    if validate is not None:
        return validate.get("n_bootstrap", 1000)

    return 1000


def get_bootstrap_v2_metric(metrics_dict, metric_name):
    bootstrap_v2 = metrics_dict["bootstrap_metrics_v2"]
    aliases = {
        "PPV@90% Recall": "PPV@90RECALL",
        "PPV@90 Recall": "PPV@90RECALL",
        "PPV@90RECALL": "PPV@90RECALL",
        "PPV@90% Recall 95% CI Lower Bound": "PPV@90RECALL 95% CI Lower Bound",
        "PPV@90RECALL 95% CI Lower Bound": "PPV@90RECALL 95% CI Lower Bound",
        "PPV@90% Recall 95% CI Upper Bound": "PPV@90RECALL 95% CI Upper Bound",
        "PPV@90RECALL 95% CI Upper Bound": "PPV@90RECALL 95% CI Upper Bound",
        "AUROC": "AUROC",
        "AUPRC": "AUPRC",
        "Score": "Score",
    }
    key = aliases.get(metric_name, metric_name)

    if key not in bootstrap_v2:
        available = ", ".join(sorted(bootstrap_v2.keys()))
        raise KeyError(
            f"Metric '{metric_name}' resolved to '{key}', but it is not in "
            f"bootstrap_metrics_v2. Available metrics: {available}"
        )

    return float(bootstrap_v2[key]), key


def is_improvement(metric_value, best_metric_value, metric_name):
    if best_metric_value is None:
        return True

    if metric_name.lower() in {"loss", "val_loss"}:
        return metric_value < best_metric_value

    return metric_value > best_metric_value


def build_classification_criterion(args, train_counts, device):
    loss_pos_weight = getattr(args, "loss_pos_weight", None)

    if loss_pos_weight is None:
        print("Using BCEWithLogitsLoss without positive-class weighting.")
        return nn.BCEWithLogitsLoss()

    if isinstance(loss_pos_weight, bool):
        if not loss_pos_weight:
            print("Using BCEWithLogitsLoss without positive-class weighting.")
            return nn.BCEWithLogitsLoss()
        raise ValueError("loss_pos_weight=true is ambiguous; use 'auto' or a positive number.")

    if isinstance(loss_pos_weight, str):
        loss_pos_weight = loss_pos_weight.strip().lower()
        if loss_pos_weight in {"", "none", "null", "false"}:
            print("Using BCEWithLogitsLoss without positive-class weighting.")
            return nn.BCEWithLogitsLoss()

        if loss_pos_weight == "auto":
            n_positive = train_counts["neoplasia"]
            n_negative = train_counts["nondysplastic"]
            if n_positive == 0:
                raise ValueError("Cannot use loss_pos_weight='auto' with zero neoplasia samples.")
            loss_pos_weight = n_negative / n_positive
        else:
            loss_pos_weight = float(loss_pos_weight)

    loss_pos_weight = float(loss_pos_weight)
    if loss_pos_weight <= 0:
        raise ValueError(f"loss_pos_weight must be positive, got {loss_pos_weight}.")

    pos_weight = torch.tensor([loss_pos_weight], dtype=torch.float32, device=device)
    print(f"Using BCEWithLogitsLoss with pos_weight={loss_pos_weight:.4f}.")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    
def main(args):
    seed_everything(args.seed)
    early_stopping = get_early_stopping_config(args)
    eval_every = early_stopping["every"]
    patience = early_stopping["patience"]
    if patience is not None:
        patience = int(patience)
    n_bootstrap = get_n_bootstrap(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_path) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    save_config(args, output_dir)

    data_train_path = os.path.join(args.data_path, "train")
    print("Loading data from:", data_train_path)

    base_dataset = RareDataset(data_train_path, transform=None)
    folds = split_kfold_dataset(base_dataset, k=args.k, seed=args.seed)

    train_transform = TrainRare26Transform(image_size=args.image_size)
    val_transform = ValidationRare26Transform(image_size=args.image_size)
    evaluated_epochs_by_fold = {}
    best_epochs_by_fold = {}

    for fold_idx, (train_subset, val_subset) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx}/{args.k} ===")

        train_dataset = Subset(
            RareDataset(data_train_path, transform=train_transform), 
            train_subset.indices
        )
        val_dataset = Subset(
            RareDataset(data_train_path, transform=val_transform),
            val_subset.indices
        )

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            args,
        )

        train_counts = get_class_counts(train_dataset)
        neo = train_counts["neoplasia"]
        ndbe = train_counts["nondysplastic"]
        print(
            f"Train class distribution: "
            f"Neoplasia={neo}, NonDysplastic={ndbe}, Ratio={neo / ndbe:.2f}"
        )

        val_counts = get_class_counts(val_dataset)
        neo = val_counts["neoplasia"]
        ndbe = val_counts["nondysplastic"]
        print(
            f"Val class distribution: "
            f"Neoplasia={neo}, NonDysplastic={ndbe}, Ratio={neo / ndbe:.2f}"
        )

        model = build_model(args, device)

        criterion = build_classification_criterion(args, train_counts, device)

        optimizer = build_optimizer(model, args)
        best_metric_value = None
        best_epoch = None
        epochs_without_improvement = 0
        evaluated_epochs_by_fold[fold_idx] = []

        for epoch in range(1, args.epochs + 1):
            print(f"\nFold {fold_idx} | Epoch {epoch}/{args.epochs}")

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )


            should_evaluate = (
                (eval_every > 0 and epoch % eval_every == 0)
                or epoch == args.epochs
            )

            if should_evaluate:
                val_metrics, y_val_true, y_val_scores = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    n_bootstrap=n_bootstrap,
                    seed=args.seed + fold_idx,
                    return_predictions=True,
                )
                evaluated_epochs_by_fold[fold_idx].append(epoch)

                # val_metrics example structure:
                # {'Loss': 0.11858299374580383, 'Accuracy': 0.9741518578352181, 
                # 'base_metrics': {'AUROC': 0.9422913117546848, 'AUPRC': 0.7517970258860072, 'PPV@90% Recall': np.float64(0.19333333333333333), 'Accuracy': np.float64(0.9741518578352181), 'Sensitivity': np.float64(0.5), 'Specificity': np.float64(1.0)}, 
                # 'bootstrap_metrics': {'AUROC': {'median': 0.94423976, 'ci_lower': 0.9441232235, 'ci_upper': 0.9455229915000001}, 'AUPRC': {'median': 0.6542209737297922, 'ci_lower': 0.6326753675180588, 'ci_upper': 0.6547025779143271}, 'PPV@90% Recall': {'median': 0.04229677539435112, 'ci_lower': 0.04193101524030579, 'ci_upper': 0.042384758072194134}, '' 'Accuracy': {'median': 0.9952574257425743, 'ci_lower': 0.9950128712871287, 'ci_upper': 0.9953420792079208}, 'Sensitivity': {'median': 0.521, 'ci_lower': 0.4963, 'ci_upper': 0.5295500000000001}, 'Specificity': {'median': 1.0, 'ci_lower': 1.0, 'ci_upper': 1.0}}, 
                # 'bootstrap_metrics_v2': {'Score': np.float64(0.10830429732868757), 'PPV@90RECALL': np.float64(0.10830429732868757), 'PPV@90RECALL 95% CI Lower Bound': np.float64(0.03946442121564073), 'PPV@90RECALL 95% CI Upper Bound': np.float64(0.5737187862950057), 'AUROC': np.float64(0.9873935264054515), 'AUROC 95% CI Lower Bound': np.float64(0.9592333901192504), 'AUROC 95% CI Upper Bound': np.float64(0.997427597955707), 'AUPRC': np.float64(0.8238095238095239), 'AUPRC 95% CI Lower Bound': np.float64(0.7273015873015873), 'AUPRC 95% CI Upper Bound': np.float64(0.8385119047619047), 'AUROC Full Dataset': 0.9422913117546848, 'AUPRC Full Dataset': 0.7517970258860072, 'PPV@90RECALL Full Dataset': np.float64(0.19225055928411633)}
                # }

                save_metrics_json(
                    val_metrics,
                    fold_dir / f"epoch_{epoch}_val_metrics.json",
                )


                save_predictions_npz(
                    y_val_true,
                    y_val_scores,
                    fold_dir / f"epoch_{epoch}_val_predictions.npz",
                )
                
                monitored_value, monitored_key = get_bootstrap_v2_metric(
                    val_metrics,
                    early_stopping["metric"],
                )

                if not early_stopping["enabled"]:
                    torch.save(
                        {
                            "fold": fold_idx,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                        },
                        fold_dir / "last_checkpoint.pth",
                    )

                if is_improvement(
                    monitored_value,
                    best_metric_value,
                    early_stopping["metric"],
                ):
                    best_metric_value = monitored_value
                    best_epoch = epoch
                    best_epochs_by_fold[fold_idx] = epoch
                    epochs_without_improvement = 0

                    if early_stopping["enabled"]:
                        torch.save(
                            {
                                "fold": fold_idx,
                                "epoch": epoch,
                                "best_metric": monitored_key,
                                "best_metric_value": monitored_value,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "args": vars(args),
                            },
                            fold_dir / "best_checkpoint.pth",
                        )
                else:
                    epochs_without_improvement += 1

                print(
                    f"Fold {fold_idx} | Epoch {epoch}/{args.epochs} | "
                    f"Train Loss: {train_loss:.3f} | "
                    f"Train Acc: {train_acc:.3f} | "
                    f"PPV@90% Recall (base): {val_metrics['base_metrics']['PPV@90% Recall']:.3f} | "
                    f"PPV@90% Recall (bootstrap v1): {val_metrics['bootstrap_metrics']['PPV@90% Recall']['ci_lower']:.3f} | "
                    f"PPV@90% Recall (bootstrap v2 median): {val_metrics['bootstrap_metrics_v2']['PPV@90RECALL']:.3f} | "
                    f"PPV@90% Recall (bootstrap v2 CI lower): {val_metrics['bootstrap_metrics_v2']['PPV@90RECALL 95% CI Lower Bound']:.3f} | "
                    f"Early-stop metric ({monitored_key}): {monitored_value:.3f} | "
                    f"Best Epoch: {best_epoch}"
                )

                if (
                    early_stopping["enabled"]
                    and patience is not None
                    and epochs_without_improvement >= patience
                ):
                    print(
                        f"Early stopping fold {fold_idx} at epoch {epoch}: "
                        f"no improvement in {monitored_key} for "
                        f"{epochs_without_improvement} evaluation(s)."
                    )
                    break

            else:
                print(
                    f"Fold {fold_idx} | Epoch {epoch}/{args.epochs} | "
                    f"Train Loss: {train_loss:.3f} | "
                    f"Train Acc: {train_acc:.3f}"
                )
       
    if eval_every > 0:
        saved_full_prediction_files = []
        common_epochs = set(evaluated_epochs_by_fold.get(1, []))
        for epochs in evaluated_epochs_by_fold.values():
            common_epochs &= set(epochs)

        for epoch in sorted(common_epochs):
            save_path = save_full_predictions_npz(output_dir, epoch)
            if save_path is not None:
                saved_full_prediction_files.append(save_path)

        print(
            f"Exported {len(saved_full_prediction_files)} combined full-prediction "
            f"file(s) for early_stopping.every={eval_every}."
        )

    best_predictions_path = None
    if early_stopping["enabled"] and best_epochs_by_fold:
        best_predictions_path = save_best_full_predictions_npz(
            output_dir,
            best_epochs_by_fold,
        )

    # Evaluate the trained model by invoking the evaluation script directly
    eval_model = (
        "best"
        if early_stopping["enabled"] and best_predictions_path is not None
        else "last"
    )
    eval_epoch = args.epochs
    if eval_model == "last" and eval_every > 0:
        common_epochs = set(evaluated_epochs_by_fold.get(1, []))
        for epochs in evaluated_epochs_by_fold.values():
            common_epochs &= set(epochs)
        if common_epochs:
            eval_epoch = max(common_epochs)

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_fullset.py",
            "--predictions_dir", str(output_dir),
            "--epoch", str(eval_epoch),
            "--eval_model", eval_model,
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold training script from YAML config"
    )

    parser.add_argument(
        "--config",
        default="configs/sanity_check.yaml",
        help="Path to YAML config file.",
    )

    cli_args = parser.parse_args()

    with open(Path(cli_args.config), "r", encoding="utf-8") as f:
        args = SimpleNamespace(**yaml.safe_load(f))

    return args


if __name__ == "__main__":
    main(parse_args())
