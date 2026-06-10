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
import torch.optim as optim
from torch.utils.data import Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import RareDataset, get_class_counts
from utils.split import split_kfold_dataset
from utils.dataloader import create_dataloaders
from utils.seed import seed_everything
from utils.io import save_config, save_metrics_json, save_predictions_npz
from models import build_model
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



def metric_improved(val_metric_name, current_metric, best_metric):
    if val_metric_name == "Loss":
        return current_metric < best_metric
    return current_metric > best_metric


def main(args):
    seed_everything(args.seed)

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

    all_fold_results = []

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

        # fold_log_file = fold_dir / "training_log.csv"
        fold_model_path = fold_dir / "best_model.pth"

        # write_log_header(fold_log_file)

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

        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )

        val_metric_name = "PPV@90% Recall"
        if args.early_stopping["enabled"]:
            val_metric_name = args.early_stopping["metric"]
            

        best_metric = float("inf") if val_metric_name == "Loss" else float("-inf")
        best_epoch = None
        best_val_metrics = None


        for epoch in range(1, args.epochs + 1):
            print(f"\nFold {fold_idx} | Epoch {epoch}/{args.epochs}")

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_metrics, y_val_true, y_val_scores = evaluate(
                model,
                val_loader,
                criterion,
                device,
                n_bootstrap=args.validate["n_bootstrap"],
                seed=args.seed + fold_idx,
                return_predictions=True,
            )

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


            current_metric = val_metrics["base_metrics"][val_metric_name]

            if metric_improved(val_metric_name, current_metric, best_metric):
                best_epoch = epoch
                best_metric = current_metric
                best_val_metrics = val_metrics

                save_metrics_json(
                    best_val_metrics,
                    fold_dir / "best_val_metrics.json",
                )

                save_predictions_npz(
                    y_val_true,
                    y_val_scores,
                    fold_dir / "best_val_predictions.npz",
                )

                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "val_metric_name": val_metric_name,
                        "args": vars(args),
                    },
                    fold_model_path,
                )

                print(
                    f"Saved best model for fold {fold_idx} "
                    f"at epoch {epoch} with {val_metric_name}={best_metric:.4f}"
                )

            print(
                f"Fold {fold_idx} | Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.3f} | "
                f"Train Acc: {train_acc:.3f} | "
                f"PPV@90% Recall (base): {val_metrics['base_metrics']['PPV@90% Recall']:.3f} | "
                f"PPV@90% Recall (bootstrap v1): {val_metrics['bootstrap_metrics']['PPV@90% Recall']['ci_lower']:.3f} | "
                f"PPV@90% Recall (bootstrap v2): {val_metrics['bootstrap_metrics_v2']['PPV@90RECALL 95% CI Lower Bound']:.3f} | "
            )

        base = best_val_metrics["base_metrics"]
        boot = best_val_metrics["bootstrap_metrics"]
        boot_v2 = best_val_metrics["bootstrap_metrics_v2"]

        all_fold_results.append(
            {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_val_metric_name": val_metric_name,
                "best_metric": best_metric,

                "Loss": best_val_metrics["Loss"],

                "base_AUROC": base["AUROC"],
                "base_AUPRC": base["AUPRC"],
                "base_PPV@90% Recall": base["PPV@90% Recall"],
                "base_Accuracy": base["Accuracy"],
                "base_Sensitivity": base["Sensitivity"],
                "base_Specificity": base["Specificity"],

                "bootstrap_AUROC_median": boot["AUROC"]["median"],
                "bootstrap_AUROC_ci_lower": boot["AUROC"]["ci_lower"],
                "bootstrap_AUROC_ci_upper": boot["AUROC"]["ci_upper"],

                "bootstrap_AUPRC_median": boot["AUPRC"]["median"],
                "bootstrap_AUPRC_ci_lower": boot["AUPRC"]["ci_lower"],
                "bootstrap_AUPRC_ci_upper": boot["AUPRC"]["ci_upper"],

                "bootstrap_PPV@90% Recall_median": boot["PPV@90% Recall"]["median"],
                "bootstrap_PPV@90% Recall_ci_lower": boot["PPV@90% Recall"]["ci_lower"],
                "bootstrap_PPV@90% Recall_ci_upper": boot["PPV@90% Recall"]["ci_upper"],

                "bootstrap_v2_AUROC": boot_v2["AUROC"],
                "bootstrap_v2_AUROC_ci_lower": boot_v2["AUROC 95% CI Lower Bound"],
                "bootstrap_v2_AUROC_ci_upper": boot_v2["AUROC 95% CI Upper Bound"],

                "bootstrap_v2_AUPRC": boot_v2["AUPRC"],
                "bootstrap_v2_AUPRC_ci_lower": boot_v2["AUPRC 95% CI Lower Bound"],
                "bootstrap_v2_AUPRC_ci_upper": boot_v2["AUPRC 95% CI Upper Bound"],

                "bootstrap_v2_PPV@90RECALL": boot_v2["PPV@90RECALL"],
                "bootstrap_v2_PPV@90RECALL_ci_lower": boot_v2["PPV@90RECALL 95% CI Lower Bound"],
                "bootstrap_v2_PPV@90RECALL_ci_upper": boot_v2["PPV@90RECALL 95% CI Upper Bound"],

                "full_AUROC": boot_v2["AUROC Full Dataset"],
                "full_AUPRC": boot_v2["AUPRC Full Dataset"],
                "full_PPV@90RECALL": boot_v2["PPV@90RECALL Full Dataset"],
            }
        )

    all_results_path = output_dir / "all_folds_results.csv"
    with open(all_results_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(all_fold_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_fold_results:
            writer.writerow(row)
    print(f"\nSaved all fold results to {all_results_path}")

    # Evaluate the trained model by invoking the evaluation script directly
    subprocess.run(
        [
            "python",
            "scripts/evaluate_fullset.py",
            "--predictions_dir", str(output_dir),
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