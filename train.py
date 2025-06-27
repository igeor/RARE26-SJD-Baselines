import os
import csv
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import timm
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    Subset,
)

from dataset import RareDataset
from utils import split_dataset
from metrics import compute_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Use tqdm to wrap your dataloader to add the progress bar
    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update the progress bar description with loss and accuracy
            pbar.set_postfix(loss=total_loss / (pbar.n + 1), accuracy=correct / total)

    # Return average loss and accuracy for the epoch
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["Loss"] = total_loss / len(dataloader)

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Create unique output folder for the experiment ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_{args.model}"
    output_dir = Path("/app/output") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Update log and model paths
    args.log_file = str(output_dir / "training_log.csv")
    args.save_model = str(output_dir / "best_model.pth")

    # Optional: Save config for reference
    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # === Initialize dataset and dataloaders ===
    print("loading data from: ", os.path.join(args.data_path, "train"))
    dataset = RareDataset(os.path.join(args.data_path, "train"))

    train_dataset, val_dataset = split_dataset(dataset, args.val_split, seed=42)

    if args.sampling == "oversample":
        print("Using WeightedRandomSampler for oversampling.")
        class_counts = [
            train_dataset.dataset.class_counts["nondysplastic"],
            train_dataset.dataset.class_counts["neoplasia"],
        ]
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

        # Get the original indices used in the split
        train_indices = (
            train_dataset.indices
            if isinstance(train_dataset, Subset)
            else range(len(train_dataset))
        )

        # Create weights based on label
        sample_weights = []
        for idx in train_indices:
            _, label = train_dataset.dataset[idx]
            sample_weights.append(class_weights[label])

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )

    elif args.sampling == "undersample":
        print("Using subset undersampling.")
        label_to_indices = defaultdict(list)

        # Work with original dataset
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            label_to_indices[label].append(i)

        min_count = min(len(label_to_indices[0]), len(label_to_indices[1]))

        balanced_indices = random.sample(
            label_to_indices[0], min_count
        ) + random.sample(label_to_indices[1], min_count)
        balanced_subset = Subset(train_dataset, balanced_indices)

        train_loader = DataLoader(
            balanced_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    else:
        print("Using standard sampling (no rebalancing).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # === Initialize model, loss function, and optimizer ===
    model = timm.create_model(args.model, pretrained=True, num_classes=1)
    model = model.to(device)

    if args.use_gastronet:
        print("Using pretrained Gastronet weights.")
        gastronet_path = os.path.join(os.getcwd(), "pretrained", "gastronet.pth")
        if os.path.exists(gastronet_path):
            state_dict = torch.load(gastronet_path, map_location=device)
            # Adjust the state dict keys if necessary
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    with open(args.log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_auroc",
                "val_auprc",
                "val_ppv90",
                "val_acc",
                "val_sens",
                "val_spec",
            ]
        )

    # === Training loop ===
    best_loss = 100000
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, **val_metrics})

        with open(args.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_metrics["Loss"],
                    val_metrics["AUROC"],
                    val_metrics["AUPRC"],
                    val_metrics["PPV@90% Recall"],
                    val_metrics["Accuracy"],
                    val_metrics["Sensitivity"],
                    val_metrics["Specificity"],
                ]
            )

        if val_metrics["Loss"] < best_loss:
            print()
            best_loss = val_metrics["Loss"]
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved at epoch {epoch + 1}")

        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Metrics: {val_metrics}"
        )

    if args.use_wandb:
        wandb.finish()

    # Evaluate the trained model by invoking the evaluation script directly
    subprocess.run(
        [
            "python",
            "evaluate.py",
            "--data_path",
            os.path.join(args.data_path, "test"),
            "--model_path",
            str(output_dir / "best_model.pth"),
            "--model",
            args.model,
            "--experiment_path",
            str(output_dir),
            "--batch_size",
            "1",
            "--output_file",
            str(output_dir / "evaluation_metrics.csv"),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model architecture from timm"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--log_file", type=str, default="training_log.csv", help="CSV log file path"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="best_model.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="RARE-Challenge", help="WandB project name"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Ratio of training to validation data",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["none", "oversample", "undersample"],
        default="none",
        help="Sampling strategy to handle class imbalance",
    )
    parser.add_argument(
        "--use_gastronet",
        action="store_true",
        help="Use pretrained gastronet-weights for the model",
    )
    args = parser.parse_args()
    main(args)
