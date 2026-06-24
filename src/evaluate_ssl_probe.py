from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ClassificationHead
from evaluate_kfold import evaluate
from utils.io import save_metrics_json, save_predictions_npz


class SSLProbeModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, image):
        features = self.encoder.extract_features(image)
        return self.head(features)


def get_n_bootstrap(args):
    metrics = getattr(args, "metrics", None)
    if metrics is not None:
        return metrics.get("n_bootstrap", 1000)

    validate = getattr(args, "validate", None)
    if validate is not None:
        return validate.get("n_bootstrap", 1000)

    return 1000


def aggregate_fold_metrics(fold_metrics):
    values_by_name = {}

    def collect(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                collect(f"{prefix}.{key}" if prefix else key, child)
            return

        if value is None:
            return

        try:
            values_by_name.setdefault(prefix, []).append(float(value))
        except (TypeError, ValueError):
            return

    for metrics in fold_metrics.values():
        collect("", metrics)

    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in values_by_name.items()
    }


def evaluate_ssl_model(
    folds,
    student,
    device,
    args,
    global_step: int = 0,
):
    probe_batch_size = getattr(args, "probe_batch_size", args.batch_size)
    probe_epochs = getattr(args, "probe_epochs", 1)
    n_bootstrap = get_n_bootstrap(args)
    fold_metrics = {}

    student.eval()

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

        for probe_epoch in range(1, probe_epochs + 1):
            fc.train()
            with tqdm(
                fc_train_loader,
                desc=f"SSL probe fold {fold_idx} epoch {probe_epoch}/{probe_epochs}",
                unit="batch",
            ) as pbar:
                for image, label in pbar:
                    image = image.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True).float().unsqueeze(1)

                    with torch.no_grad():
                        features = student.extract_features(image)

                    fc_optimizer.zero_grad(set_to_none=True)
                    logits = fc(features)
                    loss = fc_criterion(logits, label)
                    loss.backward()
                    fc_optimizer.step()

                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        probe_model = SSLProbeModel(student, fc).to(device)
        metrics, y_val_true, y_val_scores = evaluate(
            probe_model,
            fc_val_loader,
            fc_criterion,
            device,
            n_bootstrap=n_bootstrap,
            seed=args.seed + fold_idx,
            return_predictions=True,
        )
        fold_metrics[fold_idx] = metrics

        if getattr(args, "save_dir", None):
            fold_dir = Path(args.save_dir) / f"ssl_probe_step_{global_step}" / f"fold_{fold_idx}"
            save_metrics_json(
                metrics,
                fold_dir / "val_metrics.json",
            )
            save_predictions_npz(
                y_val_true,
                y_val_scores,
                fold_dir / "val_predictions.npz",
            )

    return aggregate_fold_metrics(fold_metrics)
