import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import torch

import metrics


def compute_metrics_base(y_true, y_scores):
    """Computes AUROC, AUPRC, PPV@90% Recall, Accuracy, Sensitivity, and Specificity."""

    # AUROC & AUPRC
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Find PPV @ 90% Recall
    idx = np.where(recalls >= 0.9)[0][-1]
    ppv_at_0_9_recall = precisions[idx]

    # Convert scores to binary predictions (threshold at 0.5)
    y_pred = (y_scores >= 0.5).astype(int)

    # Compute Accuracy, Sensitivity, and Specificity
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "PPV@90% Recall": ppv_at_0_9_recall,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def compute_metrics_bootstrap(
    y_true,
    y_scores,
    n_bootstrap=1000,
    min_neoplasia=1000,
    ndbe_multiplier=100,
    seed=None,
):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true).reshape(-1)
    y_scores = np.asarray(y_scores).reshape(-1)

    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")

    neoplasia_idx = np.where(y_true == 1)[0]
    ndbe_idx = np.where(y_true == 0)[0]

    if len(neoplasia_idx) == 0:
        raise ValueError("No neoplasia samples found.")

    if len(ndbe_idx) == 0:
        raise ValueError("No nondysplastic samples found.")

    results = []
    n_ndbe = min_neoplasia * ndbe_multiplier

    for _ in range(n_bootstrap):
        sampled_neoplasia_idx = rng.choice(
            neoplasia_idx,
            size=min_neoplasia,
            replace=True,
        )

        sampled_ndbe_idx = rng.choice(
            ndbe_idx,
            size=n_ndbe,
            replace=True,
        )

        sampled_idx = np.concatenate(
            [sampled_neoplasia_idx, sampled_ndbe_idx]
        )

        results.append(
            compute_metrics_base(
                y_true[sampled_idx],
                y_scores[sampled_idx],
            )
        )

    summary = {}

    for metric_name in results[0].keys():
        values = np.array(
            [result[metric_name] for result in results],
            dtype=float,
        )

        summary[metric_name] = {
            "median": float(np.nanpercentile(values, 50)),
            "ci_lower": float(np.nanpercentile(values, 2.5)),
            "ci_upper": float(np.nanpercentile(values, 97.5)),
        }

    return summary



def compute_metrics_bootstrap_v2(
    y_true, 
    y_pred, 
    n_iterations=1000, 
    imbalance_ratio=100,
    seed=None
):
    
    rng = np.random.default_rng(seed)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Separate NDBE and neoplasia image indices
    ndbe_indices = np.where(y_true == 0)[0]
    neoplasia_indices = np.where(y_true == 1)[0]

    # --------------------
    # Metrics on full dataset
    # --------------------
    auc_full = roc_auc_score(y_true, y_pred)
    auprc_full = average_precision_score(y_true, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    ppv_90_full = np.interp(0.9, recalls[::-1], precisions[::-1])

    # --------------------
    # Bootstrapping
    # --------------------
    bootstrapped_metrics = []

    n_ndbe = len(ndbe_indices)
    n_neoplasia_to_sample = max(1, int(n_ndbe / imbalance_ratio))

    for _ in range(n_iterations):
        sampled_ndbe_indices = ndbe_indices
        sampled_neoplasia_indices = rng.choice(
            neoplasia_indices, size=n_neoplasia_to_sample, replace=True
        )

        sampled_indices = np.concatenate(
            [sampled_ndbe_indices, sampled_neoplasia_indices]
        )

        y_true_sample = y_true[sampled_indices]
        y_pred_sample = y_pred[sampled_indices]

        # Calculate metrics
        auc = roc_auc_score(y_true_sample, y_pred_sample)
        auprc = average_precision_score(y_true_sample, y_pred_sample)
        precisions, recalls, _ = precision_recall_curve(y_true_sample, y_pred_sample)
        ppv_90 = np.interp(0.9, recalls[::-1], precisions[::-1])

        bootstrapped_metrics.append((auc, auprc, ppv_90))

    bootstrapped_metrics = np.array(bootstrapped_metrics)

    bootstrapped_summary = {
        "Score": np.median(bootstrapped_metrics[:, 2]),
        "PPV@90RECALL": np.median(bootstrapped_metrics[:, 2]),
        "PPV@90RECALL 95% CI Lower Bound": np.percentile(
            bootstrapped_metrics[:, 2], 2.5
        ),
        "PPV@90RECALL 95% CI Upper Bound": np.percentile(
            bootstrapped_metrics[:, 2], 97.5
        ),
        "AUROC": np.median(bootstrapped_metrics[:, 0]),
        "AUROC 95% CI Lower Bound": np.percentile(bootstrapped_metrics[:, 0], 2.5),
        "AUROC 95% CI Upper Bound": np.percentile(bootstrapped_metrics[:, 0], 97.5),
        "AUPRC": np.median(bootstrapped_metrics[:, 1]),
        "AUPRC 95% CI Lower Bound": np.percentile(bootstrapped_metrics[:, 1], 2.5),
        "AUPRC 95% CI Upper Bound": np.percentile(bootstrapped_metrics[:, 1], 97.5),
        "AUROC Full Dataset": auc_full,
        "AUPRC Full Dataset": auprc_full,
        "PPV@90RECALL Full Dataset": ppv_90_full,
    }

    return bootstrapped_summary


def evaluate(
    model=None,
    dataloader=None,
    criterion=None,
    device=None,
    y_true=None,
    y_scores=None,
    n_bootstrap=1000,
    seed=42,
    return_predictions=False,
):
    if model is not None:
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_labels, all_scores = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                logits = model(images)
                loss = criterion(logits, labels)

                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy().reshape(-1))
                all_scores.extend(probs.cpu().numpy().reshape(-1))

        y_true = np.asarray(all_labels)
        y_scores = np.asarray(all_scores)

    loss = total_loss / len(dataloader) if model is not None else None
    accuracy = correct / total if model is not None else None
    
    metrics_dict = {
        "Loss": loss,
        "Accuracy": accuracy,
        "base_metrics": compute_metrics_base(y_true, y_scores),
        "bootstrap_metrics": compute_metrics_bootstrap(
            y_true=y_true,
            y_scores=y_scores,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "bootstrap_metrics_v2": compute_metrics_bootstrap_v2(
            y_true=y_true,
            y_pred=y_scores,
            n_iterations=n_bootstrap,
            seed=seed,
        ),
    }

    if return_predictions:
        return metrics_dict, y_true, y_scores

    return metrics_dict


if __name__ == "__main__":
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    print("Base Metrics:")
    print(compute_metrics_base(y_true, y_scores))

    print("\nBootstrap Metrics:")
    print(compute_metrics_bootstrap(y_true, y_scores, n_bootstrap=1000, seed=42))

    print("\nBootstrap v2 Metrics:")
    print(compute_metrics_bootstrap_v2(y_true, y_scores, n_iterations=1000, seed=42))