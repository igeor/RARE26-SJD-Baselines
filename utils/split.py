import random
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Subset


def split_dataset(dataset, val_split=0.2, seed=42):
    """Split dataset into train and validation sets by class.

    Args:
        dataset (RareDataset): Dataset instance.
        val_split (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_subset (Subset), val_subset (Subset)
    """
    random.seed(seed)

    # Separate indices by string label
    neoplasia_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "neoplasia"
    ]
    ndbe_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "nondysplastic"
    ]

    # Shuffle both lists
    random.shuffle(neoplasia_indices)
    random.shuffle(ndbe_indices)

    # Split function
    def split(indices):
        val_size = int(len(indices) * val_split)
        return indices[val_size:], indices[:val_size]  # train, val

    # Split each class
    train_neo, val_neo = split(neoplasia_indices)
    train_ndbe, val_ndbe = split(ndbe_indices)

    # Combine splits
    train_indices = train_neo + train_ndbe
    val_indices = val_neo + val_ndbe

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_kfold_dataset(dataset, k=5, seed=42):
    """Split dataset into K folds stratified by class and center.

    Args:
        dataset (RareDataset): Dataset instance.
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list[tuple[Subset, Subset]]: One ``(train_subset, val_subset)`` pair per
        fold.
    """
    rng = random.Random(seed)
    grouped_indices = defaultdict(list)

    for idx, (path, label) in enumerate(dataset.samples):
        # RareDataset stores samples as root/center/class/image.ext.
        path = Path(path)
        center = path.parent.parent.name
        grouped_indices[(center, label)].append(idx)

    fold_indices = [[] for _ in range(k)]

    grouped_indices = list(grouped_indices.values())
    rng.shuffle(grouped_indices)

    for group_idx, indices in enumerate(grouped_indices):
        rng.shuffle(indices)
        start_fold = group_idx % k
        for offset, idx in enumerate(indices):
            fold_indices[(start_fold + offset) % k].append(idx)

    all_indices = set(range(len(dataset)))
    folds = []

    for val_indices in fold_indices:
        rng.shuffle(val_indices)
        train_indices = list(all_indices - set(val_indices))
        rng.shuffle(train_indices)
        folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))

    return folds
