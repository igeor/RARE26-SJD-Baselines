import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset


def _label_at(dataset, idx):
    if isinstance(dataset, Subset):
        return _label_at(dataset.dataset, dataset.indices[idx])

    _, label = dataset.samples[idx]
    return 1 if label == "neoplasia" else 0


def create_dataloaders(train_dataset, test_dataset, args):
    if args.sampling == "oversample":
        print("Using WeightedRandomSampler for oversampling.")
        labels = [_label_at(train_dataset, idx) for idx in range(len(train_dataset))]
        class_counts = torch.bincount(torch.tensor(labels), minlength=2).float()
        class_weights = 1.0 / class_counts

        # Create weights based on label
        sample_weights = [class_weights[label] for label in labels]

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

        # Work with dataset metadata without opening or transforming images.
        for i in range(len(train_dataset)):
            label = _label_at(train_dataset, i)
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader
