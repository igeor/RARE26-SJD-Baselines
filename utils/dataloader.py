import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

def create_dataloaders(train_dataset, test_dataset, args):
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader
