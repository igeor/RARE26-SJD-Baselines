from typing import List
import torch


def move_crops_to_device(crops, device: torch.device) -> List[torch.Tensor]:
    # Default DataLoader collation turns a per-sample list of K tensors into a
    # batch list of K tensors, each shaped [B, C, H, W].
    return [crop.to(device, non_blocking=True) for crop in crops]