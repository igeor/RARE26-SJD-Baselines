import math
from typing import List, Sequence

from torch import nn

from utils.module import get_parent_module


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: base(x) + scale * B(A(dropout(x)))."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora_to_linear_layers(
    model: nn.Module,
    target_keywords: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> List[str]:
    targets = tuple(k.strip().lower() for k in target_keywords if k.strip())
    if not targets:
        raise ValueError("At least one --lora-target keyword is required")

    to_replace: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name.lower() for k in targets):
            to_replace.append(name)

    for name in to_replace:
        parent, child_name = get_parent_module(model, name)
        old_linear = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old_linear, r=r, alpha=alpha, dropout=dropout))

    return to_replace


def get_lora_args(args):
    lora = getattr(args, "lora", None)
    if lora is None:
        return {
            "enabled": getattr(args, "use_lora", False),
            "r": getattr(args, "lora_r", 8),
            "alpha": getattr(args, "lora_alpha", 16),
            "dropout": getattr(args, "lora_dropout", 0.05),
            "targets": getattr(args, "lora_targets", ["qkv"]),
        }

    return {
        "enabled": lora.get("enabled", False),
        "r": lora.get("r", 8),
        "alpha": lora.get("alpha", 16),
        "dropout": lora.get("dropout", 0.05),
        "targets": lora.get("targets", ["qkv"]),
    }
