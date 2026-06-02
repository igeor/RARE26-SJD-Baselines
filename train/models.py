import torch
import timm
import math
from torch import nn
from typing import List, Sequence
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from utils.module import freeze_module, get_parent_module

class DINOStudentTeacherModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.hidden_size = self.backbone.num_features
        self.lora_layers: List[str] = []

        if use_lora:
            freeze_module(self.backbone)
            self.lora_layers = apply_lora_to_linear_layers(
                self.backbone,
                target_keywords=lora_targets or ["qkv"],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if len(self.lora_layers) == 0:
                raise RuntimeError("No LoRA target layers found. For timm DINOv3 ViT, try --lora-target qkv.")

        self.head = DINOHead(self.hidden_size, out_dim=out_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 16384,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
        

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


class ClassificationHead(nn.Module):
    """Small supervised head used to monitor Domain Adaptation """

    def __init__(self, in_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)