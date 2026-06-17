import os

import torch
import timm
import math
from torch import nn
from typing import List, Optional, Sequence
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


def get_classification_head(
        in_dim: int, 
        hidden_dims: List[int], 
        num_classes: int = 1,
        hidden_activation: str = "relu",   # Fix: Use non-linearities for hidden layers
        norm: Optional[str] = None,        # Options: "batch", "layer", None
        dropout: float = 0.1
) -> nn.Module:
    
    # If no hidden layers, return your original class directly
    if not hidden_dims: 
        return ClassificationHead(in_dim, num_classes)
    
    # 1. Define hidden activation factory
    if hidden_activation == "relu": act_fn = nn.ReLU
    elif hidden_activation == "gelu": act_fn = nn.GELU
    else: act_fn = nn.Identity

    # 2. Define normalization factory
    if norm == "batch": norm_fn = nn.BatchNorm1d
    elif norm == "layer": norm_fn = nn.LayerNorm
    else: norm_fn = None
        
    # 3. Build the intermediate hidden MLP structure
    layers = []
    current_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if norm_fn is not None:
            layers.append(norm_fn(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    # 4. Initialize your original class using the last hidden dimension
    final_head = ClassificationHead(current_dim, num_classes)
    layers.append(final_head)
    
    # Wrap the entire pipeline cleanly into a single module
    return nn.Sequential(*layers)


class TimmClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(args, device):
    pretrained = getattr(args, "pretrained", None)
    model_kwargs = {
        "pretrained": pretrained is None,
        "num_classes": 0,
    }
    if "vit" in args.model.lower():
        model_kwargs["img_size"] = args.image_size

    backbone = timm.create_model(
        args.model,
        **model_kwargs,
    )

    backbone = backbone.to(device)

    if pretrained:
        print(f"Using pretrained weights from: {pretrained}")
        pretrained_path = os.path.join(os.getcwd(), pretrained)

        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)

            if isinstance(state_dict, dict) and "teacher" in state_dict:
                state_dict = state_dict["teacher"]

            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            backbone_state_dict = {
                key.removeprefix("module.").removeprefix("backbone."): value
                for key, value in state_dict.items()
                if not key.removeprefix("module.").startswith("head.")
            }

            msg = backbone.load_state_dict(backbone_state_dict, strict=False)
            print(msg)
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    head = get_classification_head(
        in_dim=backbone.num_features,
        hidden_dims=args.head["hidden_dims"],
        hidden_activation=args.head["activation"],
        norm=args.head["norm"],
        dropout=args.head["dropout"],
    )

    model = TimmClassifier(backbone, head).to(device)

    freeze_backbone = getattr(args, "freeze_backbone", True)
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone

    for param in model.head.parameters():
        param.requires_grad = True

    return model
