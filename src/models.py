import os

import torch
import timm
from torch import nn
from typing import List, Optional
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from utils.module import freeze_module
from lora import apply_lora_to_linear_layers, get_lora_args

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


class TimmSegmentorClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _strip_checkpoint_prefix(key: str) -> str:
    for prefix in ("module.", "backbone."):
        key = key.removeprefix(prefix)
    return key


def _prepare_backbone_state_dict(
    state_dict,
    backbone: nn.Module,
    has_lora_wrappers: bool,
):
    backbone_keys = backbone.state_dict()
    backbone_state_dict = {}
    skipped_keys = []

    for raw_key, value in state_dict.items():
        key = _strip_checkpoint_prefix(raw_key)
        if key.startswith("head.") or key.startswith("dino_head."):
            skipped_keys.append(key)
            continue

        candidate_keys = [key]
        if has_lora_wrappers and ".base." not in key:
            for suffix in (".weight", ".bias"):
                if key.endswith(suffix):
                    candidate_keys.append(key[: -len(suffix)] + ".base" + suffix)
                    break

        loaded = False
        for candidate_key in candidate_keys:
            expected = backbone_keys.get(candidate_key)
            if expected is not None and expected.shape == value.shape:
                backbone_state_dict[candidate_key] = value
                loaded = True
                break

        if not loaded:
            skipped_keys.append(key)

    return backbone_state_dict, skipped_keys


def _print_load_summary(load_msg, loaded_count: int, skipped_keys) -> None:
    missing_keys = list(load_msg.missing_keys)
    unexpected_keys = list(load_msg.unexpected_keys)
    expected_lora_missing = [
        key for key in missing_keys if ".lora_A." in key or ".lora_B." in key
    ]
    other_missing = [key for key in missing_keys if key not in expected_lora_missing]

    print(f"Loaded {loaded_count} pretrained backbone tensor(s).")
    if expected_lora_missing:
        print(f"Initialized {len(expected_lora_missing)} new LoRA adapter tensor(s).")
    if other_missing:
        print(f"Missing non-LoRA backbone key(s): {other_missing[:10]}")
    if unexpected_keys:
        print(f"Unexpected checkpoint key(s): {unexpected_keys[:10]}")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} checkpoint key(s) not used by this backbone: {skipped_keys[:10]}")


def build_model(args, device):
    pretrained = getattr(args, "pretrained", None)
    lora_args = get_lora_args(args)
    freeze_backbone = getattr(args, "freeze_backbone", True)
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

    state_dict = None
    checkpoint_has_lora = False
    if pretrained:
        print(f"Using pretrained weights from: {pretrained}")
        pretrained_path = os.path.join(os.getcwd(), pretrained)

        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)

            if isinstance(state_dict, dict) and "teacher" in state_dict:
                state_dict = state_dict["teacher"]

            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            checkpoint_has_lora = any(
                ".lora_A." in key or ".lora_B." in key or ".base." in key
                for key in state_dict.keys()
            )
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    lora_layers = []
    should_attach_lora = lora_args["enabled"] or checkpoint_has_lora
    if should_attach_lora:
        freeze_module(backbone)
        lora_layers = apply_lora_to_linear_layers(
            backbone,
            target_keywords=lora_args["targets"],
            r=lora_args["r"],
            alpha=lora_args["alpha"],
            dropout=lora_args["dropout"],
        )
        if len(lora_layers) == 0:
            raise RuntimeError(
                "No LoRA target layers found. For timm ViT/DINO models, try targets: ['qkv']."
            )
        if lora_args["enabled"]:
            print(f"Enabled LoRA on {len(lora_layers)} layer(s): {', '.join(lora_layers[:5])}")
        else:
            print(f"Attached frozen LoRA layers for checkpoint compatibility: {len(lora_layers)} layer(s)")
    elif freeze_backbone:
        freeze_module(backbone)
    else:
        for param in backbone.parameters():
            param.requires_grad = True

    if state_dict is not None:
        backbone_state_dict, skipped_keys = _prepare_backbone_state_dict(
            state_dict,
            backbone,
            has_lora_wrappers=should_attach_lora,
        )
        msg = backbone.load_state_dict(backbone_state_dict, strict=False)
        _print_load_summary(msg, len(backbone_state_dict), skipped_keys)

    head = get_classification_head(
        in_dim=backbone.num_features,
        hidden_dims=args.head["hidden_dims"],
        hidden_activation=args.head["activation"],
        norm=args.head["norm"],
        dropout=args.head["dropout"],
    )

    model = TimmClassifier(backbone, head).to(device)

    if should_attach_lora and not lora_args["enabled"]:
        freeze_module(model.backbone)
    for param in model.head.parameters():
        param.requires_grad = True

    model = model.to(device)

    return model
