import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT
    / "output"
    / "model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42"
)
DEFAULT_CHECKPOINT_PATH_LIST = [
    DEFAULT_EXPERIMENT_DIR / f"fold_{fold_idx}" / "checkpoint_epoch_15.pth"
    for fold_idx in range(1, 6)
]


class ClassificationHead(nn.Module):
    """Small supervised head used to monitor Domain Adaptation."""

    def __init__(self, in_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_classification_head(
    in_dim: int,
    hidden_dims: List[int],
    num_classes: int = 1,
    hidden_activation: str = "relu",
    norm: Optional[str] = None,
    dropout: float = 0.1,
) -> nn.Module:
    if not hidden_dims:
        return ClassificationHead(in_dim, num_classes)

    if hidden_activation == "relu":
        act_fn = nn.ReLU
    elif hidden_activation == "gelu":
        act_fn = nn.GELU
    else:
        act_fn = nn.Identity

    if norm == "batch":
        norm_fn = nn.BatchNorm1d
    elif norm == "layer":
        norm_fn = nn.LayerNorm
    else:
        norm_fn = None

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

    final_head = ClassificationHead(current_dim, num_classes)
    layers.append(final_head)
    return nn.Sequential(*layers)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


class EnsembleModel(nn.Module):
    def __init__(
        self,
        model_name,
        checkpoint_path_list,
        image_size=224,
        head_hidden_dims=None,
        head_activation="relu",
        head_norm=None,
        head_dropout=0.1,
        device=None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not checkpoint_path_list:
            raise ValueError("checkpoint_path_list must contain at least one checkpoint.")

        self.models = nn.ModuleList()

        for checkpoint_path in checkpoint_path_list:
            checkpoint_path = resolve_path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=1,
            )

            model.fc = get_classification_head(
                in_dim=model.num_features,
                hidden_dims=head_hidden_dims or [],
                hidden_activation=head_activation,
                norm=head_norm,
                dropout=head_dropout,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Expected checkpoint with 'model_state_dict': {checkpoint_path}"
                )

            state_dict = strip_module_prefix(checkpoint["model_state_dict"])
            model.load_state_dict(state_dict, strict=True)
            model.requires_grad_(False)
            model.to(self.device)
            model.eval()

            self.models.append(model)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, x):
        probs = []

        for model in self.models:
            logits = model(x)
            prob = torch.sigmoid(logits)
            probs.append(prob)

        return torch.stack(probs, dim=0).mean(dim=0)

    def predict(self, images: list[np.ndarray]):
        """
        Accepts a list of numpy images (HWC, uint8 or float), converts them to
        PIL Images, applies transforms, and runs inference.
        """
        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        probs = []
        for img in pil_images:
            img = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prob = self.forward(img).squeeze().cpu().item()

            probs.append(prob)

        return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained RARE26 ensemble model."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help=(
            "Path to a trained checkpoint. Repeat this argument to ensemble "
            "multiple checkpoints. Defaults to all five fold checkpoints."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="resnet50",
        help="timm model name used by the checkpoints.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size used for inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used to load the ensemble.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # checkpoint_path_list = args.checkpoints or DEFAULT_CHECKPOINT_PATH_LIST
    checkpoint_path_list = [
        r"output/model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42/fold_1/checkpoint_epoch_15.pth",
        r"output/model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42/fold_2/checkpoint_epoch_15.pth",
        r"output/model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42/fold_3/checkpoint_epoch_15.pth",
        r"output/model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42/fold_4/checkpoint_epoch_15.pth",
        r"output/model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42/fold_5/checkpoint_epoch_15.pth"
    ]

    model = EnsembleModel(
        model_name=args.model_name,
        checkpoint_path_list=checkpoint_path_list,
        image_size=args.image_size,
        head_hidden_dims=[256],
        head_activation="relu",
        head_norm=None,
        head_dropout=0.1,
        device=torch.device(args.device),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # fake_input_batch = [
    #     np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    #     for _ in range(2)
    # ]
    
    # Load the .tiff batch from data\test\sanity
    test_images_dir = REPO_ROOT / "data" / "test" / "sanity"
    import tifffile 
    test_input_batch = tifffile.imread(test_images_dir / "example_batch_0_15.tiff")

    with torch.no_grad():
        output = model.predict(test_input_batch)
    print(output)

    print(f"Loaded ensemble with {len(model.models)} model(s).")
    print(f"Device: {model.device}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    main()
