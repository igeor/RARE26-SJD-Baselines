import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import timm
import torch
import yaml
from PIL import Image, ImageSequence
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from models import build_model, get_classification_head  # noqa: E402


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def strip_module_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def default_model_args():
    return {
        "model": "resnet50",
        "pretrained": "",
        "freeze_backbone": True,
        "image_size": 224,
        "head": {"hidden_dims": [], "activation": "relu", "norm": None, "dropout": 0.1},
    }


def load_args(config_path, checkpoint, cli_args):
    args = default_model_args()

    if config_path:
        with open(resolve_path(config_path), "r", encoding="utf-8") as f:
            args.update(yaml.safe_load(f) or {})

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        args.update(checkpoint["args"])

    if cli_args.model:
        args["model"] = cli_args.model
    if cli_args.image_size:
        args["image_size"] = cli_args.image_size

    args["pretrained"] = ""
    args.setdefault("head", default_model_args()["head"])

    if args.get("lora_target") and not args.get("lora_targets"):
        args["lora_targets"] = args["lora_target"]

    return SimpleNamespace(**args)


def has_lora(state_dict):
    return any(".lora_A." in key or ".lora_B." in key or ".base." in key for key in state_dict)


def has_wrapped_model_keys(state_dict):
    return any(key.startswith(("backbone.", "head.")) for key in state_dict)


def has_direct_custom_fc(state_dict):
    return any(
        key.startswith("fc.") and key not in {"fc.weight", "fc.bias"}
        for key in state_dict
    )


def build_direct_timm_model(model_args, state_dict, device):
    model = timm.create_model(model_args.model, pretrained=False, num_classes=1).to(device)

    if has_direct_custom_fc(state_dict):
        classifier = model.get_classifier()
        in_features = getattr(classifier, "in_features", None)
        if in_features is None:
            in_features = getattr(model, "num_features", None)
        if in_features is None:
            raise ValueError(
                f"Could not infer classifier input size for direct checkpoint model: {model_args.model}"
            )

        model.fc = get_classification_head(
            in_dim=in_features,
            hidden_dims=model_args.head["hidden_dims"],
            hidden_activation=model_args.head["activation"],
            norm=model_args.head["norm"],
            dropout=model_args.head["dropout"],
        )
        model = model.to(device)

    return model


def load_model(checkpoint_path, cli_args, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(checkpoint_state(checkpoint))
    model_args = load_args(cli_args.config, checkpoint, cli_args)

    if has_lora(state_dict):
        model_args.use_lora = True
        if hasattr(model_args, "lora") and isinstance(model_args.lora, dict):
            model_args.lora["enabled"] = True

    if has_wrapped_model_keys(state_dict) or has_lora(state_dict):
        model = build_model(model_args, device)
    else:
        model = build_direct_timm_model(model_args, state_dict, device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_args


def load_tiff_images(path):
    try:
        import tifffile
    except ModuleNotFoundError:
        with Image.open(path) as tiff:
            return [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(tiff)]

    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[None, :, :, :]

    images = []
    for img in arr:
        if img.dtype != np.uint8:
            max_value = np.max(img) or 1
            img = (img.astype(np.float32) / max_value * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        images.append(Image.fromarray(img).convert("RGB"))
    return images


def predict_stack(models, transform, images, batch_size, device):
    scores = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([transform(img) for img in images[start:start + batch_size]]).to(device)
        probs = []
        with torch.no_grad():
            for model in models:
                probs.append(torch.sigmoid(model(batch)).flatten())
        scores.extend(torch.stack(probs).mean(0).cpu().numpy().tolist())
    return scores


def load_labels_csv(path):
    with open(resolve_path(path), "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "label"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Labels CSV must contain columns: {', '.join(sorted(required))}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run external inference on TIFF stacks.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeat for an ensemble.")
    parser.add_argument("--input-dir", default=REPO_ROOT / "data" / "val", type=Path)
    parser.add_argument("--output-dir", default=REPO_ROOT / "output" / "external_predictions", type=Path)
    parser.add_argument("--labels-csv", type=Path, help="Optional CSV with filename,label rows matching TIFF frame order.")
    parser.add_argument("--config", help="Optional YAML config used to build the model.")
    parser.add_argument("--model", help="Override timm model name, e.g. resnet50 or vit_large_patch16_dinov3.lvd1689m.")
    parser.add_argument("--image-size", type=int, help="Override inference image size.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_paths = sorted([*input_dir.rglob("*.tif"), *input_dir.rglob("*.tiff")])
    if not tiff_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found under: {input_dir}")

    loaded = [load_model(path, args, device) for path in args.checkpoint]
    models = [model for model, _ in loaded]
    image_size = args.image_size or getattr(loaded[0][1], "image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rows = []
    for tiff_path in tqdm(tiff_paths, desc="TIFF stacks"):
        images = load_tiff_images(tiff_path)
        scores = predict_stack(models, transform, images, args.batch_size, device)
        rows.extend((str(tiff_path.relative_to(input_dir)), idx, score) for idx, score in enumerate(scores))

    label_rows = load_labels_csv(args.labels_csv) if args.labels_csv else None
    if label_rows is not None and len(label_rows) != len(rows):
        raise ValueError(
            f"Labels CSV has {len(label_rows)} rows, but inference produced {len(rows)} predictions"
        )

    csv_path = output_dir / "external_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if label_rows is None:
            writer.writerow(["stack", "frame", "score"])
            writer.writerows(rows)
        else:
            writer.writerow(["filename", "label", "pred"])
            writer.writerows(
                (label_row["filename"], label_row["label"], row[2])
                for label_row, row in zip(label_rows, rows)
            )

    npz_payload = {
        "stacks": np.array([row[0] for row in rows]),
        "frames": np.array([row[1] for row in rows]),
        "scores": np.array([row[2] for row in rows], dtype=np.float32),
    }
    if label_rows is not None:
        npz_payload["filenames"] = np.array([row["filename"] for row in label_rows])
        npz_payload["labels"] = np.array([int(row["label"]) for row in label_rows], dtype=np.int64)
    np.savez_compressed(output_dir / "external_predictions.npz", **npz_payload)
    print(f"Saved {len(rows)} predictions to: {csv_path}")


if __name__ == "__main__":
    main()
