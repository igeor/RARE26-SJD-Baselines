import argparse
import csv
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageSequence
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(SCRIPTS_ROOT))
sys.path.insert(2, str(REPO_ROOT))

from evaluate_external import load_model, load_tiff_images  # noqa: E402


DEFAULT_GASTRONET5M_DIR = Path(r"D:\Data\RARE26\pretrain\data")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
TIFF_EXTENSIONS = (".tif", ".tiff")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def default_input_dir():
    if DEFAULT_GASTRONET5M_DIR.exists():
        return DEFAULT_GASTRONET5M_DIR

    config_path = REPO_ROOT / "configs" / "dinov3_ssl_gastronet5m.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data_path = (yaml.safe_load(f) or {}).get("data")
        if data_path:
            data_path = resolve_path(data_path)
            if data_path.exists():
                return data_path

    return REPO_ROOT / "data" / "val"


def iter_tiff_images(input_dir):
    tiff_paths = sorted([*input_dir.rglob("*.tif"), *input_dir.rglob("*.tiff")])
    for tiff_path in tiff_paths:
        for frame_idx, image in enumerate(load_tiff_images(tiff_path)):
            yield str(tiff_path.relative_to(input_dir)), frame_idx, image


def iter_loose_images(input_dir):
    for image_path in sorted(input_dir.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            with Image.open(image_path) as image:
                yield str(image_path.relative_to(input_dir)), 0, image.convert("RGB").copy()


def iter_zip_images(input_dir):
    for zip_path in sorted(input_dir.rglob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(
                name for name in zf.namelist()
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS + TIFF_EXTENSIONS
            )
            for name in names:
                data = zf.read(name)
                if Path(name).suffix.lower() in TIFF_EXTENSIONS:
                    with Image.open(BytesIO(data)) as tiff:
                        for frame_idx, frame in enumerate(ImageSequence.Iterator(tiff)):
                            yield f"{zip_path.relative_to(input_dir)}!{name}", frame_idx, frame.convert("RGB").copy()
                else:
                    with Image.open(BytesIO(data)) as image:
                        yield f"{zip_path.relative_to(input_dir)}!{name}", 0, image.convert("RGB").copy()


def iter_images(input_dir):
    found = False
    for iterator in (iter_tiff_images, iter_loose_images, iter_zip_images):
        for item in iterator(input_dir):
            found = True
            yield item
    if not found:
        raise FileNotFoundError(f"No TIFF/image/zip inputs found under: {input_dir}")


def write_batch(writer, models, transform, batch, device):
    paths, frames, images = zip(*batch)
    tensor = torch.stack([transform(image) for image in images]).to(device)
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(torch.sigmoid(model(tensor)).flatten())
    scores = torch.stack(probs).mean(0).cpu().numpy()

    for stack, frame, score in zip(paths, frames, scores):
        writer.writerow([stack, frame, float(score)])


def main():
    parser = argparse.ArgumentParser(description="Run model inference on GastroNet5M unlabeled images.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeat for an ensemble.")
    parser.add_argument("--input-dir", type=Path, help=r"Defaults to D:\Data\RARE26\pretrain\data when it exists.")
    parser.add_argument("--output-csv", default=REPO_ROOT / "output" / "gastronet5m_predictions" / "predictions.csv", type=Path)
    parser.add_argument("--config", help="Optional YAML config used to build the model.")
    parser.add_argument("--model", help="Override timm model name, e.g. resnet50 or vit_large_patch16_dinov3.lvd1689m.")
    parser.add_argument("--image-size", type=int, help="Override inference image size.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    input_dir = resolve_path(args.input_dir) if args.input_dir else default_input_dir()
    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    loaded = [load_model(path, args, device) for path in args.checkpoint]
    models = [model for model, _ in loaded]
    image_size = args.image_size or getattr(loaded[0][1], "image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch = []
    count = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stack", "frame", "score"])

        for item in tqdm(iter_images(input_dir), desc="GastroNet5M inference"):
            batch.append(item)
            if len(batch) == args.batch_size:
                write_batch(writer, models, transform, batch, device)
                f.flush()
                count += len(batch)
                batch.clear()

        if batch:
            write_batch(writer, models, transform, batch, device)
            f.flush()
            count += len(batch)

    print(f"Saved {count} predictions to: {output_csv}")


if __name__ == "__main__":
    main()
