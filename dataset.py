import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile
import warnings

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class RareDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the main dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else self.default_transforms()
        self.classes = {"neo": "neoplasia", "ndbe": "nondysplastic"}
        self.class_counts = {"neoplasia": 0, "nondysplastic": 0}
        self.samples = self.load_samples()

        # Print counts after loading
        print(
            f"Loaded dataset with {self.class_counts['neoplasia']} 'neo' (neoplasia) images and "
            f"{self.class_counts['nondysplastic']} 'ndbe' (nondysplastic) images."
        )

    def default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_samples(self):
        samples = []
        for center in self.root_dir.iterdir():
            if center.is_dir():
                for class_folder in ["neo", "ndbe"]:
                    class_dir = center / class_folder
                    if class_dir.exists():
                        for img_path in class_dir.glob("*.png"):
                            label = self.classes[class_folder]
                            samples.append((img_path, label))
                            self.class_counts[label] += 1  # Count the sample
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if label == "neoplasia" else 0
        return image, label


class RareTestSet(Dataset):
    def __init__(self, root_dir, transform=None, return_paths=False):
        """
        Args:
            root_dir (str): Path to the main dataset folder (contains 'neo/' and 'ndbe/').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else self.default_transforms()
        self.classes = {"neo": "neoplasia", "ndbe": "nondysplastic"}
        self.class_counts = {"neoplasia": 0, "nondysplastic": 0}
        self.samples = self.load_samples()
        self.return_paths = return_paths

        # Print counts after loading
        print(
            f"Loaded test set with {self.class_counts['neoplasia']} 'neo' (neoplasia) images and "
            f"{self.class_counts['nondysplastic']} 'ndbe' (nondysplastic) images."
        )

    def default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_samples(self):
        samples = []
        for class_folder in ["neo", "ndbe"]:
            class_dir = self.root_dir / class_folder
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    label = self.classes[class_folder]
                    samples.append((img_path, label))
                    self.class_counts[label] += 1

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if label == "neoplasia" else 0
        if self.return_paths:
            return image, label, str(img_path)
        else:
            return image, label


class GastroNetZipDataset(Dataset):
    """Recursively reads loose images and images inside zip files."""

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, root: str, transform, max_images: Optional[int] = None, skip_bad_images: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.items: List[Tuple[str, str, Optional[str]]] = []
        self.skip_bad_images = skip_bad_images
        self._warned_bad_items = set()

        if not self.root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.root}")

        self._index_files()

        if max_images is not None and max_images > 0:
            self.items = self.items[:max_images]

        if not self.items:
            raise RuntimeError(f"No images found under: {self.root}")

    def _index_files(self) -> None:
        for dp, _, files in os.walk(self.root):
            for f in files:
                p = os.path.join(dp, f)
                lower = f.lower()

                if lower.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(p) as z:
                            for n in z.namelist():
                                if n.lower().endswith(self.IMG_EXTENSIONS):
                                    self.items.append(("zip", p, n))
                    except zipfile.BadZipFile:
                        print(f"Skipping bad zip file: {p}")
                elif lower.endswith(self.IMG_EXTENSIONS):
                    self.items.append(("file", p, None))

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, item: Tuple[str, str, Optional[str]]) -> Image.Image:
        typ, path, name = item
        if typ == "zip":
            assert name is not None
            with zipfile.ZipFile(path) as z:
                return Image.open(BytesIO(z.read(name))).convert("RGB")
        return Image.open(path).convert("RGB")

    @staticmethod
    def _describe_item(item: Tuple[str, str, Optional[str]]) -> str:
        typ, path, name = item
        if typ == "zip":
            return f"{path}!{name}"
        return path

    def __getitem__(self, idx: int):
        for offset in range(len(self.items)):
            item_idx = (idx + offset) % len(self.items)
            item = self.items[item_idx]
            try:
                img = self._load(item)
                return self.transform(img)
            except (OSError, EOFError, KeyError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
                if not self.skip_bad_images:
                    raise
                item_key = self._describe_item(item)
                if item_key not in self._warned_bad_items:
                    warnings.warn(f"Skipping unreadable image: {item_key} ({exc})")
                    self._warned_bad_items.add(item_key)

        raise RuntimeError(f"All {len(self.items)} indexed images failed to load under: {self.root}")

