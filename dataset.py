from pathlib import Path

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
