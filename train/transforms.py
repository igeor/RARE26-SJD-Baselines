from PIL import Image
from sympy import im
from typing import Tuple
from torchvision import transforms


class GastroDINOv3MultiCropTransform:
    """
    Conservative medical-image multi-crop transform.

    By default, local crops are small source regions resized back to image_size.
    This keeps tensor shapes compatible with timm ViTs that assert a fixed input
    size, while preserving the DINO multi-crop idea.
    """

    def __init__(
        self,
        image_size: int = 256,
        local_crops_number: int = 6,
        global_crop_scale: Tuple[float, float] = (0.50, 1.0),
        local_crop_scale: Tuple[float, float] = (0.12, 0.50),
        strong_color_jitter: bool = False,
    ):
        self.local_crops_number = local_crops_number

        if strong_color_jitter:
            jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            grayscale_p = 0.2
        else:
            jitter = transforms.ColorJitter(0.15, 0.15, 0.15, 0.03)
            grayscale_p = 0.05

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        common_tail = [
            transforms.RandomHorizontalFlip(p=0.5),
            jitter,
            transforms.RandomGrayscale(p=grayscale_p),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crop_scale),
            *common_tail,
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=local_crop_scale),
            *common_tail,
        ])

    def __call__(self, img: Image.Image):
        crops = [self.global_transform(img), self.global_transform(img)]
        crops.extend(self.local_transform(img) for _ in range(self.local_crops_number))
        return crops


class TrainRare26Transform:
    """Simple training-time transform for RARE26 classification probe."""

    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img: Image.Image):
        return self.transform(img)
    


class ValidationRare26Transform:
    """Simple validation-time transform for RARE26 classification probe."""

    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img: Image.Image):
        return self.transform(img)