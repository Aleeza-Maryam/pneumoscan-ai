"""Custom Dataset for chest X-ray images."""
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- Constants ----------
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


# ---------- Preprocessing ----------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves contrast in X-ray images, making lung structures clearer.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel (EfficientNet expects 3 channels)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced


# ---------- Transforms ----------
def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Augmentation pipeline for training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation/test transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ---------- Dataset ----------
class ChestXRayDataset(Dataset):
    """
    Dataset for Chest X-Ray Pneumonia classification.

    Directory structure expected:
        root/
        ├── NORMAL/
        │   ├── img1.jpeg
        │   └── ...
        └── PNEUMONIA/
            ├── img1.jpeg
            └── ...
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[A.Compose] = None,
        use_clahe: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_clahe = use_clahe

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Collect all image paths and labels
        self.samples: list[Tuple[Path, int]] = []
        for class_name in CLASSES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = CLASS_TO_IDX[class_name]
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image (OpenCV reads as BGR)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE for contrast enhancement
        if self.use_clahe:
            image = apply_clahe(image)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback: convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_class_distribution(self) -> dict:
        """Return count of samples per class."""
        dist = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            dist[CLASSES[label]] += 1
        return dist