"""Folder-based dataset reader and reproducible image transforms."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "valid", "test")
EXPECTED_CLASSES = ("Banarasi", "Bandhani", "Ikat", "Pichwai")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.65, hue=0.10)
        ], p=0.9),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def synthetic_recolor_transform(image_size: int = 224) -> transforms.Compose:
    """A synthetic color perturbation used only for same-source-image robustness checks."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.55, contrast=0.5, saturation=0.9, hue=0.16)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CategoryFolderDataset(Dataset):
    """Reads split/class/image folders while preserving one shared class mapping."""
    def __init__(self, data_root: str | Path, split: str, class_to_idx: dict[str, int] | None = None,
                 transform=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        self.data_root = Path(data_root)
        self.split = split
        split_dir = self.data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        folder_ds = datasets.ImageFolder(str(split_dir), transform=None)
        if class_to_idx is not None:
            if folder_ds.class_to_idx != class_to_idx:
                raise ValueError(f"Class folders differ in {split}: found {folder_ds.class_to_idx}, expected {class_to_idx}")
        self.classes = folder_ds.classes
        self.class_to_idx = folder_ds.class_to_idx
        if set(self.classes) != set(EXPECTED_CLASSES):
            raise ValueError(f"Expected only the verified broad category folders {EXPECTED_CLASSES}; found {self.classes}")
        self.samples = folder_ds.samples
        self.transform = transform
        if not self.samples:
            raise ValueError(f"No supported images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def inspect_splits(data_root: str | Path):
    """Load folder datasets and assert identical class names in all three splits."""
    root = Path(data_root)
    train = CategoryFolderDataset(root, "train", transform=train_transform())
    val = CategoryFolderDataset(root, "valid", class_to_idx=train.class_to_idx, transform=eval_transform())
    test = CategoryFolderDataset(root, "test", class_to_idx=train.class_to_idx, transform=eval_transform())
    return train, val, test


def resolve_data_root(data_root: str | Path) -> Path:
    """Accept either a root with train/valid/test or one containing an archive/ child."""
    root = Path(data_root)
    if all((root / split).is_dir() for split in SPLITS):
        return root
    nested = root / "archive"
    if all((nested / split).is_dir() for split in SPLITS):
        return nested
    expected = ", ".join(str(root / split) for split in SPLITS)
    raise FileNotFoundError(f"Expected train/, valid/, test/ under {root} or {nested}. Checked: {expected}")


def list_image_paths(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
