from typing import Optional
import torchvision
import torchvision.transforms as T
from .base import DatasetConfig

def get_dataset(data_root: str = "./data", image_size: Optional[int] = None, kernel_size: int = 5, sigma: float = 2.0, **kw) -> DatasetConfig:
    mean, std, sz = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), image_size or 32
    resize = [T.Resize(sz)] if sz != 32 else []
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    train_ds = torchvision.datasets.CIFAR10(data_root, True, T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *resize, *norm]), download=True)
    test_ds = torchvision.datasets.CIFAR10(data_root, False, T.Compose([T.GaussianBlur(kernel_size, sigma), *resize, *norm]), download=True)
    return DatasetConfig("cifar10_blurred", train_ds, test_ds, test_ds.classes, len(test_ds.classes), sz, mean, std)