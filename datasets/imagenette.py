from typing import Optional
import torchvision
import torchvision.transforms as T
from .base import DatasetConfig

def get_dataset(data_root: str = "./data", image_size: Optional[int] = None, blur: bool = False, kernel_size: int = 5, sigma: float = 2.0, **kw) -> DatasetConfig:
    mean, std, sz = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), image_size or 224
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    blur_op = [T.GaussianBlur(kernel_size, sigma)] if blur else []

    train_ds = torchvision.datasets.Imagenette(data_root, split="train", size="full", download=True, transform=T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), *norm]))
    test_ds = torchvision.datasets.Imagenette(data_root, split="val", size="full", download=True, transform=T.Compose([T.Resize(int(sz * 256 / 224)), T.CenterCrop(sz), *blur_op, *norm]))

    classes = [c[0] if isinstance(c, (tuple, list)) else c for c in test_ds.classes]
    test_ds.classes = classes
    if not hasattr(test_ds, "targets"):
        test_ds.targets = [y for _, y in getattr(test_ds, "_samples", [])]

    name = "imagenette_blurred" if blur else "imagenette"
    return DatasetConfig(name, train_ds, test_ds, classes, len(classes), sz, mean, std, batch_size=64)