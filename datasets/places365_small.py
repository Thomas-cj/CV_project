import os
from typing import Optional
import torchvision
import torchvision.transforms as T
from .base import DatasetConfig

def get_dataset(data_root: str = "./data", image_size: Optional[int] = None, blur: bool = False, kernel_size: int = 5, sigma: float = 2.0, **kw) -> DatasetConfig:
    mean, std, sz = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), image_size or 224
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    blur_op = [T.GaussianBlur(kernel_size, sigma)] if blur else []

    train_tf = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), *norm])
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(sz), *blur_op, *norm])

    dl_train = not os.path.exists(os.path.join(data_root, "data_256_standard"))
    dl_val = not os.path.exists(os.path.join(data_root, "val_256"))

    train_ds = torchvision.datasets.Places365(data_root, split="train-standard", small=True, download=dl_train, transform=train_tf)
    test_ds = torchvision.datasets.Places365(data_root, split="val", small=True, download=dl_val, transform=test_tf)

    classes = [c.rsplit("/", 1)[-1] for c in test_ds.classes]
    test_ds.classes = classes
    if not hasattr(test_ds, "targets"):
        test_ds.targets = [y for _, y in test_ds.imgs]

    name = "places365_small_blurred" if blur else "places365_small"
    return DatasetConfig(name, train_ds, test_ds, classes, len(classes), sz, mean, std, batch_size=64)