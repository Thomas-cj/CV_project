import os
from typing import Optional
import torchvision
import torchvision.transforms as T
from .base import DatasetConfig

def get_dataset(data_root: str = "./data/my_dataset", image_size: Optional[int] = None, **kw) -> DatasetConfig:
    mean, std, sz = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), image_size or 224
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    train_ds = torchvision.datasets.ImageFolder(os.path.join(data_root, "train"), T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), *norm]))
    test_ds = torchvision.datasets.ImageFolder(os.path.join(data_root, "test"), T.Compose([T.Resize(int(sz * 256 / 224)), T.CenterCrop(sz), *norm]))
    name = os.path.basename(os.path.normpath(data_root))
    return DatasetConfig(name, train_ds, test_ds, test_ds.classes, len(test_ds.classes), sz, mean, std, batch_size=64)