import importlib
from typing import Optional
from .base import DatasetConfig

def load_dataset_config(name: str, data_root: str = "./data", image_size: Optional[int] = None, **kw) -> DatasetConfig:
    return importlib.import_module(f"datasets.{name}").get_dataset(data_root=data_root, image_size=image_size, **kw)