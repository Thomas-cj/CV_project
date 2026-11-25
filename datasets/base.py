from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import Dataset

@dataclass
class DatasetConfig:
    name: str
    train_dataset: Dataset
    test_dataset: Dataset
    classes: List[str]
    num_classes: int
    image_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    batch_size: int = 128