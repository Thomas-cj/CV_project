from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch.nn as nn
import torch.optim as optim

@dataclass
class ModelConfig:
    name: str
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[Any]
    visualize_fn: Callable