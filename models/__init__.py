import importlib
from typing import Optional
from .base import ModelConfig

def get_model_required_size(name: str) -> Optional[int]:
    return getattr(importlib.import_module(f"models.{name}"), "REQUIRED_IMAGE_SIZE", None)

def load_model_config(name: str, num_classes: int, image_size: int, pretrained: bool = False, lr: Optional[float] = None, num_epochs: int = 20, **kw) -> ModelConfig:
    return importlib.import_module(f"models.{name}").build_model(num_classes=num_classes, image_size=image_size, pretrained=pretrained, lr=lr, num_epochs=num_epochs, **kw)