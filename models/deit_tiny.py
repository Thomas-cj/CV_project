from typing import Optional
import torch, timm
from .base import ModelConfig
from utils.attention_utils import extract_deit_attention_subset, plot_attention_grid

REQUIRED_IMAGE_SIZE = 224

def _vis(model, dataset, samples_per_class, device, seed, mean, std, save_path):
    res = extract_deit_attention_subset(model, dataset, samples_per_class, device, seed=seed)
    plot_attention_grid(res, save_path=save_path, mean=mean, std=std)

def build_model(num_classes: int, image_size: int = 224, pretrained: bool = False, lr: Optional[float] = None, num_epochs: int = 20, **kw) -> ModelConfig:
    model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or 5e-4, weight_decay=0.05)
    return ModelConfig("deit_tiny", model, opt, None, _vis)