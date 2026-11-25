from typing import Optional
import torch, torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .base import ModelConfig
from utils.gradcam_utils import extract_resnet_gradcam_subset, plot_gradcam_grid

REQUIRED_IMAGE_SIZE = None

def _vis(model, dataset, samples_per_class, device, seed, mean, std, save_path):
    res = extract_resnet_gradcam_subset(model, dataset, model.layer4[-1], samples_per_class, device, seed=seed)
    plot_gradcam_grid(res, save_path=save_path, mean=mean, std=std)

def build_model(num_classes: int, image_size: int = 32, pretrained: bool = False, lr: Optional[float] = None, num_epochs: int = 20, **kw) -> ModelConfig:
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=lr or 0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    return ModelConfig("resnet18", model, opt, sched, _vis)