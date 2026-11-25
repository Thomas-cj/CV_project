import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def sample_per_class(dataset, samples_per_class: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    targets = np.asarray(getattr(dataset, "targets", getattr(dataset, "labels", None)))
    return [i for c in range(len(dataset.classes)) for i in rng.choice(np.where(targets == c)[0], min(samples_per_class, (targets == c).sum()), replace=False)]

def get_raw_image(dataset, idx: int):
    if hasattr(dataset, "data"):
        raw = dataset.data[idx]
        return Image.fromarray(raw) if isinstance(raw, np.ndarray) else raw
    tf, dataset.transform = dataset.transform, None
    img, _ = dataset[idx]
    dataset.transform = tf
    return img

def plot_saliency_grid(results, save_path: str, title: str = "Saliency", mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    mean_t, std_t = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    items = [x for v in results.values() for x in v]
    fig, axes = plt.subplots(len(items), 3, figsize=(9, 2.3 * len(items)))
    if len(items) == 1:
        axes = axes[None, :]
    for r, item in enumerate(items):
        smap = item.get("attention_map", item.get("saliency_map"))
        vis = (item["transformed_image"] * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[r, 0].imshow(item["original_image"])
        axes[r, 0].set_title(f"Orig: {item['true_label']} (#{item['index']})")
        axes[r, 1].imshow(vis)
        axes[r, 1].set_title(f"Input (Pred: {item['predicted_label']})")
        axes[r, 2].imshow(vis)
        axes[r, 2].imshow(smap, cmap="jet", alpha=0.5)
        axes[r, 2].set_title(title)
        for c in range(3):
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {len(items)} maps to {save_path}")