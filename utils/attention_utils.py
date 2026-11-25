import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_deit_attention_subset(
    model,
    dataset,
    samples_per_class=2,
    device=None,
    rollout=True,
    head_fusion="mean",
    discard_ratio=0.1,
    seed=42,
):
    """
    Selects `samples_per_class` images for each class in `dataset` and returns:
      - original_image (PIL.Image)
      - transformed_image (Tensor [3, 224, 224])
      - attention_map (np.ndarray [224, 224] normalized to [0, 1])
      - true_label (str)
      - predicted_label (str)

    Works on both clean and corrupted (e.g., Gaussian blur) datasets.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rng = np.random.default_rng(seed)

    # 1. Select balanced indices per class
    targets = np.array(getattr(dataset, "targets", getattr(dataset, "labels", None)))
    if targets is None:
        raise AttributeError("Dataset must expose `.targets` or `.labels` to sample per class.")

    num_classes = len(dataset.classes)
    selected_indices = []

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        chosen = rng.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False,
        )
        selected_indices.extend(chosen.tolist())

    # 2. Temporarily disable fused_attn so `attn_drop` receives explicit [B, H, N, N] weights
    original_fused_flags = []
    attention_weights = []
    hooks = []

    def _hook_fn(module, inputs, output):
        # inputs[0] is the post-softmax attention matrix: [B, num_heads, tokens, tokens]
        attention_weights.append(inputs[0].detach().cpu())

    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            original_fused_flags.append((block.attn, block.attn.fused_attn))
            block.attn.fused_attn = False
        hooks.append(block.attn.attn_drop.register_forward_hook(_hook_fn))

    results = {cls_name: [] for cls_name in dataset.classes}

    try:
        with torch.no_grad():
            for idx in selected_indices:
                attention_weights.clear()

                # Retrieve raw original image before transforms
                if hasattr(dataset, "data"):
                    raw_item = dataset.data[idx]
                    orig_img = Image.fromarray(raw_item) if isinstance(raw_item, np.ndarray) else raw_item
                else:
                    saved_tf = dataset.transform
                    dataset.transform = None
                    orig_img, _ = dataset[idx]
                    dataset.transform = saved_tf

                # Retrieve transformed image as passed to the model
                trans_img, label = dataset[idx]
                input_tensor = trans_img.unsqueeze(0).to(device)

                # Forward pass triggers the hooks across all transformer blocks
                logits = model(input_tensor)
                pred_idx = logits.argmax(dim=-1).item()

                # 3. Compute Attention Map ([CLS] token -> 14x14 spatial patches)
                if rollout:
                    num_tokens = attention_weights[0].shape[-1]
                    joint_att = torch.eye(num_tokens)

                    for attn in attention_weights:
                        if head_fusion == "mean":
                            attn_fused = attn.mean(dim=1)[0]
                        elif head_fusion == "max":
                            attn_fused = attn.max(dim=1)[0][0]
                        elif head_fusion == "min":
                            attn_fused = attn.min(dim=1)[0][0]
                        else:
                            raise ValueError(f"Unsupported head_fusion: {head_fusion}")

                        if discard_ratio > 0:
                            flat = attn_fused.view(-1)
                            _, drop_idx = flat.topk(int(flat.size(0) * discard_ratio), largest=False)
                            flat[drop_idx] = 0

                        attn_res = 0.5 * attn_fused + 0.5 * torch.eye(num_tokens)
                        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
                        joint_att = torch.matmul(attn_res, joint_att)

                    cls_attn = joint_att[0, 1:]
                else:
                    last_attn = attention_weights[-1].mean(dim=1)[0]
                    cls_attn = last_attn[0, 1:]

                # Reshape 196 patches -> 14x14 grid and upsample to 224x224
                grid_size = int(cls_attn.shape[0] ** 0.5)
                attn_grid = cls_attn.reshape(1, 1, grid_size, grid_size)
                attn_map = F.interpolate(
                    attn_grid, size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze().numpy()

                # Normalize map to [0, 1]
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                results[dataset.classes[label]].append({
                    "index": idx,
                    "original_image": orig_img,
                    "transformed_image": trans_img.cpu(),
                    "attention_map": attn_map,
                    "true_label": dataset.classes[label],
                    "predicted_label": dataset.classes[pred_idx],
                })

    finally:
        # Always restore model hooks and fused_attn flags cleanly
        for h in hooks:
            h.remove()
        for attn_module, flag in original_fused_flags:
            attn_module.fused_attn = flag

    return results

def plot_attention_grid(
    results,
    save_path="deit_attention_maps.png",
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
):
    """
    Plots all extracted samples per class showing:
      [Original Image | Transformed Image | Attention Overlay]
    """
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    # Flatten all extracted samples across classes
    all_items = [item for cls_items in results.values() for item in cls_items]
    total_rows = len(all_items)

    fig, axes = plt.subplots(total_rows, 3, figsize=(9, 2.3 * total_rows))
    if total_rows == 1:
        axes = axes[None, :]

    for row, item in enumerate(all_items):
        saliency = item.get("attention_map", item.get("saliency_map"))
        vis_trans = (item["transformed_image"] * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(item["original_image"])
        axes[row, 0].set_title(f"Orig: {item['true_label']} (#{item['index']})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vis_trans)
        axes[row, 1].set_title(f"Trans (Pred: {item['predicted_label']})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vis_trans)
        axes[row, 2].imshow(saliency, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("Attention Rollout")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {total_rows} attention visualizations to {save_path}")