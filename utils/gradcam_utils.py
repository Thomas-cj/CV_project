import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_resnet_gradcam_subset(
    model,
    dataset,
    target_layer=None,
    samples_per_class=2,
    device=None,
    target_mode="predicted",
    seed=42,
):
    """
    Selects `samples_per_class` images for each class in `dataset` and returns:
      - original_image (PIL.Image)
      - transformed_image (Tensor [3, H, W])
      - saliency_map (np.ndarray [H, W] normalized to [0, 1])
      - true_label (str)
      - predicted_label (str)

    Args:
        target_layer: Conv layer to hook (defaults to `model.layer4[-1]`).
        target_mode: "predicted" (explains model's guess) or "true" (explains ground truth).
    """
    if device is None:
        device = next(model.parameters()).device

    if target_layer is None:
        target_layer = model.layer4[-1]

    model.eval()
    rng = np.random.default_rng(seed)

    # 1. Select balanced indices per class (matches DeiT seed=42 selection)
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

    # 2. Register forward hook + tensor gradient hook (safe with inplace=True ReLU)
    activations = []
    gradients = []

    def _forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad.detach().cpu()))

    handle = target_layer.register_forward_hook(_forward_hook)
    results = {cls_name: [] for cls_name in dataset.classes}

    try:
        for idx in selected_indices:
            activations.clear()
            gradients.clear()
            model.zero_grad(set_to_none=True)

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

            # Forward & backward pass (requires grad enabled)
            with torch.enable_grad():
                logits = model(input_tensor)
                pred_idx = logits.argmax(dim=-1).item()
                class_idx = pred_idx if target_mode == "predicted" else label

                score = logits[0, class_idx]
                score.backward()

            # 3. Compute Grad-CAM: global-average-pool gradients -> weighted sum -> ReLU
            act = activations[0].detach().cpu()   # [1, C, H_f, W_f]
            grad = gradients[0]                   # [1, C, H_f, W_f]

            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Upsample feature map to match transformed image spatial size
            img_h, img_w = trans_img.shape[-2:]
            cam_map = F.interpolate(
                cam, size=(img_h, img_w), mode="bilinear", align_corners=False
            ).squeeze().numpy()

            # Normalize to [0, 1]
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

            results[dataset.classes[label]].append({
                "index": idx,
                "original_image": orig_img,
                "transformed_image": trans_img.cpu(),
                "saliency_map": cam_map,
                "true_label": dataset.classes[label],
                "predicted_label": dataset.classes[pred_idx],
            })

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    return results

def plot_gradcam_grid(
    results,
    save_path="resnet_gradcam_maps.png",
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
):
    """
    Plots all extracted samples per class showing:
      [Original Image | Model Input Image | Grad-CAM Overlay]
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
        saliency = item.get("saliency_map", item.get("attention_map"))
        vis_trans = (item["transformed_image"] * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(item["original_image"])
        axes[row, 0].set_title(f"Orig: {item['true_label']} (#{item['index']})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vis_trans)
        axes[row, 1].set_title(f"Input (Pred: {item['predicted_label']})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vis_trans)
        axes[row, 2].imshow(saliency, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("Grad-CAM (layer4)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {total_rows} Grad-CAM visualizations to {save_path}")