import torch
import torch.nn.functional as F
from . import sample_per_class, get_raw_image, plot_saliency_grid

def extract_resnet_gradcam_subset(model, dataset, target_layer=None, samples_per_class=2, device=None, target_mode="predicted", seed=42):
    device = device or next(model.parameters()).device
    target_layer = target_layer or model.layer4[-1]
    model.eval()
    indices = sample_per_class(dataset, samples_per_class, seed)
    acts, grads = [], []

    def _fhook(m, inp, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g.detach().cpu()))

    handle = target_layer.register_forward_hook(_fhook)
    results = {c: [] for c in dataset.classes}
    try:
        for idx in indices:
            acts.clear(); grads.clear()
            model.zero_grad(set_to_none=True)
            orig_img = get_raw_image(dataset, idx)
            trans_img, label = dataset[idx]
            with torch.enable_grad():
                logits = model(trans_img.unsqueeze(0).to(device))
                pred_idx = logits.argmax(-1).item()
                logits[0, pred_idx if target_mode == "predicted" else label].backward()

            act, grad = acts[0].detach().cpu(), grads[0]
            cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
            cmap = F.interpolate(cam, size=trans_img.shape[-2:], mode="bilinear", align_corners=False).squeeze().numpy()
            cmap = (cmap - cmap.min()) / (cmap.max() - cmap.min() + 1e-8)

            results[dataset.classes[label]].append({
                "index": idx, "original_image": orig_img, "transformed_image": trans_img.cpu(),
                "saliency_map": cmap, "true_label": dataset.classes[label], "predicted_label": dataset.classes[pred_idx],
            })
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return results

def plot_gradcam_grid(results, save_path="resnet_gradcam_maps.png", mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    plot_saliency_grid(results, save_path, title="Grad-CAM", mean=mean, std=std)