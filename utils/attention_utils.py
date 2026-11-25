import torch
import torch.nn.functional as F
from . import sample_per_class, get_raw_image, plot_saliency_grid

def extract_deit_attention_subset(model, dataset, samples_per_class=2, device=None, rollout=True, head_fusion="mean", discard_ratio=0.1, seed=42):
    device = device or next(model.parameters()).device
    model.eval()
    indices = sample_per_class(dataset, samples_per_class, seed)
    fused_flags, attns, hooks = [], [], []

    for blk in model.blocks:
        if hasattr(blk.attn, "fused_attn"):
            fused_flags.append((blk.attn, blk.attn.fused_attn))
            blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, inp, out: attns.append(inp[0].detach().cpu())))

    results = {c: [] for c in dataset.classes}
    try:
        with torch.no_grad():
            for idx in indices:
                attns.clear()
                orig_img = get_raw_image(dataset, idx)
                trans_img, label = dataset[idx]
                pred_idx = model(trans_img.unsqueeze(0).to(device)).argmax(-1).item()

                if rollout:
                    n = attns[0].shape[-1]
                    joint = torch.eye(n)
                    for a in attns:
                        af = a.mean(1)[0] if head_fusion == "mean" else (a.max(1)[0][0] if head_fusion == "max" else a.min(1)[0][0])
                        if discard_ratio > 0:
                            flat = af.view(-1)
                            flat[flat.topk(int(flat.numel() * discard_ratio), largest=False)[1]] = 0
                        ar = 0.5 * af + 0.5 * torch.eye(n)
                        joint = (ar / ar.sum(-1, keepdim=True)) @ joint
                    cls_attn = joint[0, 1:]
                else:
                    cls_attn = attns[-1].mean(1)[0][0, 1:]

                g = int(cls_attn.shape[0] ** 0.5)
                amap = F.interpolate(cls_attn.reshape(1, 1, g, g), size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()
                amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                results[dataset.classes[label]].append({
                    "index": idx, "original_image": orig_img, "transformed_image": trans_img.cpu(),
                    "attention_map": amap, "true_label": dataset.classes[label], "predicted_label": dataset.classes[pred_idx],
                })
    finally:
        for h in hooks: h.remove()
        for m, f in fused_flags: m.fused_attn = f
    return results

def plot_attention_grid(results, save_path="deit_attention_maps.png", mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    plot_saliency_grid(results, save_path, title="Attention Rollout", mean=mean, std=std)