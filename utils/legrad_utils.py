import torch
import torch.nn.functional as F
from . import sample_per_class, get_raw_image, plot_saliency_grid


def extract_deit_attention_subset(
    model,
    dataset,
    samples_per_class=2,
    device=None,
    target_mode="predicted",
    layers=None,
    seed=42,
):
    """
    Extracts LeGrad saliency maps across transformer blocks for a balanced per-class subset.
    """
    device = device or next(model.parameters()).device
    model.eval()
    indices = sample_per_class(dataset, samples_per_class, seed)
    num_prefix = getattr(model, "num_prefix_tokens", 1)
    target_blocks = model.blocks if layers is None else [model.blocks[i] for i in layers]

    fused_flags, attns, hooks = [], [], []
    for blk in target_blocks:
        if hasattr(blk.attn, "fused_attn"):
            fused_flags.append((blk.attn, blk.attn.fused_attn))
            blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, inp, out: attns.append(inp[0])))

    results = {c: [] for c in dataset.classes}
    try:
        for idx in indices:
            attns.clear()
            orig_img = get_raw_image(dataset, idx)
            trans_img, label = dataset[idx]

            with torch.enable_grad():
                logits = model(trans_img.unsqueeze(0).to(device))
                pred_idx = logits.argmax(-1).item()
                score = logits[0, pred_idx if target_mode == "predicted" else label]
                grads = torch.autograd.grad(score, attns, retain_graph=False)

            # LeGrad: ReLU on attention gradients -> average over layers, heads, and query tokens
            grad_stack = F.relu(torch.stack(grads))          # [L, 1, H, N_q, N_k]
            patch_scores = grad_stack.mean(dim=(0, 2, 3))[0, num_prefix:]  # [num_patches]

            g = int(patch_scores.shape[0] ** 0.5)
            amap = F.interpolate(
                patch_scores.reshape(1, 1, g, g).detach().cpu(),
                size=trans_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

            results[dataset.classes[label]].append({
                "index": idx,
                "original_image": orig_img,
                "transformed_image": trans_img.cpu(),
                "attention_map": amap,
                "true_label": dataset.classes[label],
                "predicted_label": dataset.classes[pred_idx],
            })
    finally:
        for h in hooks:
            h.remove()
        for m, f in fused_flags:
            m.fused_attn = f
        model.zero_grad(set_to_none=True)

    return results


extract_deit_legrad_subset = extract_deit_attention_subset


def plot_attention_grid(
    results,
    save_path="deit_attention_maps.png",
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
):
    plot_saliency_grid(results, save_path, title="LeGrad", mean=mean, std=std)