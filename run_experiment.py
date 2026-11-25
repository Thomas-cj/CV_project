import argparse, os
import torch, torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset_config
from models import get_model_required_size, load_model_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_tiny")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--samples-per-class", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_cfg = load_dataset_config(args.dataset, args.data_root, get_model_required_size(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = load_model_config(args.model, data_cfg.num_classes, data_cfg.image_size, args.pretrained, args.lr, args.epochs)

    model, opt, sched = model_cfg.model.to(device), model_cfg.optimizer, model_cfg.scheduler
    bs = args.batch_size or data_cfg.batch_size
    run_dir = os.path.join(args.output_dir, f"{model_cfg.name}_{data_cfg.name}")
    os.makedirs(run_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(data_cfg.train_dataset, bs, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_cfg.test_dataset, bs, shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            bar.set_postfix(loss=f"{loss_sum / i:.4f}")
        if sched:
            sched.step()

    model.eval()
    n_cls, classes = data_cfg.num_classes, data_cfg.classes
    cm = torch.zeros(n_cls, n_cls, dtype=torch.int64)
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            preds = model(x.to(device)).argmax(1).cpu()
            for t, pr in zip(y.view(-1), preds.view(-1)):
                cm[t, pr] += 1

    correct, total = cm.diag().sum().item(), cm.sum().item()
    acc = 100 * correct / total
    prefix = f"{model_cfg.name}_{data_cfg.name}"

    lines = [f"Model: {model_cfg.name} | Dataset: {data_cfg.name}", f"Overall Test Accuracy: {acc:.2f}% ({correct}/{total})", "-" * 40, "Per-Class Accuracy:"]
    for i, c in enumerate(classes):
        c_ok, c_tot = cm[i, i].item(), cm[i].sum().item()
        lines.append(f"  {c:<12}: {100 * c_ok / c_tot:.2f}% ({c_ok}/{c_tot})")

    header = f"{'Actual / Pred':<14}" + "".join(f"{c[:6]:>7}" for c in classes)
    lines += ["\n" + "=" * 75, "Confusion Matrix (Rows = Actual, Columns = Predicted):\n", header, "-" * len(header)]
    for i, c in enumerate(classes):
        lines.append(f"{c:<14}" + "".join(f"{cm[i, j].item():>7}" for j in range(n_cls)))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(run_dir, f"{prefix}_accuracy_results.txt"), "w") as f:
        f.write(report + "\n")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(n_cls), yticks=range(n_cls), xticklabels=classes, yticklabels=classes, ylabel="Actual Label", xlabel="Predicted Label", title=f"{model_cfg.name} on {data_cfg.name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max().item() / 2.0
    for i in range(n_cls):
        for j in range(n_cls):
            v = cm[i, j].item()
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close(fig)

    model_cfg.visualize_fn(model, data_cfg.test_dataset, args.samples_per_class, device, args.seed, data_cfg.mean, data_cfg.std, os.path.join(run_dir, f"{prefix}_visual_maps.png"))

if __name__ == "__main__":
    main()