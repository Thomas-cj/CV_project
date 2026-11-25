import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.gradcam_utils import extract_resnet_gradcam_subset, plot_gradcam_grid

# Data Preparation (native 32x32 for CIFAR-10 ResNet)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Deterministic transform for testing & reproducible Grad-CAM extraction
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Data downloading")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Download finished")

# Model Initialization with CIFAR-10 3x3 stem (keeps layer4 at 4x4 for Grad-CAM)
model = resnet18(weights=None, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print("Model Initialized")

# Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
num_epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training Loop
print("Training")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)

    for batch_idx, (images, labels) in enumerate(train_bar, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")

    scheduler.step()

# Testing Loop
print("Testing")
model.eval()
correct = 0
total = 0

classes = test_dataset.classes
num_classes = len(classes)
class_correct = [0] * num_classes
class_total = [0] * num_classes

confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        matches = predicted.eq(labels)
        correct += matches.sum().item()

        for t, p, is_correct in zip(labels.view(-1), predicted.view(-1), matches):
            class_total[t.item()] += 1
            class_correct[t.item()] += is_correct.item()
            confusion_matrix[t.long(), p.long()] += 1

overall_acc = 100 * correct / total
print(f"Test Accuracy: {overall_acc:.2f}%")

# Save overall, per-class accuracy, and confusion matrix to file
output_file = "resnet18_accuracy_results.txt"
with open(output_file, "w") as f:
    f.write("Model: ResNet-18\n")
    f.write(f"Overall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})\n")
    f.write("-" * 40 + "\n")
    f.write("Per-Class Accuracy:\n")
    for i, class_name in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        line = f"  {class_name:<12}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})"
        print(line)
        f.write(line + "\n")

    f.write("\n" + "=" * 75 + "\n")
    f.write("Confusion Matrix (Rows = Actual, Columns = Predicted):\n\n")

    header = f"{'Actual / Pred':<14}" + "".join([f"{c[:6]:>7}" for c in classes])
    print("\n" + header)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    for i, class_name in enumerate(classes):
        row_vals = "".join([f"{confusion_matrix[i, j].item():>7}" for j in range(num_classes)])
        row_str = f"{class_name:<14}{row_vals}"
        print(row_str)
        f.write(row_str + "\n")

print(f"\nResults saved to {output_file}")

# Plot and save Confusion Matrix heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion_matrix.numpy(), cmap="Blues")
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=range(num_classes),
    yticks=range(num_classes),
    xticklabels=classes,
    yticklabels=classes,
    ylabel="Actual Label",
    xlabel="Predicted Label",
    title="ResNet-18 CIFAR-10 Confusion Matrix"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = confusion_matrix.max().item() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        val = confusion_matrix[i, j].item()
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > thresh else "black")

plt.tight_layout()
plot_file = "resnet18_confusion_matrix.png"
plt.savefig(plot_file, dpi=300)
plt.close(fig)
print(f"Confusion matrix plot saved to {plot_file}")

# Extract and visualize Grad-CAM maps (seed=42 matches DeiT samples)
print("\nExtracting Grad-CAM maps...")
gradcam_results = extract_resnet_gradcam_subset(
    model=model,
    dataset=test_dataset,
    target_layer=model.layer4[-1],
    samples_per_class=2,
    device=device,
    seed=42,
)
plot_gradcam_grid(gradcam_results, save_path="resnet18_gradcam_maps.png")