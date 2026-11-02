"""
Fashion-MNIST Image Classification using a CNN feature extractor + ANN classifier.

Dataset: Fashion-MNIST (downloaded automatically by torchvision on first run)
Input: grayscale image, 1 x 28 x 28
CNN: Conv2D(1->32), Conv2D(32->64), Conv2D(64->128)
ANN: 128 -> 64 -> 10
"""

from pathlib import Path
import copy
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ---------------------------- Configuration ----------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Windows-friendly

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ------------------------------- Dataset --------------------------------

# Fashion-MNIST images are 28x28 grayscale. Normalize using dataset-level
# values commonly used for Fashion-MNIST.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

print("\nDownloading/loading Fashion-MNIST...")
full_train = datasets.FashionMNIST(
    root=str(DATA_DIR), train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR), train=False, download=True, transform=transform
)

train_size = 54_000
val_size = 6_000
train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)

print(f"Training images:   {len(train_dataset):,}")
print(f"Validation images: {len(val_dataset):,}")
print(f"Test images:       {len(test_dataset):,}")
print("Image shape:", tuple(full_train[0][0].shape))
print("Number of classes:", len(CLASS_NAMES))

# ------------------------------- Model ----------------------------------

class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.30)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False):
        fmap1 = self.pool1(self.relu1(self.conv1(x)))  # (B, 32, 14, 14)
        fmap2 = self.pool2(self.relu2(self.conv2(fmap1)))  # (B, 64, 7, 7)
        fmap3 = self.relu3(self.conv3(fmap2))  # (B, 128, 7, 7)
        pooled = self.global_pool(fmap3).flatten(1)  # (B, 128)
        hidden = self.dropout(self.relu4(self.fc1(pooled)))
        logits = self.fc_out(hidden)

        if return_features:
            return logits, fmap1, fmap2, fmap3, pooled
        return logits


model = FashionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# Print actual shapes with a dummy image.
dummy = torch.zeros(2, 1, 28, 28, device=DEVICE)
with torch.no_grad():
    _, m1, m2, m3, ann_input = model(dummy, return_features=True)

print("\n========== MODEL ARCHITECTURE ==========")
print(model)
print("\n========== FEATURE MAP / KERNEL DETAILS ==========")
print("Input per image:", (1, 28, 28))
print("After Conv1 + MaxPool:", tuple(m1.shape[1:]))
print("After Conv2 + MaxPool:", tuple(m2.shape[1:]))
print("After Conv3:", tuple(m3.shape[1:]))
print("After global average pooling:", tuple(ann_input.shape[1:]))
print("Conv1 filters:", model.conv1.out_channels, "| kernel:", model.conv1.kernel_size)
print("Conv2 filters:", model.conv2.out_channels, "| kernel:", model.conv2.kernel_size)
print("Conv3 filters:", model.conv3.out_channels, "| kernel:", model.conv3.kernel_size)
print("Total convolution filters:", sum(
    layer.out_channels for layer in [model.conv1, model.conv2, model.conv3]
))
print("Values passed into ANN per image:", ann_input.shape[1])
print("ANN hidden units:", model.fc1.out_features)
print("Output units:", model.fc_out.out_features)
print("Trainable parameters:", count_parameters(model))

parameter_details = []
for name, param in model.named_parameters():
    parameter_details.append({
        "Parameter": name,
        "Shape": str(tuple(param.shape)),
        "Count": param.numel()
    })
pd.DataFrame(parameter_details).to_csv(
    OUTPUT_DIR / "model_parameter_details.csv", index=False
)

# ------------------------------- Training -------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    all_true, all_pred = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(all_true, all_pred)
    )

history = {
    "train_loss": [], "val_loss": [],
    "train_accuracy": [], "val_accuracy": []
}
best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

print("\n========== TRAINING ==========")
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}.")
        break

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

# ----------------------------- Training plots ---------------------------

epochs_axis = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, history["train_loss"], label="Training loss")
plt.plot(epochs_axis, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
plt.title("Training and Validation Loss"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_loss_curves.png", dpi=250)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, history["train_accuracy"], label="Training accuracy")
plt.plot(epochs_axis, history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_accuracy_curves.png", dpi=250)
plt.close()

# -------------------------------- Testing --------------------------------

model.eval()
all_probs, all_preds, all_targets = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

probabilities = np.vstack(all_probs)
y_pred = np.asarray(all_preds)
y_true = np.asarray(all_targets)

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
}
y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
try:
    metrics["ROC_AUC_macro_OvR"] = roc_auc_score(
        y_true_bin, probabilities, average="macro", multi_class="ovr"
    )
    metrics["ROC_AUC_weighted_OvR"] = roc_auc_score(
        y_true_bin, probabilities, average="weighted", multi_class="ovr"
    )
except ValueError:
    metrics["ROC_AUC_macro_OvR"] = float("nan")
    metrics["ROC_AUC_weighted_OvR"] = float("nan")

print("\n========== TEST METRICS ==========")
for key, value in metrics.items():
    print(f"{key:25s}: {value:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
))
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted class"); plt.ylabel("Actual class")
plt.title("Fashion-MNIST Test Confusion Matrix")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=250)
plt.close()

# Multiclass ROC curves
plt.figure(figsize=(9, 7))
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves"); plt.legend(fontsize=8)
plt.grid(alpha=.2); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_roc_curves.png", dpi=250)
plt.close()

# Precision-recall curves
plt.figure(figsize=(9, 7))
for i, name in enumerate(CLASS_NAMES):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probabilities[:, i])
    ap = average_precision_score(y_true_bin[:, i], probabilities[:, i])
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("One-vs-Rest Precision-Recall Curves")
plt.legend(fontsize=8); plt.grid(alpha=.2); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_precision_recall_curves.png", dpi=250)
plt.close()

# -------------------------- Feature map analysis ------------------------

# Visualize one test image and the corresponding feature maps.
sample_image, sample_label = test_dataset[0]
sample_batch = sample_image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    _, fmap1, fmap2, fmap3, pooled = model(
        sample_batch, return_features=True
    )

def save_feature_maps(feature_maps, filename, title, max_maps=32):
    maps = feature_maps[0].detach().cpu().numpy()
    count = min(len(maps), max_maps)
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < count:
            ax.imshow(maps[i], cmap="viridis")
            ax.set_title(f"Feature map {i+1}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=250)
    plt.close()

save_feature_maps(fmap1, "06_conv1_feature_maps.png", "Conv1: 32 feature maps")
save_feature_maps(fmap2, "07_conv2_feature_maps.png", "Conv2: 64 feature maps")
save_feature_maps(fmap3, "08_conv3_feature_maps.png", "Conv3: first 32 of 128 feature maps")

# Kernel weights: first layer filters are directly viewable as 3x3 images.
weights = model.conv1.weight.detach().cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(12, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[i, 0], cmap="coolwarm")
    ax.set_title(f"Filter {i+1}")
    ax.axis("off")
fig.suptitle("Learned Conv1 3x3 Kernels (32 filters)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_conv1_kernel_weights.png", dpi=250)
plt.close()

# ANN input values
ann_values = pooled[0].detach().cpu().numpy()
pd.DataFrame({
    "Feature_index": np.arange(1, len(ann_values) + 1),
    "ANN_input_value": ann_values
}).to_csv(OUTPUT_DIR / "ann_input_values_sample.csv", index=False)

plt.figure(figsize=(11, 4))
plt.bar(np.arange(1, len(ann_values) + 1), ann_values)
plt.xlabel("Pooled CNN feature index")
plt.ylabel("Activation")
plt.title("128 Values Passed from CNN Feature Extractor into ANN")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_ann_input_activations.png", dpi=250)
plt.close()

# Save a sample image with its true and predicted labels.
predicted_class = int(y_pred[0])
plt.figure(figsize=(4, 4))
plt.imshow(sample_image.squeeze().numpy() * 0.3530 + 0.2860, cmap="gray")
plt.title(f"Sample test image\nTrue: {CLASS_NAMES[sample_label]}\nPred: {CLASS_NAMES[predicted_class]}")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_sample_prediction.png", dpi=250)
plt.close()

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "input_shape": [1, 28, 28],
    "architecture": "FashionCNN"
}, OUTPUT_DIR / "fashion_mnist_cnn.pth")

print("\n========== FINAL ARCHITECTURE SUMMARY ==========")
print("Input image: 1 x 28 x 28")
print("Conv1: 32 filters, 3x3 kernels -> 32 x 14 x 14 after pooling")
print("Conv2: 64 filters, 3x3 kernels -> 64 x 7 x 7 after pooling")
print("Conv3: 128 filters, 3x3 kernels -> 128 x 7 x 7")
print("Global average pooling: 128 x 7 x 7 -> 128 values")
print("ANN: 128 -> 64 -> 10")
print("Total convolution filters: 224")
print("ANN input values per image: 128")
print("Trainable parameters:", count_parameters(model))
print("Best validation loss:", best_val_loss)
print("Test accuracy:", metrics["Accuracy"])
print("\nOutputs saved to:", OUTPUT_DIR)
