"""
Star Dataset: 1D CNN + ANN Classification
========================================

Task:
    Predict the six Star Type classes (0-5) from four numerical features.

Why 1D CNN?
    The dataset is tabular, not a collection of images. Therefore, this
    implementation uses a small Conv1D network over the four standardized
    numerical feature positions. This is a demonstration of CNN feature
    extraction on tabular data; it is not equivalent to image classification.

Architecture:
    Input: (batch, 1 channel, 4 feature positions)
    Conv1D #1: 16 filters, kernel size 2 -> feature maps (16, 4)
    Conv1D #2: 32 filters, kernel size 2 -> feature maps (32, 4)
    Adaptive average pooling -> 32 values
    ANN Dense layer: 32 -> 32
    Output layer: 32 -> 6 class scores

The ANN receives 32 values after global average pooling.
The two convolution layers learn 16 + 32 = 48 filters in total.

Run:
    pip install -r requirements.txt
    python star_cnn_classification.py

Expected CSV:
    archive/6 class csv.csv
or:
    Stars.csv beside this script.

Outputs are saved in ./outputs
"""

from pathlib import Path
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. CONFIGURATION AND REPRODUCIBILITY
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

POSSIBLE_DATASETS = [
    BASE_DIR / "archive" / "6 class csv.csv",
    BASE_DIR / "6 class csv.csv",
    BASE_DIR / "Stars.csv",
]

FEATURES = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
]
TARGET = "Star type"

RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ---------------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------------

dataset_path = next((p for p in POSSIBLE_DATASETS if p.exists()), None)
if dataset_path is None:
    raise FileNotFoundError(
        "Could not find the dataset. Place '6 class csv.csv' inside an "
        "'archive' folder beside this script, or place 'Stars.csv' beside it."
    )

df = pd.read_csv(dataset_path)
print("\nDataset:", dataset_path)
print("Shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

data = df[required].copy()
for col in FEATURES + [TARGET]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# ---------------------------------------------------------------------
# DATA QUALITY AUDIT: EXACT DUPLICATES AND CONFLICTING LABELS
# ---------------------------------------------------------------------
# If two rows have identical input features but different target labels,
# the dataset contains an ambiguity that must be reviewed rather than
# silently learned by the classifier. Stop so this cannot go unnoticed.

same_features_label_counts = data.groupby(FEATURES, dropna=False)[TARGET].nunique()
conflicting_feature_groups = same_features_label_counts[same_features_label_counts > 1]
if len(conflicting_feature_groups) > 0:
    conflict_path = OUTPUT_DIR / "conflicting_feature_labels.csv"
    conflict_keys = conflicting_feature_groups.index
    if len(FEATURES) == 1:
        conflict_mask = data[FEATURES[0]].isin(conflict_keys)
    else:
        conflict_frame = pd.DataFrame(list(conflict_keys), columns=FEATURES)
        conflict_mask = data[FEATURES].merge(
            conflict_frame.drop_duplicates(), on=FEATURES, how="left", indicator=True
        )["_merge"].eq("both")
    data.loc[conflict_mask].to_csv(conflict_path, index=False)
    raise ValueError(
        "Identical feature vectors have conflicting Star type labels. "
        f"Review {conflict_path} before training."
    )

rows_before_deduplication = len(data)
# Identical feature+label rows can make a random holdout look artificially
# strong if copies land in both train and test. Keep one copy of each.
data = data.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
exact_duplicate_rows_removed = rows_before_deduplication - len(data)

X_raw = data[FEATURES].to_numpy(dtype=np.float32)
y_raw = data[TARGET].to_numpy(dtype=np.int64)

print("\n========== DATA QUALITY AUDIT ==========")
print("Rows before exact-duplicate removal:", rows_before_deduplication)
print("Exact duplicate rows removed:", exact_duplicate_rows_removed)
print("Unique rows used:", len(data))
print("Conflicting identical-feature labels:", 0)

classes = np.sort(np.unique(y_raw))
num_classes = len(classes)
if num_classes != 6:
    print(f"Warning: expected 6 star types; found {num_classes}: {classes}")

# Convert labels to contiguous class indices in case labels are not 0..N-1.
class_to_index = {label: i for i, label in enumerate(classes)}
index_to_class = {i: label for label, i in class_to_index.items()}
y = np.array([class_to_index[v] for v in y_raw], dtype=np.int64)

print("\nRows after cleaning:", len(data))
print("Feature count:", len(FEATURES))
print("Star type classes:", classes)
print("Class distribution:\n", pd.Series(y_raw).value_counts().sort_index())


# ---------------------------------------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------
# Split 70% train, 15% validation, 15% test.
# The test set is kept untouched until final evaluation.

X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# Verify that no exact feature vector appears in more than one split.
def row_set(array):
    return {tuple(row.tolist()) for row in array}

train_feature_rows = row_set(X_train_raw)
val_feature_rows = row_set(X_val_raw)
test_feature_rows = row_set(X_test_raw)
train_val_overlap = train_feature_rows.intersection(val_feature_rows)
train_test_overlap = train_feature_rows.intersection(test_feature_rows)
val_test_overlap = val_feature_rows.intersection(test_feature_rows)

split_audit = pd.DataFrame([
    ["Train", len(X_train_raw), len(train_feature_rows)],
    ["Validation", len(X_val_raw), len(val_feature_rows)],
    ["Test", len(X_test_raw), len(test_feature_rows)],
], columns=["Split", "Rows", "Unique feature vectors"])
split_audit.to_csv(OUTPUT_DIR / "split_audit.csv", index=False)

print("\n========== SPLIT LEAKAGE AUDIT ==========")
print(split_audit.to_string(index=False))
print("Exact feature-vector overlap train/validation:", len(train_val_overlap))
print("Exact feature-vector overlap train/test:", len(train_test_overlap))
print("Exact feature-vector overlap validation/test:", len(val_test_overlap))
if train_val_overlap or train_test_overlap or val_test_overlap:
    raise RuntimeError(
        "Exact feature-vector duplicates remain across data splits. "
        "Review the dataset deduplication/audit before trusting evaluation."
    )

# Fit scaler on training data only to avoid data leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ---------------------------------------------------------------------
# 4. CONVERT TABULAR ROWS INTO 1D CNN INPUT
# ---------------------------------------------------------------------
# Each sample has four feature positions and one input channel.
# PyTorch Conv1D expects: (batch_size, channels, sequence_length).
# Thus: (N, 4) -> (N, 1, 4).

X_train_t = torch.tensor(X_train).unsqueeze(1)
X_val_t = torch.tensor(X_val).unsqueeze(1)
X_test_t = torch.tensor(X_test).unsqueeze(1)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)


# ---------------------------------------------------------------------
# 5. DEFINE CNN FEATURE EXTRACTOR + ANN CLASSIFIER
# ---------------------------------------------------------------------

class StarCNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        # CNN block 1: 1 input channel -> 16 learned filters.
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=2,
            stride=1, padding="same"
        )
        self.relu1 = nn.ReLU()

        # CNN block 2: 16 input feature maps -> 32 learned filters.
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=2,
            stride=1, padding="same"
        )
        self.relu2 = nn.ReLU()

        # Converts (batch, 32, 4) to (batch, 32).
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ANN / fully connected classifier.
        self.fc1 = nn.Linear(32, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(32, number_of_classes)

    def forward(self, x, return_features=False):
        fmap1 = self.relu1(self.conv1(x))       # (B, 16, 4)
        fmap2 = self.relu2(self.conv2(fmap1))   # (B, 32, 4)
        pooled = self.global_pool(fmap2).squeeze(-1)  # (B, 32)
        hidden = self.dropout(self.relu3(self.fc1(pooled)))
        logits = self.fc_out(hidden)

        if return_features:
            return logits, fmap1, fmap2, pooled
        return logits


model = StarCNN(num_classes).to(DEVICE)

print("\n========== MODEL ARCHITECTURE ==========")
print(model)

# Explicit parameter and shape accounting.
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

dummy = torch.zeros(2, 1, len(FEATURES)).to(DEVICE)
with torch.no_grad():
    logits, map1, map2, ann_input = model(dummy, return_features=True)

print("\n========== FEATURE MAP / KERNEL DETAILS ==========")
print("Input shape per sample:", tuple(dummy.shape[1:]), "(channels, feature positions)")
print("Conv1 output shape per sample:", tuple(map1.shape[1:]), "(16 feature maps x 4 positions)")
print("Conv2 output shape per sample:", tuple(map2.shape[1:]), "(32 feature maps x 4 positions)")
print("After global average pooling:", tuple(ann_input.shape[1:]))
print("Number of Conv1 filters:", model.conv1.out_channels)
print("Conv1 kernel size:", model.conv1.kernel_size)
print("Number of Conv2 filters:", model.conv2.out_channels)
print("Conv2 kernel size:", model.conv2.kernel_size)
print("Total learned convolution filters:", model.conv1.out_channels + model.conv2.out_channels)
print("Values passed into ANN per sample:", ann_input.shape[1])
print("ANN hidden layer width:", model.fc1.out_features)
print("ANN output units:", model.fc_out.out_features)
print("Trainable parameters:", count_parameters(model))

parameter_rows = []
for name, parameter in model.named_parameters():
    parameter_rows.append({
        "Layer parameter": name,
        "Shape": str(tuple(parameter.shape)),
        "Parameter count": parameter.numel(),
    })
pd.DataFrame(parameter_rows).to_csv(OUTPUT_DIR / "model_parameter_details.csv", index=False)


# ---------------------------------------------------------------------
# 6. LOSS, OPTIMIZER, AND CLASS WEIGHTS
# ---------------------------------------------------------------------
# The dataset is designed with 40 examples per class, but class weights
# are calculated from the training split to remain robust if it changes.

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------
# 7. TRAINING AND VALIDATION
# ---------------------------------------------------------------------

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    all_true, all_pred = [], []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(xb)
            loss = criterion(outputs, yb)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(xb)
        predictions = outputs.argmax(dim=1)
        all_true.extend(yb.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_true, all_pred)
    return average_loss, accuracy


history = {
    "train_loss": [], "val_loss": [],
    "train_accuracy": [], "val_accuracy": [],
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

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}.")
        break

model.load_state_dict(best_state)

# Save training history.
pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)


# ---------------------------------------------------------------------
# 8. TRAINING CURVES
# ---------------------------------------------------------------------

epochs_ran = len(history["train_loss"])
epoch_axis = range(1, epochs_ran + 1)

plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, history["train_loss"], label="Training loss")
plt.plot(epoch_axis, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_training_validation_loss.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, history["train_accuracy"], label="Training accuracy")
plt.plot(epoch_axis, history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_training_validation_accuracy.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 9. FINAL TEST SET EVALUATION
# ---------------------------------------------------------------------

model.eval()
all_probabilities, all_predictions, all_targets = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)

        all_probabilities.append(probabilities)
        all_predictions.extend(predictions)
        all_targets.extend(yb.numpy())

probabilities = np.vstack(all_probabilities)
y_pred = np.array(all_predictions)
y_true = np.array(all_targets)

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

try:
    y_true_onehot = label_binarize(y_true, classes=np.arange(num_classes))
    roc_auc_macro = roc_auc_score(
        y_true_onehot, probabilities, average="macro", multi_class="ovr"
    )
    roc_auc_weighted = roc_auc_score(
        y_true_onehot, probabilities, average="weighted", multi_class="ovr"
    )
except ValueError:
    roc_auc_macro, roc_auc_weighted = np.nan, np.nan

metric_table = pd.DataFrame([
    ["Accuracy", accuracy],
    ["Precision (macro)", precision_macro],
    ["Recall (macro)", recall_macro],
    ["F1-score (macro)", f1_macro],
    ["Precision (weighted)", precision_weighted],
    ["Recall (weighted)", recall_weighted],
    ["F1-score (weighted)", f1_weighted],
    ["ROC-AUC (macro OvR)", roc_auc_macro],
    ["ROC-AUC (weighted OvR)", roc_auc_weighted],
], columns=["Metric", "Test value"])

print("\n========== TEST METRICS ==========")
print(metric_table.to_string(index=False))
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_true, y_pred,
    labels=np.arange(num_classes),
    target_names=[str(index_to_class[i]) for i in range(num_classes)],
    zero_division=0,
))
metric_table.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

report = classification_report(
    y_true, y_pred, output_dict=True, zero_division=0
)
pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv")


# ---------------------------------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------------------------------

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=[index_to_class[i] for i in range(num_classes)],
    yticklabels=[index_to_class[i] for i in range(num_classes)],
)
plt.xlabel("Predicted Star Type")
plt.ylabel("Actual Star Type")
plt.title("CNN Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 11. MULTICLASS ROC AND PRECISION-RECALL CURVES
# ---------------------------------------------------------------------

y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    if len(np.unique(y_true_bin[:, i])) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {index_to_class[i]} (AUC={class_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_roc_curves.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    if len(np.unique(y_true_bin[:, i])) < 2:
        continue
    p, r, _ = precision_recall_curve(y_true_bin[:, i], probabilities[:, i])
    ap = average_precision_score(y_true_bin[:, i], probabilities[:, i])
    plt.plot(r, p, label=f"Class {index_to_class[i]} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("One-vs-Rest Precision-Recall Curves")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_precision_recall_curves.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 12. VISUALIZE LEARNED FEATURE MAPS
# ---------------------------------------------------------------------
# Display activations for one test observation. The map positions correspond
# to the four input feature positions, not to pixels in an image.

sample_index = 0
sample = X_test_t[sample_index:sample_index+1].to(DEVICE)
model.eval()

with torch.no_grad():
    _, feature_maps_1, feature_maps_2, pooled_ann_input = model(
        sample, return_features=True
    )

fm1 = feature_maps_1[0].cpu().numpy()  # (16, 4)
fm2 = feature_maps_2[0].cpu().numpy()  # (32, 4)

fig, axes = plt.subplots(4, 4, figsize=(11, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(range(len(FEATURES)), fm1[i], marker="o")
    ax.set_title(f"Conv1 map {i+1}")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(["T", "L", "R", "M"], fontsize=7)
    ax.grid(alpha=0.2)
fig.suptitle("Conv1D Layer 1: 16 Learned Feature Maps")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_conv1_feature_maps.png", dpi=300)
plt.close()

fig, axes = plt.subplots(8, 4, figsize=(12, 15))
for i, ax in enumerate(axes.flat):
    ax.plot(range(len(FEATURES)), fm2[i], marker="o")
    ax.set_title(f"Conv2 map {i+1}")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(["T", "L", "R", "M"], fontsize=7)
    ax.grid(alpha=0.2)
fig.suptitle("Conv1D Layer 2: 32 Learned Feature Maps")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_conv2_feature_maps.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 13. VISUALIZE LEARNED CONVOLUTION KERNEL WEIGHTS
# ---------------------------------------------------------------------

w1 = model.conv1.weight.detach().cpu().numpy()  # (16, 1, 2)
w2 = model.conv2.weight.detach().cpu().numpy()  # (32, 16, 2)

fig, axes = plt.subplots(4, 4, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    sns.heatmap(w1[i, 0, :].reshape(1, -1), annot=True, fmt=".3f",
                cmap="coolwarm", center=0, cbar=False, ax=ax)
    ax.set_title(f"Conv1 filter {i+1}")
    ax.set_yticks([])
    ax.set_xlabel("Kernel positions")
plt.suptitle("Learned Conv1D Layer 1 Kernel Weights (16 filters)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_conv1_kernel_weights.png", dpi=300)
plt.close()

# For Conv2 each filter spans 16 input channels and 2 positions.
# Show each filter as a 16 x 2 weight heatmap.
fig, axes = plt.subplots(8, 4, figsize=(12, 16))
for i, ax in enumerate(axes.flat):
    sns.heatmap(w2[i], cmap="coolwarm", center=0, cbar=False, ax=ax)
    ax.set_title(f"Conv2 filter {i+1}")
    ax.set_xlabel("Kernel positions")
    ax.set_ylabel("Input maps")
plt.suptitle("Learned Conv1D Layer 2 Kernel Weights (32 filters)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_conv2_kernel_weights.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 14. SHOW EXACT VALUES PASSED INTO ANN
# ---------------------------------------------------------------------

ann_values = pooled_ann_input[0].detach().cpu().numpy()
pd.DataFrame({
    "Pooled feature index": np.arange(1, len(ann_values) + 1),
    "Value passed to ANN": ann_values,
}).to_csv(OUTPUT_DIR / "ann_input_values_sample.csv", index=False)

plt.figure(figsize=(10, 4))
plt.bar(np.arange(1, len(ann_values) + 1), ann_values)
plt.xlabel("Pooled CNN feature index")
plt.ylabel("Activation")
plt.title("32 Values Passed from CNN Feature Extractor into ANN")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_ann_input_activations.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 15. SAVE MODEL AND FINAL SUMMARY
# ---------------------------------------------------------------------

torch.save({
    "model_state_dict": model.state_dict(),
    "features": FEATURES,
    "classes": classes.tolist(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}, OUTPUT_DIR / "star_cnn_model.pth")

print("\n========== CNN ARCHITECTURE SUMMARY ==========")
print("Input per sample: 4 numerical values")
print("Input tensor: (1 channel, 4 positions)")
print("Conv1: 16 filters x kernel size 2 -> 16 feature maps, each length 4")
print("Conv2: 32 filters x kernel size 2 -> 32 feature maps, each length 4")
print("Total convolution filters:", 16 + 32)
print("Global average pooling: 32 x 4 -> 32 values")
print("ANN input size:", 32)
print("ANN hidden layer:", 32)
print("Output neurons:", num_classes)
print("Trainable parameters:", count_parameters(model))
print("\nBest validation loss:", best_val_loss)
print("Test accuracy:", accuracy)
print("Test macro F1:", f1_macro)
print("Test observations:", len(y_true))
print(
    "Interpretation: each test observation represents "
    f"{100.0 / len(y_true):.2f}% of test accuracy. "
    "Treat a perfect score on this small test set as preliminary, not proof "
    "of generalization."
)
print("\nAll metrics, model details, plots, feature maps, and weights saved to:")
print(OUTPUT_DIR)

# Important interpretation note.
print(
    "\nNOTE: This is a small 1D CNN applied to four tabular features. "
    "CNNs are primarily designed for data with meaningful local structure "
    "(such as images or signals). Compare this model with a conventional "
    "MLP baseline before claiming that CNN is better for this dataset."
)
