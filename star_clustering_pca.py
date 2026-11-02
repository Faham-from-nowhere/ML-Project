"""
Star Dataset: K-Means Clustering from Scratch + PCA from Scratch
================================================================

Expected dataset file: Stars.csv
Download it from:
https://www.kaggle.com/datasets/deepu1109/star-dataset

Place Stars.csv in the same folder as this script.

Install dependencies:
    pip install -r requirements.txt

Run:
    python star_clustering_pca.py

K-Means and PCA are implemented manually using NumPy.
Scikit-learn is used only for scaling and evaluation metrics.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "archive" / "6 class csv.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_VALUES = [2, 3, 4, 5, 6]
MAX_ITERATIONS = 300
TOLERANCE = 1e-5
N_INITIALIZATIONS = 10

NUMERIC_FEATURES = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
]

# ---------------------------------------------------------------------
# 2. LOAD AND EXPLORE THE DATASET
# ---------------------------------------------------------------------

def load_dataset():
    """Load the Kaggle Star dataset and validate expected columns."""
    if not DATASET_PATH.exists():
        csv_files = list(BASE_DIR.glob("*.csv"))
        if len(csv_files) == 1:
            path = csv_files[0]
            print(f"Stars.csv was not found. Using: {path.name}")
        else:
            raise FileNotFoundError(
                "Dataset not found. Download the Kaggle CSV, rename it "
                "to 'Stars.csv', and place it beside this Python file."
            )
    else:
        path = DATASET_PATH

    data = pd.read_csv(path)
    print("\n========== DATASET OVERVIEW ==========")
    print("Dataset:", path.name)
    print("Shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())
    print("\nColumn names:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nStatistical summary:")
    print(data.describe(include="all").T)

    return data


data = load_dataset()

# ---------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------------------

# Check for the four physical numeric attributes.
missing_columns = [c for c in NUMERIC_FEATURES if c not in data.columns]
if missing_columns:
    raise ValueError(
        f"Expected columns not found: {missing_columns}\n"
        f"Available columns: {data.columns.tolist()}\n"
        "Check the column names in NUMERIC_FEATURES near the top of this file."
    )

# Convert selected columns to numeric and handle missing/infinite values.
numeric_data = data[NUMERIC_FEATURES].copy()
for column in NUMERIC_FEATURES:
    numeric_data[column] = pd.to_numeric(numeric_data[column], errors="coerce")

numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
valid_mask = numeric_data.notna().all(axis=1)
removed_rows = int((~valid_mask).sum())

if removed_rows:
    print(f"\nRemoving {removed_rows} rows with missing/non-numeric feature values.")

numeric_data = numeric_data.loc[valid_mask].reset_index(drop=True)
clean_data = data.loc[valid_mask].reset_index(drop=True)

if len(numeric_data) < max(K_VALUES) + 1:
    raise ValueError("Not enough valid rows to run the requested K values.")

print("\nRows used for clustering:", len(numeric_data))

# Feature distribution plots.
numeric_data.hist(figsize=(12, 9), bins=20, edgecolor="black")
plt.suptitle("Distributions of Numerical Star Features", fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_feature_distributions.png", dpi=300)
plt.close()

# Correlation heatmap.
plt.figure(figsize=(9, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_correlation_matrix.png", dpi=300)
plt.close()

# Standardization is essential because the four attributes use very
# different numerical scales.
scaler = StandardScaler()
X = scaler.fit_transform(numeric_data.to_numpy(dtype=float))

print("\nStandardized feature matrix shape:", X.shape)

# ---------------------------------------------------------------------
# 4. K-MEANS CLUSTERING IMPLEMENTED FROM SCRATCH
# ---------------------------------------------------------------------

def initialize_centroids(X, k, rng):
    """Choose k distinct observations as the initial centroids."""
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def calculate_distances(X, centroids):
    """Return Euclidean distances from every point to every centroid."""
    # Shape: (number of observations, number of clusters)
    return np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))


def assign_clusters(X, centroids):
    """Assign each observation to its nearest centroid."""
    distances = calculate_distances(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, old_centroids, k, rng):
    """Recalculate each centroid as the mean of its assigned points."""
    new_centroids = np.zeros_like(old_centroids)

    for cluster_id in range(k):
        members = X[labels == cluster_id]

        if len(members) > 0:
            new_centroids[cluster_id] = members.mean(axis=0)
        else:
            # If a cluster becomes empty, reinitialize it to a random point.
            new_centroids[cluster_id] = X[rng.integers(0, len(X))]

    return new_centroids


def calculate_wcss(X, labels, centroids):
    """Within-Cluster Sum of Squares: sum of squared point-centroid distances."""
    total = 0.0
    for cluster_id in range(len(centroids)):
        members = X[labels == cluster_id]
        if len(members) > 0:
            total += ((members - centroids[cluster_id]) ** 2).sum()
    return float(total)


def kmeans_single_run(X, k, seed=RANDOM_STATE):
    """Run one complete K-Means initialization and retain iteration history."""
    rng = np.random.default_rng(seed)
    centroids = initialize_centroids(X, k, rng)

    loss_history = []
    centroid_movement_history = []

    for iteration in range(MAX_ITERATIONS):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids, k, rng)

        movement = float(np.sqrt(((new_centroids - centroids) ** 2).sum()))
        centroids = new_centroids

        # Reassign after centroid update so WCSS matches the final centroids.
        labels = assign_clusters(X, centroids)
        loss = calculate_wcss(X, labels, centroids)

        loss_history.append(loss)
        centroid_movement_history.append(movement)

        if movement < TOLERANCE:
            break

    return {
        "labels": labels,
        "centroids": centroids,
        "wcss": calculate_wcss(X, labels, centroids),
        "loss_history": loss_history,
        "movement_history": centroid_movement_history,
        "iterations": len(loss_history),
    }


def kmeans_from_scratch(X, k, n_init=N_INITIALIZATIONS):
    """
    Run K-Means several times with different random initial centroids.
    Keep the run with the lowest WCSS.
    """
    best_result = None

    for run in range(n_init):
        result = kmeans_single_run(X, k, seed=RANDOM_STATE + run)
        if best_result is None or result["wcss"] < best_result["wcss"]:
            best_result = result

    return best_result


# ---------------------------------------------------------------------
# 5. EXPERIMENT WITH K = 2, 3, 4, 5, 6
# ---------------------------------------------------------------------

results = {}
summary_rows = []

for k in K_VALUES:
    print(f"\nRunning K-Means from scratch for K={k} ...")
    result = kmeans_from_scratch(X, k)
    results[k] = result

    labels = result["labels"]
    number_of_clusters_found = len(np.unique(labels))

    # These metrics require at least 2 non-empty clusters and fewer
    # clusters than observations.
    if 2 <= number_of_clusters_found < len(X):
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
    else:
        silhouette, db_index, ch_score = np.nan, np.nan, np.nan

    summary_rows.append({
        "K": k,
        "WCSS": result["wcss"],
        "Iterations": result["iterations"],
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Score": ch_score,
    })

    print(f"WCSS: {result['wcss']:.4f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Cluster sizes: {np.bincount(labels, minlength=k)}")

summary = pd.DataFrame(summary_rows)
print("\n========== K-MEANS EVALUATION ==========")
print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / "kmeans_evaluation.csv", index=False)

# ---------------------------------------------------------------------
# 6. ELBOW CURVE (WCSS AGAINST K)
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(summary["K"], summary["WCSS"], marker="o", linewidth=2)
plt.xticks(K_VALUES)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method for Selecting K")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_elbow_curve.png", dpi=300)
plt.close()

# Loss/convergence curves for all tested K values.
plt.figure(figsize=(9, 6))
for k, result in results.items():
    plt.plot(
        range(1, len(result["loss_history"]) + 1),
        result["loss_history"],
        marker=".",
        label=f"K={k}",
    )
plt.xlabel("Iteration")
plt.ylabel("WCSS")
plt.title("K-Means WCSS Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_kmeans_loss_convergence.png", dpi=300)
plt.close()

# Centroid movement curves.
plt.figure(figsize=(9, 6))
for k, result in results.items():
    plt.plot(
        range(1, len(result["movement_history"]) + 1),
        result["movement_history"],
        marker=".",
        label=f"K={k}",
    )
plt.xlabel("Iteration")
plt.ylabel("Total Centroid Movement")
plt.title("Centroid Movement During K-Means")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_centroid_movement.png", dpi=300)
plt.close()

# Cluster size plot for each K.
fig, axes = plt.subplots(1, len(K_VALUES), figsize=(17, 4))
for ax, k in zip(axes, K_VALUES):
    sizes = np.bincount(results[k]["labels"], minlength=k)
    ax.bar(range(k), sizes)
    ax.set_title(f"K={k}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Stars")
plt.suptitle("Cluster Sizes for Different K Values")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_cluster_sizes.png", dpi=300)
plt.close()

# Evaluation metric plots.
metric_plots = [
    ("Silhouette Score", "07_silhouette_score.png", "Higher is generally better"),
    ("Davies-Bouldin Index", "08_davies_bouldin.png", "Lower is generally better"),
    ("Calinski-Harabasz Score", "09_calinski_harabasz.png", "Higher is generally better"),
]

for metric, filename, subtitle in metric_plots:
    plt.figure(figsize=(8, 5))
    plt.plot(summary["K"], summary[metric], marker="o", linewidth=2)
    plt.xticks(K_VALUES)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel(metric)
    plt.title(f"{metric} ({subtitle})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# 7. PCA FROM SCRATCH USING COVARIANCE + EIGEN DECOMPOSITION
# ---------------------------------------------------------------------

# Step 1: Center the standardized observations.
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Step 2: Calculate covariance matrix.
covariance_matrix = np.cov(X_centered, rowvar=False)

# Step 3: Eigenvalue decomposition.
# eigh is designed for real symmetric matrices such as covariance matrices.
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Step 4: Sort eigenvalues/eigenvectors in descending order.
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Explained variance ratio.
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print("\n========== PCA RESULTS ==========")
print("\nCovariance matrix:")
print(covariance_matrix)
print("\nEigenvalues (descending):")
print(eigenvalues)
print("\nEigenvectors (columns correspond to principal components):")
print(eigenvectors)
print("\nExplained variance ratio:")
print(explained_variance_ratio)
print("\nCumulative explained variance:")
print(cumulative_explained_variance)

pca_table = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(eigenvalues))],
    "Eigenvalue": eigenvalues,
    "Explained Variance Ratio": explained_variance_ratio,
    "Cumulative Explained Variance": cumulative_explained_variance,
})
pca_table.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)
print("\nPCA explained variance table:")
print(pca_table.to_string(index=False))

# Plot individual explained variance.
plt.figure(figsize=(8, 5))
plt.bar(
    [f"PC{i+1}" for i in range(len(eigenvalues))],
    explained_variance_ratio,
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_explained_variance.png", dpi=300)
plt.close()

# Plot cumulative explained variance.
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_explained_variance) + 1),
    cumulative_explained_variance,
    marker="o",
    linewidth=2,
)
plt.xticks(range(1, len(eigenvalues) + 1))
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.ylim(0, 1.05)
plt.title("Cumulative Explained Variance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_cumulative_explained_variance.png", dpi=300)
plt.close()

# Step 6: Project original standardized data into 2D and 3D.
components_2d = eigenvectors[:, :2]
components_3d = eigenvectors[:, :3]

X_pca_2d = X_centered @ components_2d
X_pca_3d = X_centered @ components_3d

# Save transformed representations.
pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"]).to_csv(
    OUTPUT_DIR / "pca_2_components.csv", index=False
)
pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"]).to_csv(
    OUTPUT_DIR / "pca_3_components.csv", index=False
)

# ---------------------------------------------------------------------
# 8. VISUALIZE DATA USING THE FIRST TWO PRINCIPAL COMPONENTS
# ---------------------------------------------------------------------

plt.figure(figsize=(9, 7))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.75)
plt.xlabel(f"PC1 ({explained_variance_ratio[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1]*100:.2f}% variance)")
plt.title("Star Dataset Projected onto First Two Principal Components")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_pca_2d.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 9. RUN K-MEANS ON PCA REPRESENTATION AND VISUALIZE CLUSTERS
# ---------------------------------------------------------------------

# Select K after examining the elbow curve and evaluation metrics.
# This is a starting value only; do not claim it is optimal without analysis.
SELECTED_K = 4

# Use the best K-Means result obtained from the original
# standardized four-dimensional feature space.
pca_cluster_result = results[SELECTED_K]
pca_labels = pca_cluster_result["labels"]

# pca_cluster_result = kmeans_from_scratch(X_pca_2d, SELECTED_K)
# pca_labels = pca_cluster_result["labels"]

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=pca_labels,
    cmap="viridis",
    alpha=0.8,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-Means Clusters Visualized in 2D PCA Space (K={SELECTED_K})")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_pca_kmeans_clusters.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 10. COMPARE CLUSTERS WITH STAR TYPE (LABEL USED ONLY FOR INTERPRETATION)
# ---------------------------------------------------------------------

# Star Type is deliberately excluded from the clustering input features.
# It can be compared after clustering to understand cluster composition.
if "Star type" in clean_data.columns:
    comparison = pd.crosstab(
        pd.Series(pca_labels, name="Predicted Cluster"),
        clean_data["Star type"],
    )
    print("\n========== CLUSTER VS STAR TYPE ==========")
    print(comparison)
    comparison.to_csv(OUTPUT_DIR / "cluster_vs_star_type.csv")

    plt.figure(figsize=(9, 6))
    sns.heatmap(comparison, annot=True, fmt="d", cmap="Blues")
    plt.title("K-Means Cluster vs Actual Star Type")
    plt.xlabel("Actual Star Type")
    plt.ylabel("Predicted Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_cluster_star_type_heatmap.png", dpi=300)
    plt.close()
else:
    print("\n' Star type ' column not found; skipping cluster-label comparison.")

# ---------------------------------------------------------------------
# 11. OPTIONAL 3D PCA VISUALIZATION
# ---------------------------------------------------------------------

if X_pca_3d.shape[1] == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(
        X_pca_3d[:, 0],
        X_pca_3d[:, 1],
        X_pca_3d[:, 2],
        c=pca_labels,
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA Visualization with K-Means Clusters (K={SELECTED_K})")
    fig.colorbar(points, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_pca_3d_clusters.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# 12. FINAL SUMMARY
# ---------------------------------------------------------------------

print("\n========== FINAL SUMMARY ==========")
print("K values tested:", K_VALUES)
print("\nWCSS values:")
for k in K_VALUES:
    print(f"K={k}: {results[k]['wcss']:.4f}")

print("\nPCA cumulative explained variance:")
for i, value in enumerate(cumulative_explained_variance, start=1):
    print(f"First {i} component(s): {value*100:.2f}%")

print(f"\nPCA visualizations use {SELECTED_K} clusters.")
print("Review the elbow curve and evaluation metrics before finalizing SELECTED_K.")
print(f"\nAll output files have been saved in: {OUTPUT_DIR}")
