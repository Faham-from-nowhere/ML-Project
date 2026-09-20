import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("Stars.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDataset statistics:")
print(df.describe())


# =========================
# 2. SELECT FEATURES
# =========================

features = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)"
]

X = df[features].copy()

print("\nFeatures used for clustering:")
print(features)

print("\nStar Type is NOT used for clustering.")


# =========================
# 3. LOG TRANSFORMATION
# =========================

X["Luminosity(L/Lo)"] = np.log1p(
    X["Luminosity(L/Lo)"]
)

X["Radius(R/Ro)"] = np.log1p(
    X["Radius(R/Ro)"]
)


# =========================
# 4. STANDARDIZATION
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# 5. ELBOW METHOD
# =========================

inertias = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()


# =========================
# 6. SILHOUETTE ANALYSIS
# =========================

silhouette_scores = []

print("\n======================================")
print("SILHOUETTE ANALYSIS")
print("======================================")

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")

plt.tight_layout()
plt.savefig("silhouette_scores.png")
plt.close()


# =========================
# 7. BEST K FROM SILHOUETTE
# =========================

best_k = range(2, 11)[
    np.argmax(silhouette_scores)
]

print("\nBest K based on silhouette score:", best_k)


# =========================
# 8. K = 2 CLUSTERING
# =========================

kmeans_2 = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=20
)

clusters_2 = kmeans_2.fit_predict(X_scaled)

df["Cluster_K2"] = clusters_2

silhouette_2 = silhouette_score(
    X_scaled,
    clusters_2
)


# =========================
# 9. K = 6 CLUSTERING
# =========================

kmeans_6 = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=20
)

clusters_6 = kmeans_6.fit_predict(X_scaled)

df["Cluster_K6"] = clusters_6

silhouette_6 = silhouette_score(
    X_scaled,
    clusters_6
)


# =========================
# 10. PCA
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

print("\nPCA explained variance:")
print(
    f"PC1: {pca.explained_variance_ratio_[0]:.4f}"
)

print(
    f"PC2: {pca.explained_variance_ratio_[1]:.4f}"
)


# =========================
# 11. K = 2 PCA VISUALIZATION
# =========================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster_K2",
    palette="tab10",
    s=70
)

plt.title("K-Means Clustering (K=2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("kmeans_k2_pca.png")
plt.close()


# =========================
# 12. K = 6 PCA VISUALIZATION
# =========================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster_K6",
    palette="tab10",
    s=70
)

plt.title("K-Means Clustering (K=6)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("kmeans_k6_pca.png")
plt.close()


# =========================
# 13. K = 2 RESULTS
# =========================

print("\n======================================")
print("K = 2 RESULTS")
print("======================================")

print(f"Silhouette Score: {silhouette_2:.4f}")

print("\nCluster sizes:")

print(
    df["Cluster_K2"]
    .value_counts()
    .sort_index()
)


# =========================
# 14. K = 2 CLUSTER CENTERS
# =========================

centers_2 = scaler.inverse_transform(
    kmeans_2.cluster_centers_
)

centers_2_df = pd.DataFrame(
    centers_2,
    columns=features
)

print("\nK = 2 Cluster Centers:")

print(centers_2_df)


# =========================
# 15. K = 2 VS STAR TYPE
# =========================

print("\nK = 2 vs Star Type:")

print(
    pd.crosstab(
        df["Cluster_K2"],
        df["Star type"]
    )
)


# =========================
# 16. K = 2 ARI AND NMI
# =========================

ari_2 = adjusted_rand_score(
    df["Star type"],
    clusters_2
)

nmi_2 = normalized_mutual_info_score(
    df["Star type"],
    clusters_2
)

print("\nK = 2 Evaluation:")

print(f"ARI: {ari_2:.4f}")
print(f"NMI: {nmi_2:.4f}")


# =========================
# 17. K = 6 RESULTS
# =========================

print("\n======================================")
print("K = 6 RESULTS")
print("======================================")

print(f"Silhouette Score: {silhouette_6:.4f}")

print("\nCluster sizes:")

print(
    df["Cluster_K6"]
    .value_counts()
    .sort_index()
)


# =========================
# 18. K = 6 CLUSTER CENTERS
# =========================

centers_6 = scaler.inverse_transform(
    kmeans_6.cluster_centers_
)

centers_6_df = pd.DataFrame(
    centers_6,
    columns=features
)

print("\nK = 6 Cluster Centers:")

print(centers_6_df)


# =========================
# 19. K = 6 VS STAR TYPE
# =========================

print("\nK = 6 vs Star Type:")

print(
    pd.crosstab(
        df["Cluster_K6"],
        df["Star type"]
    )
)


# =========================
# 20. K = 6 ARI AND NMI
# =========================

ari_6 = adjusted_rand_score(
    df["Star type"],
    clusters_6
)

nmi_6 = normalized_mutual_info_score(
    df["Star type"],
    clusters_6
)

print("\nK = 6 Evaluation:")

print(f"ARI: {ari_6:.4f}")
print(f"NMI: {nmi_6:.4f}")


# =========================
# 21. FINAL COMPARISON
# =========================

comparison = pd.DataFrame({
    "K": [2, 6],
    "Silhouette Score": [
        silhouette_2,
        silhouette_6
    ],
    "ARI": [
        ari_2,
        ari_6
    ],
    "NMI": [
        nmi_2,
        nmi_6
    ]
})

print("\n======================================")
print("FINAL COMPARISON")
print("======================================")

print(
    comparison.to_string(index=False)
)


# =========================
# 22. FINAL SUMMARY
# =========================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    f"Best K by Silhouette: {best_k}"
)

print(
    f"K=2 Silhouette: {silhouette_2:.4f}"
)

print(
    f"K=6 Silhouette: {silhouette_6:.4f}"
)

print(
    f"K=2 ARI: {ari_2:.4f}"
)

print(
    f"K=6 ARI: {ari_6:.4f}"
)

print(
    f"K=2 NMI: {nmi_2:.4f}"
)

print(
    f"K=6 NMI: {nmi_6:.4f}"
)

print("\nPlots saved:")
print("elbow_method.png")
print("silhouette_scores.png")
print("kmeans_k2_pca.png")
print("kmeans_k6_pca.png")