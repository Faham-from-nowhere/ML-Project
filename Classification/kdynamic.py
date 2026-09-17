import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the data
df = pd.read_csv('propublica_data_for_fairml.csv')
X = df.drop(columns=['Two_yr_Recidivism'])

# 2. Scale the features (Mandatory)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply Dynamic PCA 
# Instead of hardcoding 3 dimensions, we tell the math to keep 95% of the variance
dynamic_pca = PCA(n_components=0.95, random_state=42)
X_pca_dynamic = dynamic_pca.fit_transform(X_scaled)

print("--- Dynamic PCA Results ---")
print(f"Original Features: {X_scaled.shape[1]}")
print(f"Features retained to keep 95% variance: {X_pca_dynamic.shape[1]}")

# 4. Run K-Means on the dynamically compressed data
km_dynamic = KMeans(n_clusters=2, random_state=42, n_init=10)
dynamic_labels = km_dynamic.fit_predict(X_pca_dynamic)

# 5. Evaluate the cluster density
dynamic_silhouette = silhouette_score(X_pca_dynamic, dynamic_labels)
print(f"Optimized Silhouette Score: {dynamic_silhouette:.4f}")