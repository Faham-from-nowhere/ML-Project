import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 1. Load and prepare the full dataset for unsupervised learning
df = pd.read_csv("propublica_data_for_fairml.csv")
X = df.drop(columns=['Two_yr_Recidivism'])

# 2. Scale the data (Mandatory for distance-based clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Calculate metrics for K=2 through K=10
k_values = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# 4. Generate the Visualization for the Report
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Inertia (Elbow Method)
color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (WCSS)', color=color)
ax1.plot(k_values, inertias, marker='o', color=color, linewidth=2, label="Inertia")
ax1.tick_params(axis='y', labelcolor=color)

# Plot Silhouette Score on the same graph
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color)  
ax2.plot(k_values, silhouette_scores, marker='s', color=color, linewidth=2, linestyle='--', label="Silhouette Score")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Optimization: Elbow Method & Silhouette Scores')
fig.tight_layout()  
plt.show()

# 5. Apply the Optimal Model (k=2) and analyze the groups
final_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Assigned_Cluster'] = final_kmeans.fit_predict(X_scaled)

# 6. Profile the sociological bias inside the mathematical clusters
print("--- K-Means Cluster Profiling (k=2) ---")
cluster_summary = df.groupby('Assigned_Cluster')[['Two_yr_Recidivism', 'Number_of_Priors', 'African_American', 'Female']].mean()
print(cluster_summary.round(3))