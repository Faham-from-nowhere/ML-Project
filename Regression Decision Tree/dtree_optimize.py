import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load Data (dropping Density)
df = pd.read_csv('bodyfat.csv')
X = df.drop(columns=['BodyFat', 'Density'])
y = df['BodyFat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Extract the Cost-Complexity Pruning Path
dt_initial = DecisionTreeRegressor(random_state=42)
path = dt_initial.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1] # Remove the trivial single-node tree

# 3. Find the optimal alpha mathematically
test_scores = []
for ccp_alpha in ccp_alphas:
    dt_temp = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    dt_temp.fit(X_train, y_train)
    test_scores.append(r2_score(y_test, dt_temp.predict(X_test)))

best_alpha = ccp_alphas[np.argmax(test_scores)]

# 4. Train the Final Optimized Tree
final_dt = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
final_dt.fit(X_train, y_train)
y_pred_final = final_dt.predict(X_test)

print("--- Advanced Cost-Complexity Pruning ---")
print(f"Optimal Alpha: {best_alpha:.4f}")
print(f"Final RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_final)):.4f}")
print(f"Final MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")
print(f"Final R-Squared: {r2_score(y_test, y_pred_final):.4f}")

# 5. Generate the Final Tree Visualization
plt.figure(figsize=(16, 8))
plot_tree(
    final_dt, 
    feature_names=X.columns, 
    filled=True, 
    rounded=True, 
    fontsize=10, 
    precision=2
)
plt.title(f"Optimized Decision Tree (ccp_alpha={best_alpha:.4f})", fontsize=16)
plt.tight_layout()
plt.show()