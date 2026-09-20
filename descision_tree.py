import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("bodyfat.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDataset statistics:\n", df.describe())


# =========================
# 2. VISUALIZATION
# =========================

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("tree_correlation_heatmap.png")
plt.close()

correlations = df.corr(numeric_only=True)["BodyFat"].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title("Feature Correlation with Body Fat")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("tree_feature_correlation.png")
plt.close()


# =========================
# 3. PREPROCESSING
# =========================

X = df.drop(columns=["BodyFat"])
y = df["BodyFat"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget:")
print("BodyFat")


# =========================
# 4. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 5. K-FOLD CROSS VALIDATION
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================
# 6. DECISION TREE PIPELINE
# =========================

pipeline = Pipeline([
    ("feature_selection", SelectKBest(score_func=f_regression)),
    ("model", DecisionTreeRegressor(random_state=42))
])


# =========================
# 7. HYPERPARAMETER TUNING
# =========================

params = {
    "feature_selection__k": [5, 8, 10, 12, 14, "all"],
    "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)


# =========================
# 8. CROSS-VALIDATION RESULTS
# =========================

print("\n======================================")
print("CROSS-VALIDATION RESULTS")
print("======================================")

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV RMSE:")
print(-grid.best_score_)


# =========================
# 9. FINAL TESTING
# =========================

model = grid.best_estimator_

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# =========================
# 10. SELECTED FEATURES
# =========================

selector = model.named_steps["feature_selection"]

selected_features = X.columns[selector.get_support()]

print("\n======================================")
print("SELECTED FEATURES")
print("======================================")

for feature in selected_features:
    print(feature)


# =========================
# 11. ADJUSTED R²
# =========================

n = len(y_test)
p = len(selected_features)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))


# =========================
# 12. DECISION TREE RESULTS
# =========================

print("\n======================================")
print("DECISION TREE REGRESSION RESULTS")
print("======================================")

print(f"MAE         : {mae:.4f}")
print(f"MSE         : {mse:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adjusted_r2:.4f}")


# =========================
# 13. FEATURE IMPORTANCE
# =========================

tree_model = model.named_steps["model"]

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": tree_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n======================================")
print("FEATURE IMPORTANCE")
print("======================================")

print(importance_df.to_string(index=False))


# =========================
# 14. FEATURE IMPORTANCE PLOT
# =========================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig("tree_feature_importance.png")
plt.close()


# =========================
# 15. ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(6, 5))

plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Body Fat")
plt.ylabel("Predicted Body Fat")
plt.title("Decision Tree: Actual vs Predicted")

plt.tight_layout()
plt.savefig("tree_actual_vs_predicted.png")
plt.close()


# =========================
# 16. RESIDUAL PLOT
# =========================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Body Fat")
plt.ylabel("Residual")
plt.title("Decision Tree: Residual Plot")

plt.tight_layout()
plt.savefig("tree_residuals.png")
plt.close()


# =========================
# 17. TREE VISUALIZATION
# =========================

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=selected_features,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.close()


# =========================
# 18. FINAL SUMMARY
# =========================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(f"Best Parameters : {grid.best_params_}")
print(f"CV RMSE         : {-grid.best_score_:.4f}")
print(f"Test MAE        : {mae:.4f}")
print(f"Test MSE        : {mse:.4f}")
print(f"Test RMSE       : {rmse:.4f}")
print(f"Test R²         : {r2:.4f}")
print(f"Adjusted R²     : {adjusted_r2:.4f}")

print("\nPlots saved:")
print("tree_correlation_heatmap.png")
print("tree_feature_correlation.png")
print("tree_feature_importance.png")
print("tree_actual_vs_predicted.png")
print("tree_residuals.png")
print("decision_tree.png")