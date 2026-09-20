import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("bodyfat.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDataset statistics:")
print(df.describe())


# ============================================================
# 2. VISUALIZATION
# ============================================================

# Correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("ridge_correlation_heatmap.png")
plt.close()


# Correlation with target
correlations = df.corr(numeric_only=True)["BodyFat"].sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=correlations.values,
    y=correlations.index
)

plt.title("Feature Correlation with Body Fat")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("ridge_feature_correlation.png")
plt.close()


# ============================================================
# 3. FEATURE / TARGET SEPARATION
# ============================================================

# Keep Density because it is part of the provided dataset
X = df.drop(columns=["BodyFat"])
y = df["BodyFat"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. K-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. RIDGE PIPELINE
# ============================================================

pipeline = Pipeline([
    ("feature_selection", SelectKBest(score_func=f_regression)),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

params = {
    "feature_selection__k": [5, 8, 10, 12, 14, "all"],
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)


# ============================================================
# 8. CROSS-VALIDATION RESULTS
# ============================================================

print("\n======================================")
print("CROSS-VALIDATION RESULTS")
print("======================================")

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV RMSE:")
print(f"{-grid.best_score_:.4f}")


# ============================================================
# 9. FINAL MODEL
# ============================================================

model = grid.best_estimator_

y_pred = model.predict(X_test)


# ============================================================
# 10. BASIC EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 11. SELECTED FEATURES
# ============================================================

selector = model.named_steps["feature_selection"]

selected_features = X.columns[
    selector.get_support()
]

print("\n======================================")
print("SELECTED FEATURES")
print("======================================")

for feature in selected_features:
    print(feature)


# ============================================================
# 12. ADJUSTED R²
# ============================================================

n = len(y_test)
p = len(selected_features)

adjusted_r2 = 1 - (
    ((1 - r2) * (n - 1))
    / (n - p - 1)
)


# ============================================================
# 13. FINAL RIDGE RESULTS
# ============================================================

print("\n======================================")
print("RIDGE REGRESSION RESULTS")
print("======================================")

print(f"MAE         : {mae:.4f}")
print(f"MSE         : {mse:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adjusted_r2:.4f}")


# ============================================================
# 14. RIDGE COEFFICIENTS
# ============================================================

ridge_model = model.named_steps["model"]

coefficients = ridge_model.coef_

coefficient_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients
})

coefficient_df = coefficient_df.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\n======================================")
print("RIDGE COEFFICIENTS")
print("======================================")

print(
    coefficient_df.to_string(index=False)
)


# ============================================================
# 15. COEFFICIENT VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=coefficient_df,
    x="Coefficient",
    y="Feature"
)

plt.title("Ridge Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("ridge_coefficients.png")
plt.close()


# ============================================================
# 16. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Body Fat")
plt.ylabel("Predicted Body Fat")
plt.title("Ridge Regression: Actual vs Predicted")

plt.tight_layout()
plt.savefig("ridge_actual_vs_predicted.png")
plt.close()


# ============================================================
# 17. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(7, 6))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Body Fat")
plt.ylabel("Residual")
plt.title("Ridge Regression: Residual Plot")

plt.tight_layout()
plt.savefig("ridge_residuals.png")
plt.close()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    f"Best Alpha      : "
    f"{grid.best_params_['model__alpha']}"
)

print(
    f"Selected K      : "
    f"{grid.best_params_['feature_selection__k']}"
)

print(
    f"CV RMSE         : "
    f"{-grid.best_score_:.4f}"
)

print(
    f"Test MAE        : "
    f"{mae:.4f}"
)

print(
    f"Test MSE        : "
    f"{mse:.4f}"
)

print(
    f"Test RMSE       : "
    f"{rmse:.4f}"
)

print(
    f"Test R²         : "
    f"{r2:.4f}"
)

print(
    f"Adjusted R²     : "
    f"{adjusted_r2:.4f}"
)

print("\nPlots saved:")
print("ridge_correlation_heatmap.png")
print("ridge_feature_correlation.png")
print("ridge_coefficients.png")
print("ridge_actual_vs_predicted.png")
print("ridge_residuals.png")