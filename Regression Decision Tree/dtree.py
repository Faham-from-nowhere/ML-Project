import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data and Prevent Data Leakage
df = pd.read_csv('bodyfat.csv')
# Drop Density so the model actually has to learn from the physical measurements
X = df.drop(columns=['BodyFat', 'Density']) 
y = df['BodyFat']

# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL 1: Baseline Decision Tree
# A default tree grows infinitely until it perfectly memorizes the training data.
dt_base = DecisionTreeRegressor(random_state=42)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)

print("--- Baseline Decision Tree ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_base):.4f}")
print(f"R-Squared: {r2_score(y_test, y_pred_base):.4f}")


# MODEL 2: Optimized Decision Tree (Pruning)
# I used GridSearch to find the best limits to stop the tree from overfitting
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

print("\nRunning Grid Search Optimization...")
grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
y_pred_best = best_dt.predict(X_test)

print("\n--- Optimized Decision Tree ---")
print(f"Best Parameters: {grid_dt.best_params_}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"R-Squared: {r2_score(y_test, y_pred_best):.4f}")

# ANALYSIS: Feature Importance
importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_dt.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(importances.head(5))
