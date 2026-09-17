import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load and prepare
df = pd.read_csv('propublica_data_for_fairml.csv')
X = df.drop(columns=['Two_yr_Recidivism'])
y = df['Two_yr_Recidivism']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale the data (Mandatory for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Set up the KNN parameter grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25, 31],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("Running Grid Search for KNN...")
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='f1')
grid_knn.fit(X_train_scaled, y_train)

# 4. Extract Best Model
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)

# 5. Print Final Results
print("\n--- Optimized KNN Results ---")
print(f"Best Parameters: {grid_knn.best_params_}")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))