import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load and prepare
df = pd.read_csv('propublica_data_for_fairml.csv')
X = df.drop(columns=['Two_yr_Recidivism'])
y = df['Two_yr_Recidivism']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale the data (Mandatory for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Optimize SVM with RBF Kernel
param_grid_svm = {
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 0.01, 0.1, 1]
}

print("Running Grid Search for SVM (This might take a minute)...")
grid_svm = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid_svm, cv=5, scoring='f1')
grid_svm.fit(X_train_scaled, y_train)

# 4. Extract Best Model
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

print("\n--- Optimized SVM (RBF) Results ---")
print(f"Best Parameters: {grid_svm.best_params_}")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))