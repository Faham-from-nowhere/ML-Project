import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# 1. Load, split, scale
df = pd.read_csv('propublica_data_for_fairml.csv')
X = df.drop(columns=['Two_yr_Recidivism'])
y = df['Two_yr_Recidivism']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train the Optimized SVM
svm = SVC(C=50, gamma='scale', kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# 3. Create an Audit DataFrame
test_df = X_test.copy()
test_df['Actual'] = y_test
test_df['Predicted'] = y_pred_svm

# 4. Define Error Calculation Function
def calculate_error_rates(subset):
    cm = confusion_matrix(subset['Actual'], subset['Predicted'])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fpr, fnr

# 5. Split by Demographics and Calculate
aa_group = test_df[test_df['African_American'] == 1]
non_aa_group = test_df[test_df['African_American'] == 0]

fpr_aa, fnr_aa = calculate_error_rates(aa_group)
fpr_non, fnr_non = calculate_error_rates(non_aa_group)

print("--- SVM Fairness Audit ---")
print(f"African American - False Positive Rate: {fpr_aa:.4f} | False Negative Rate: {fnr_aa:.4f}")
print(f"Non-African American - False Positive Rate: {fpr_non:.4f} | False Negative Rate: {fnr_non:.4f}")