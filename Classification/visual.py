import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# 1. Load and prepare the data
df = pd.read_csv('propublica_data_for_fairml.csv')
X = df.drop(columns=['Two_yr_Recidivism'])
y = df['Two_yr_Recidivism']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the SVM (Must set probability=True for ROC/PR curves)
svm = SVC(C=50, gamma='scale', kernel='rbf', random_state=42, probability=True)
svm.fit(X_train_scaled, y_train)

# Extract raw probabilities for the positive class (Recidivism = 1)
y_prob = svm.predict_proba(X_test_scaled)[:, 1]

# 4. Create an evaluation DataFrame to isolate subgroups
eval_df = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob,
    'African_American': X_test['African_American']
})

aa_mask = eval_df['African_American'] == 1
non_aa_mask = eval_df['African_American'] == 0

# 5. Calculate ROC Metrics
fpr_aa, tpr_aa, _ = roc_curve(eval_df[aa_mask]['Actual'], eval_df[aa_mask]['Probability'])
roc_auc_aa = auc(fpr_aa, tpr_aa)

fpr_non, tpr_non, _ = roc_curve(eval_df[non_aa_mask]['Actual'], eval_df[non_aa_mask]['Probability'])
roc_auc_non = auc(fpr_non, tpr_non)

# 6. Calculate Precision-Recall Metrics
prec_aa, rec_aa, _ = precision_recall_curve(eval_df[aa_mask]['Actual'], eval_df[aa_mask]['Probability'])
ap_aa = average_precision_score(eval_df[aa_mask]['Actual'], eval_df[aa_mask]['Probability'])

prec_non, rec_non, _ = precision_recall_curve(eval_df[non_aa_mask]['Actual'], eval_df[non_aa_mask]['Probability'])
ap_non = average_precision_score(eval_df[non_aa_mask]['Actual'], eval_df[non_aa_mask]['Probability'])

# 7. Generate the Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot ROC Curves
ax1.plot(fpr_aa, tpr_aa, color='tab:red', lw=2, label=f'African American (AUC = {roc_auc_aa:.2f})')
ax1.plot(fpr_non, tpr_non, color='tab:blue', lw=2, label=f'Non-African American (AUC = {roc_auc_non:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('Subgroup ROC Curves', fontsize=14)
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

# Plot Precision-Recall Curves
ax2.plot(rec_aa, prec_aa, color='tab:red', lw=2, label=f'African American (AP = {ap_aa:.2f})')
ax2.plot(rec_non, prec_non, color='tab:blue', lw=2, label=f'Non-African American (AP = {ap_non:.2f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Subgroup Precision-Recall Curves', fontsize=14)
ax2.legend(loc="lower left")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()