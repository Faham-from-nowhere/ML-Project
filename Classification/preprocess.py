import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("propublica_data_for_fairml.csv")

# 2. Separate the Features (X) from the Target (y)
X = df.drop(columns=['Two_yr_Recidivism'])
y = df['Two_yr_Recidivism']

# 3. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the data
# I fit the scaler ONLY on the training data to prevent data leakage,
# then I transformed both the training and testing sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For K-Means (which is unsupervised and doesn't use train/test splits in the same way), 
# I scaled the entire feature set at once.
X_scaled_full = scaler.fit_transform(X)

print("Data successfully loaded and scaled!")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")