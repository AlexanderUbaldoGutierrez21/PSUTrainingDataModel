import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv('Data_Train_HW2Problem1.csv')

# Initial Inspection
print("First 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nData description:")
print(df.describe())

# Split Data
X = df[['x1', 'x2', 'x3']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training features (first 5 rows):")
print(X_train_scaled[:5])
print("\nScaled testing features (first 5 rows):")
print(X_test_scaled[:5])