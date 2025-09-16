import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv('DataTrain_HW2Problem1.csv')

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
# Model Implementation and Training

# 1. Standard Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 2. Lasso Regression (L1)
lasso_model = Lasso(alpha=0.1)  # Alpha value needs to be tuned
lasso_model.fit(X_train_scaled, y_train)

# 3. Ridge Regression (L2)
ridge_model = Ridge(alpha=1.0)  # Alpha value needs to be tuned
ridge_model.fit(X_train_scaled, y_train)

# Model Evaluation
# Predict on the test set
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Calculate and print metrics
print(f"\nLinear Regression R2: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Lasso R2: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Ridge R2: {r2_score(y_test, y_pred_ridge):.4f}")

# Hyperparameter Tuning
# Define a range of alphas to test
alphas = np.logspace(-4, 2, 100)

# Use cross-validation to find the best alpha for Lasso
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Optimal Lasso alpha: {lasso_cv.alpha_:.4f}")

# Use cross-validation to find the best alpha for Ridge
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Optimal Ridge alpha: {ridge_cv.alpha_:.4f}")

# Analyze Coefficients
print("Linear Regression Coefficients:", lr_model.coef_)
print("Lasso Coefficients:", lasso_cv.coef_)  # Using the model with the best alpha
print("Ridge Coefficients:", ridge_cv.coef_)  # Using the model with the best alpha
# Set style for beautiful plots
sns.set(style="whitegrid")

# 1. Data Distribution Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
df[['x1', 'x2', 'x3']].hist(ax=axes[0,0], bins=20)
axes[0,0].set_title('Feature Distributions')
sns.boxplot(data=df[['x1', 'x2', 'x3']], ax=axes[0,1])
axes[0,1].set_title('Feature Boxplots')
sns.scatterplot(data=df, x='x1', y='y', ax=axes[1,0])
axes[1,0].set_title('x1 vs y')
sns.scatterplot(data=df, x='x2', y='y', ax=axes[1,1])
axes[1,1].set_title('x2 vs y')
plt.tight_layout()
plt.show()

# 2. Coefficient Comparison Bar Plot
models = ['Linear Regression', 'Lasso', 'Ridge']
coefficients = [lr_model.coef_, lasso_cv.coef_, ridge_cv.coef_]
features = ['x1', 'x2', 'x3']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(features))
width = 0.25
for i, (model, coef) in enumerate(zip(models, coefficients)):
    ax.bar(x + i*width, coef, width, label=model)
ax.set_xlabel('Features')
ax.set_ylabel('Coefficients')
ax.set_title('Model Coefficients Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(features)
ax.legend()
plt.show()

# 3. Predictions vs Actual Scatter Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (model_name, y_pred) in enumerate([('Linear', y_pred_lr), ('Lasso', y_pred_lasso), ('Ridge', y_pred_ridge)]):
    axes[i].scatter(y_test, y_pred, alpha=0.7)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_xlabel('Actual y')
    axes[i].set_ylabel('Predicted y')
    axes[i].set_title(f'{model_name} Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()