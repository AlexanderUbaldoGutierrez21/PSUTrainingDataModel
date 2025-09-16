# PSU Data Exploration and Regression Analysis - Final Report

## Project Overview
This project involves data exploration, preprocessing, and regression modeling on a dataset (`DataTrain_HW2Problem1.csv`) containing features `x1`, `x2`, `x3` and target `y`. The goal is to build and compare Linear Regression, Lasso (L1), and Ridge (L2) models, analyze their coefficients, and visualize the results.

## Data Description
- **Dataset**: 30 samples with 4 columns (y, x1, x2, x3)
- **Features**: x1, x2 (continuous), x3 (binary)
- **Target**: y (continuous)
- **Statistics**:
  - y: mean=34.25, std=47.32, range=[-14.4, 137.0]
  - x1: mean=0.63, std=5.83
  - x2: mean=0.33, std=1.72
  - x3: 37% ones, 63% zeros

## Methodology
1. **Data Loading and Inspection**: Loaded CSV, inspected structure and statistics.
2. **Data Splitting**: 80% train, 20% test (random_state=42).
3. **Feature Scaling**: Standardized features using StandardScaler.
4. **Model Training**:
   - Linear Regression
   - Lasso (initial alpha=0.1)
   - Ridge (initial alpha=1.0)
5. **Hyperparameter Tuning**: Used cross-validation to find optimal alphas.
6. **Evaluation**: R² scores on test set.
7. **Coefficient Analysis**: Compared coefficients across models.
8. **Visualization**: Added plots for data distribution, coefficients, and predictions.

## Results
- **Model Performance (R² on Test Set)**:
  - Linear Regression: 0.6687
  - Lasso: 0.6665
  - Ridge: 0.6422
- **Optimal Alphas**:
  - Lasso: 3.5112
  - Ridge: 4.6416
- **Coefficients**:
  - Linear: [-37.57, -4.84, 6.17]
  - Lasso: [-34.42, -2.46, 2.31] (some shrinkage)
  - Ridge: [-30.99, -5.98, 4.50] (moderate shrinkage)

## Visualizations
- Feature distributions (histograms)
- Boxplots for features
- Scatter plots (x1 vs y, x2 vs y)
- Coefficient comparison bar chart
- Predictions vs actual scatter plots for each model

## Key Insights
- Lasso shows more coefficient shrinkage than Ridge, indicating feature selection.
- x1 has the largest coefficient magnitude, suggesting strongest impact.
- Models perform similarly, with Linear Regression slightly better.

## Code and Dependencies
- **Main Script**: `PSUData_Exploration.py`
- **Dependencies**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Data File**: `DataTrain_HW2Problem1.csv`

## Fixes Applied
- Installed scikit-learn to resolve import issues.
- Corrected data file extension from .cvs to .csv.
- Added coefficient analysis and visualizations.

## Repository
All code and changes are committed to GitHub: https://github.com/AlexanderUbaldoGutierrez21/PSUTrainingDataModel