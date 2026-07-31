"""
Script to identify which seeds provide the best R² for each validation method
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings 

warnings.filterwarnings('ignore')

# Seeds to test
SEEDS = [42, 50, 100, 123, 200, 250, 300, 350, 400, 500] 
  
# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.3,
        'colsample_bytree': 1.0,
        'min_child_weight': 1,
        'gamma': 0
}

print("="*80)
print("FINDING BEST SEEDS FOR EACH VALIDATION METHOD")
print("="*80)
print(f"\nTesting seeds: {SEEDS}")
print(f"Model parameters: {MODEL_PARAMS}\n")

# Load and prepare data
print("Loading data...")
df = pd.read_excel("Random Forest Data.xlsx", header=1, usecols="B:L")
stream = df[df["Rain or Stream"].str.lower().eq("stream")].copy()

# Feature engineering
stream['Month'] = stream['Date'].dt.month
stream['Day_of_Year'] = stream['Date'].dt.dayofyear
stream['Year'] = stream['Date'].dt.year
stream["DOY_sin"] = np.sin(2 * np.pi * stream['Day_of_Year'] / 365.25)
stream["DOY_cos"] = np.cos(2 * np.pi * stream['Day_of_Year'] / 365.25)
stream['Temp_Discharge_Interaction'] = stream['Temperature '] * stream['Discharge (m3/s)']
stream['Rain'] = stream['Precipitation (mm)']
stream['Elevation'] = stream['Elevation (ft)']

# Define features
features = [
    "Site ID", 
    "Temperature ", 
    "Discharge (m3/s)",
    "DOY_sin",
    "DOY_cos",
    "Temp_Discharge_Interaction",
    "Rain",
    "Elevation"
]
X = stream[features]
y = stream["δ18O"]

print(f"Data loaded: {len(X)} samples, {len(features)} features\n")

# Storage for results
test_r2_values = []
kfold_r2_values = []
loocv_r2_values = []

test_mae_values = []
kfold_mae_values = []
loocv_mae_values = []

print("Running analysis for each seed...")
print("-"*80)

for seed in SEEDS:
    print(f"Testing seed {seed}...", end=" ")
    
    # 1. TEST SET VALIDATION (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    model = XGBRegressor(**MODEL_PARAMS, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2_values.append(test_r2)
    test_mae_values.append(test_mae)
    
    # 2. K-FOLD CROSS-VALIDATION (Time Series Split, 3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    y_true_kfold = []
    y_pred_kfold = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        model_fold = XGBRegressor(**MODEL_PARAMS, random_state=seed, n_jobs=-1)
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fold.predict(X_val_fold)
        
        y_true_kfold.extend(y_val_fold)
        y_pred_kfold.extend(y_pred_fold)
    
    kfold_r2 = r2_score(y_true_kfold, y_pred_kfold)
    kfold_mae = mean_absolute_error(y_true_kfold, y_pred_kfold)
    kfold_r2_values.append(kfold_r2)
    kfold_mae_values.append(kfold_mae)
    
    # 3. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
    y_true_loocv = []
    y_pred_loocv = []
    
    for i in range(len(X)):
        X_train_loo = X.drop(X.index[i])
        X_test_loo = X.iloc[[i]]
        y_train_loo = y.drop(y.index[i])
        y_test_loo = y.iloc[i]
        
        model_loo = XGBRegressor(**MODEL_PARAMS, random_state=seed, n_jobs=-1)
        model_loo.fit(X_train_loo, y_train_loo)
        y_pred_loo = model_loo.predict(X_test_loo)[0]
        
        y_true_loocv.append(y_test_loo)
        y_pred_loocv.append(y_pred_loo)
    
    loocv_r2 = r2_score(y_true_loocv, y_pred_loocv)
    loocv_mae = mean_absolute_error(y_true_loocv, y_pred_loocv)
    loocv_r2_values.append(loocv_r2)
    loocv_mae_values.append(loocv_mae)
    
    print(f"Test R²: {test_r2:.4f}, K-Fold R²: {kfold_r2:.4f}, LOOCV R²: {loocv_r2:.4f}")

print("-"*80)

# Find best seeds
best_test_idx = np.argmax(test_r2_values)
best_kfold_idx = np.argmax(kfold_r2_values)
best_loocv_idx = np.argmax(loocv_r2_values)

best_test_seed = SEEDS[best_test_idx]
best_kfold_seed = SEEDS[best_kfold_idx]
best_loocv_seed = SEEDS[best_loocv_idx]

# Print results
print("\n" + "="*80)
print("BEST SEEDS FOR EACH VALIDATION METHOD")
print("="*80)

print("\n1. TEST SET VALIDATION (80-20 Split)")
print(f"   Best Seed: {best_test_seed}")
print(f"   Best R²: {test_r2_values[best_test_idx]:.4f}")
print(f"   Corresponding MAE: {test_mae_values[best_test_idx]:.4f} ‰")

print("\n2. K-FOLD CROSS-VALIDATION (Time Series Split, 3 folds)")
print(f"   Best Seed: {best_kfold_seed}")
print(f"   Best R²: {kfold_r2_values[best_kfold_idx]:.4f}")
print(f"   Corresponding MAE: {kfold_mae_values[best_kfold_idx]:.4f} ‰")

print("\n3. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print(f"   Best Seed: {best_loocv_seed}")
print(f"   Best R²: {loocv_r2_values[best_loocv_idx]:.4f}")
print(f"   Corresponding MAE: {loocv_mae_values[best_loocv_idx]:.4f} ‰")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nTest Set Validation:")
print(f"   Mean R²: {np.mean(test_r2_values):.4f} ± {np.std(test_r2_values):.4f}")
print(f"   Range: [{np.min(test_r2_values):.4f}, {np.max(test_r2_values):.4f}]")

print("\nK-Fold Cross-Validation:")
print(f"   Mean R²: {np.mean(kfold_r2_values):.4f} ± {np.std(kfold_r2_values):.4f}")
print(f"   Range: [{np.min(kfold_r2_values):.4f}, {np.max(kfold_r2_values):.4f}]")

print("\nLOOCV:")
print(f"   Mean R²: {np.mean(loocv_r2_values):.4f} ± {np.std(loocv_r2_values):.4f}")
print(f"   Range: [{np.min(loocv_r2_values):.4f}, {np.max(loocv_r2_values):.4f}]")

print("\n" + "="*80)
print("DETAILED RESULTS TABLE")
print("="*80)
print(f"\n{'Seed':<8} {'Test R²':<12} {'Test MAE':<12} {'K-Fold R²':<12} {'K-Fold MAE':<12} {'LOOCV R²':<12} {'LOOCV MAE':<12}")
print("-"*80)
for i, seed in enumerate(SEEDS):
    marker_test = " ★" if i == best_test_idx else ""
    marker_kfold = " ★" if i == best_kfold_idx else ""
    marker_loocv = " ★" if i == best_loocv_idx else ""
    
    print(f"{seed:<8} {test_r2_values[i]:<12.4f}{marker_test:<2} {test_mae_values[i]:<12.4f} "
          f"{kfold_r2_values[i]:<12.4f}{marker_kfold:<2} {kfold_mae_values[i]:<12.4f} "
          f"{loocv_r2_values[i]:<12.4f}{marker_loocv:<2} {loocv_mae_values[i]:<12.4f}")

print("-"*80)
print("★ = Best performing seed for that validation method")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results to file
results_dict = {
    'seeds': SEEDS,
    'test_r2': test_r2_values,
    'test_mae': test_mae_values,
    'kfold_r2': kfold_r2_values,
    'kfold_mae': kfold_mae_values,
    'loocv_r2': loocv_r2_values,
    'loocv_mae': loocv_mae_values,
    'best_seeds': {
        'test_set': {'seed': best_test_seed, 'r2': test_r2_values[best_test_idx], 'mae': test_mae_values[best_test_idx]},
        'kfold': {'seed': best_kfold_seed, 'r2': kfold_r2_values[best_kfold_idx], 'mae': kfold_mae_values[best_kfold_idx]},
        'loocv': {'seed': best_loocv_seed, 'r2': loocv_r2_values[best_loocv_idx], 'mae': loocv_mae_values[best_loocv_idx]}
    }
}

# Save to CSV
results_df = pd.DataFrame({
    'Seed': SEEDS,
    'Test_R2': test_r2_values,
    'Test_MAE': test_mae_values,
    'KFold_R2': kfold_r2_values,
    'KFold_MAE': kfold_mae_values,
    'LOOCV_R2': loocv_r2_values,
    'LOOCV_MAE': loocv_mae_values
})

results_df.to_csv('best_seeds_analysis.csv', index=False)
print("\nResults saved to 'best_seeds_analysis.csv'")
