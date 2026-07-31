import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load data
df = pd.read_excel('Random Forest Data.xlsx', header=1, usecols='B:J')
stream = df[df['Rain or Stream'].str.lower().eq('stream')].copy()

# Feature engineering
stream['Month'] = stream['Date'].dt.month
stream['Day_of_Year'] = stream['Date'].dt.dayofyear
stream['Year'] = stream['Date'].dt.year
stream['Temp_Discharge_Interaction'] = stream['Temperature '] * stream['Discharge (m3/s)']
stream['d_excess'] = stream['d-excess (‰)']
stream['d2H'] = stream['δ2H']

# Define feature sets
feature_col_basic = ['Site ID', 'Temperature ', 'Discharge (m3/s)']
feature_col_extended = ['Site ID', 'Temperature ', 'Discharge (m3/s)', 'Month', 'Day_of_Year', 'd_excess', 'd2H', 'Temp_Discharge_Interaction']

# Prepare data
X_basic = stream[feature_col_basic]
X_extended = stream[feature_col_extended]
y = stream['δ18O']

print("="*70)
print("ANALYZING WHY BASELINE MODEL MIGHT PERFORM BETTER")
print("="*70)

# Check correlations
print("\n1. FEATURE CORRELATIONS WITH δ18O:")
print("-" * 70)
correlations = stream[feature_col_extended + ['δ18O']].corr()['δ18O'].sort_values(ascending=False)
print(correlations)

# Check multicollinearity
print("\n2. CHECKING FOR MULTICOLLINEARITY:")
print("-" * 70)
print("Correlation between δ2H and δ18O:", stream[['δ2H', 'δ18O']].corr().iloc[0, 1])
print("Correlation between d-excess and δ18O:", stream[['d-excess (‰)', 'δ18O']].corr().iloc[0, 1])
print("\nNote: δ2H and d-excess are DERIVED from δ18O in many cases!")
print("This creates data leakage if they're measured together.")

# Split data
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_basic, y, test_size=0.2, random_state=42, shuffle=True
)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_extended, y, test_size=0.2, random_state=42, shuffle=True
)

# Train models
print("\n3. MODEL COMPARISON:")
print("-" * 70)

rf_basic = RandomForestRegressor(random_state=42, n_estimators=100)
rf_basic.fit(X_train_b, y_train_b)
y_pred_basic = rf_basic.predict(X_test_b)

rf_extended = RandomForestRegressor(random_state=42, n_estimators=100)
rf_extended.fit(X_train_e, y_train_e)
y_pred_extended = rf_extended.predict(X_test_e)

print('\nBASIC MODEL (3 features):')
print(f'  R²:   {r2_score(y_test_b, y_pred_basic):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test_b, y_pred_basic)):.4f}')
print(f'  MAE:  {mean_absolute_error(y_test_b, y_pred_basic):.4f}')

print('\nEXTENDED MODEL (8 features):')
print(f'  R²:   {r2_score(y_test_e, y_pred_extended):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test_e, y_pred_extended)):.4f}')
print(f'  MAE:  {mean_absolute_error(y_test_e, y_pred_extended):.4f}')

# Cross-validation
print("\n4. CROSS-VALIDATION SCORES (More Reliable):")
print("-" * 70)
cv_basic = cross_val_score(rf_basic, X_basic, y, cv=5, scoring='r2')
cv_extended = cross_val_score(rf_extended, X_extended, y, cv=5, scoring='r2')

print(f'\nBasic Model CV R²:    {cv_basic.mean():.4f} (+/- {cv_basic.std() * 2:.4f})')
print(f'Extended Model CV R²: {cv_extended.mean():.4f} (+/- {cv_extended.std() * 2:.4f})')

# Feature importance
print("\n5. FEATURE IMPORTANCE (Extended Model):")
print("-" * 70)
importance_df = pd.DataFrame({
    'feature': feature_col_extended,
    'importance': rf_extended.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))

# Test without isotope features (to avoid data leakage)
print("\n6. MODEL WITHOUT ISOTOPE FEATURES (No Data Leakage):")
print("-" * 70)
feature_col_no_isotopes = ['Site ID', 'Temperature ', 'Discharge (m3/s)', 'Month', 'Day_of_Year', 'Temp_Discharge_Interaction']
X_no_iso = stream[feature_col_no_isotopes]
X_train_ni, X_test_ni, y_train_ni, y_test_ni = train_test_split(
    X_no_iso, y, test_size=0.2, random_state=42, shuffle=True
)

rf_no_iso = RandomForestRegressor(random_state=42, n_estimators=100)
rf_no_iso.fit(X_train_ni, y_train_ni)
y_pred_no_iso = rf_no_iso.predict(X_test_ni)

print('\nModel WITHOUT isotope features (6 features):')
print(f'  R²:   {r2_score(y_test_ni, y_pred_no_iso):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test_ni, y_pred_no_iso)):.4f}')
print(f'  MAE:  {mean_absolute_error(y_test_ni, y_pred_no_iso):.4f}')

cv_no_iso = cross_val_score(rf_no_iso, X_no_iso, y, cv=5, scoring='r2')
print(f'  CV R²: {cv_no_iso.mean():.4f} (+/- {cv_no_iso.std() * 2:.4f})')

print("\n" + "="*70)
print("CONCLUSIONS:")
print("="*70)
print("""
If the baseline model performs better, it's likely due to:

1. OVERFITTING: Extended model has more features but limited data (n=123)
   - More features = more complexity = easier to overfit
   - Random Forest can overfit with too many features on small datasets

2. DATA LEAKAGE: δ2H and d-excess are related to δ18O
   - If measured together, they contain information about the target
   - This is "cheating" - you wouldn't have these in real predictions
   
3. IRRELEVANT FEATURES: Some added features may add noise
   - Temporal features (Month, Day_of_Year) might not be predictive
   - Interaction terms might not capture real relationships
   
4. SMALL DATASET: With only 123 samples:
   - Simpler models often generalize better
   - Complex models memorize training data
   
RECOMMENDATIONS:
- Use the model WITHOUT isotope features for fair comparison
- Focus on collecting more data
- Use feature selection to remove unhelpful features
- Consider regularization or simpler models
""")
