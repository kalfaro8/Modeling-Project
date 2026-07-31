"""
Model Comparison Script: Random Forest vs XGBoost
This script trains both models and provides a comprehensive comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import time
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("MODEL COMPARISON: Random Forest vs XGBoost for δ18O Prediction")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading and preparing data...")

df = pd.read_excel("Random Forest Data.xlsx", header=1, usecols="B:J")
stream = df[df["Rain or Stream"].str.lower().eq("stream")].copy()

# Feature engineering
stream['Month'] = stream['Date'].dt.month
stream['Day_of_Year'] = stream['Date'].dt.dayofyear
stream['Year'] = stream['Date'].dt.year
stream['Temp_Discharge_Interaction'] = stream['Temperature '] * stream['Discharge (m3/s)']
stream['d_excess'] = stream['d-excess (‰)']
stream['d2H'] = stream['δ2H']

feature_col = [
    "Site ID", "Temperature ", "Discharge (m3/s)",
    "Month", "Day_of_Year", "d_excess", "d2H",
    "Temp_Discharge_Interaction"
]

X = stream[feature_col]
y = stream["δ18O"]

# Split data (same seed for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, shuffle=True
)

print(f"✓ Data loaded: {len(stream)} samples, {len(feature_col)} features")
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

# ============================================================================
# 2. TRAIN RANDOM FOREST
# ============================================================================
print("\n[2/6] Training Random Forest model...")

# Baseline Random Forest
start_time = time.time()
rf_baseline = RandomForestRegressor(random_state=50, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
rf_baseline_time = time.time() - start_time

y_pred_rf_baseline = rf_baseline.predict(X_test)
rf_baseline_r2 = r2_score(y_test, y_pred_rf_baseline)
rf_baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_baseline))
rf_baseline_mae = mean_absolute_error(y_test, y_pred_rf_baseline)

print(f"✓ Baseline RF trained in {rf_baseline_time:.2f}s")
print(f"  R²: {rf_baseline_r2:.4f}, RMSE: {rf_baseline_rmse:.4f}")

# Optimized Random Forest (smaller grid for speed)
print("  Tuning hyperparameters...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

start_time = time.time()
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=50, n_jobs=-1),
    rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)
rf_tuning_time = time.time() - start_time

rf_optimized = rf_grid.best_estimator_
y_pred_rf_opt = rf_optimized.predict(X_test)
rf_opt_r2 = r2_score(y_test, y_pred_rf_opt)
rf_opt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_opt))
rf_opt_mae = mean_absolute_error(y_test, y_pred_rf_opt)
rf_cv_scores = cross_val_score(rf_optimized, X, y, cv=5, scoring='r2')

print(f"✓ Optimized RF trained in {rf_tuning_time:.2f}s")
print(f"  R²: {rf_opt_r2:.4f}, RMSE: {rf_opt_rmse:.4f}")
print(f"  Best params: {rf_grid.best_params_}")

# ============================================================================
# 3. TRAIN XGBOOST
# ============================================================================
print("\n[3/6] Training XGBoost model...")

# Baseline XGBoost
start_time = time.time()
xgb_baseline = XGBRegressor(
    objective='reg:squarederror', random_state=50, n_jobs=-1
)
xgb_baseline.fit(X_train, y_train)
xgb_baseline_time = time.time() - start_time

y_pred_xgb_baseline = xgb_baseline.predict(X_test)
xgb_baseline_r2 = r2_score(y_test, y_pred_xgb_baseline)
xgb_baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_baseline))
xgb_baseline_mae = mean_absolute_error(y_test, y_pred_xgb_baseline)

print(f"✓ Baseline XGBoost trained in {xgb_baseline_time:.2f}s")
print(f"  R²: {xgb_baseline_r2:.4f}, RMSE: {xgb_baseline_rmse:.4f}")

# Optimized XGBoost (smaller grid for speed)
print("  Tuning hyperparameters...")
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

start_time = time.time()
xgb_grid = GridSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=50, n_jobs=-1),
    xgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
)
xgb_grid.fit(X_train, y_train)
xgb_tuning_time = time.time() - start_time

xgb_optimized = xgb_grid.best_estimator_
y_pred_xgb_opt = xgb_optimized.predict(X_test)
xgb_opt_r2 = r2_score(y_test, y_pred_xgb_opt)
xgb_opt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_opt))
xgb_opt_mae = mean_absolute_error(y_test, y_pred_xgb_opt)
xgb_cv_scores = cross_val_score(xgb_optimized, X, y, cv=5, scoring='r2')

print(f"✓ Optimized XGBoost trained in {xgb_tuning_time:.2f}s")
print(f"  R²: {xgb_opt_r2:.4f}, RMSE: {xgb_opt_rmse:.4f}")
print(f"  Best params: {xgb_grid.best_params_}")

# ============================================================================
# 4. FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n[4/6] Analyzing feature importance...")

rf_importance = pd.DataFrame({
    'feature': feature_col,
    'rf_importance': rf_optimized.feature_importances_
}).sort_values('rf_importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': feature_col,
    'xgb_importance': xgb_optimized.feature_importances_
}).sort_values('xgb_importance', ascending=False)

importance_comparison = rf_importance.merge(xgb_importance, on='feature')
print("✓ Feature importance calculated")

# ============================================================================
# 5. PRINT COMPARISON TABLE
# ============================================================================
print("\n[5/6] Generating comparison report...")
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

comparison_data = {
    'Metric': ['R² Score', 'RMSE (‰)', 'MAE (‰)', 'CV R² Mean', 'CV R² Std', 'Training Time (s)'],
    'RF Baseline': [
        f"{rf_baseline_r2:.4f}",
        f"{rf_baseline_rmse:.4f}",
        f"{rf_baseline_mae:.4f}",
        "N/A",
        "N/A",
        f"{rf_baseline_time:.2f}"
    ],
    'RF Optimized': [
        f"{rf_opt_r2:.4f}",
        f"{rf_opt_rmse:.4f}",
        f"{rf_opt_mae:.4f}",
        f"{rf_cv_scores.mean():.4f}",
        f"{rf_cv_scores.std():.4f}",
        f"{rf_tuning_time:.2f}"
    ],
    'XGB Baseline': [
        f"{xgb_baseline_r2:.4f}",
        f"{xgb_baseline_rmse:.4f}",
        f"{xgb_baseline_mae:.4f}",
        "N/A",
        "N/A",
        f"{xgb_baseline_time:.2f}"
    ],
    'XGB Optimized': [
        f"{xgb_opt_r2:.4f}",
        f"{xgb_opt_rmse:.4f}",
        f"{xgb_opt_mae:.4f}",
        f"{xgb_cv_scores.mean():.4f}",
        f"{xgb_cv_scores.std():.4f}",
        f"{xgb_tuning_time:.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Winner determination
if xgb_opt_r2 > rf_opt_r2:
    winner = "XGBoost"
    improvement = ((xgb_opt_r2 - rf_opt_r2) / rf_opt_r2) * 100
else:
    winner = "Random Forest"
    improvement = ((rf_opt_r2 - xgb_opt_r2) / xgb_opt_r2) * 100

print(f"\n🏆 WINNER: {winner} (R² improvement: {improvement:.2f}%)")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. R² Score Comparison
ax1 = fig.add_subplot(gs[0, 0])
models = ['RF\nBaseline', 'RF\nOptimized', 'XGB\nBaseline', 'XGB\nOptimized']
r2_values = [rf_baseline_r2, rf_opt_r2, xgb_baseline_r2, xgb_opt_r2]
colors = ['lightblue', 'steelblue', 'lightcoral', 'firebrick']
bars = ax1.bar(models, r2_values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison', fontweight='bold')
ax1.set_ylim([min(r2_values) - 0.1, 1.0])
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, r2_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 2. RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
rmse_values = [rf_baseline_rmse, rf_opt_rmse, xgb_baseline_rmse, xgb_opt_rmse]
bars = ax2.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('RMSE (‰)')
ax2.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, rmse_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 3. Training Time Comparison
ax3 = fig.add_subplot(gs[0, 2])
time_values = [rf_baseline_time, rf_tuning_time, xgb_baseline_time, xgb_tuning_time]
bars = ax3.bar(models, time_values, color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('Time (seconds)')
ax3.set_title('Training Time Comparison', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, time_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

# 4. RF Predicted vs Observed
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test, y_pred_rf_opt, alpha=0.6, edgecolors='k', color='steelblue')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Observed δ18O (‰)')
ax4.set_ylabel('Predicted δ18O (‰)')
ax4.set_title(f'Random Forest\nR²={rf_opt_r2:.4f}, RMSE={rf_opt_rmse:.4f}', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. XGBoost Predicted vs Observed
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_test, y_pred_xgb_opt, alpha=0.6, edgecolors='k', color='firebrick')
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Observed δ18O (‰)')
ax5.set_ylabel('Predicted δ18O (‰)')
ax5.set_title(f'XGBoost\nR²={xgb_opt_r2:.4f}, RMSE={xgb_opt_rmse:.4f}', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Residuals Comparison
ax6 = fig.add_subplot(gs[1, 2])
rf_residuals = y_test - y_pred_rf_opt
xgb_residuals = y_test - y_pred_xgb_opt
ax6.scatter(y_pred_rf_opt, rf_residuals, alpha=0.5, label='RF', color='steelblue', edgecolors='k')
ax6.scatter(y_pred_xgb_opt, xgb_residuals, alpha=0.5, label='XGB', color='firebrick', edgecolors='k')
ax6.axhline(y=0, color='black', linestyle='--', lw=2)
ax6.set_xlabel('Predicted δ18O (‰)')
ax6.set_ylabel('Residuals (‰)')
ax6.set_title('Residuals Comparison', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Feature Importance - Random Forest
ax7 = fig.add_subplot(gs[2, 0])
rf_imp_sorted = rf_importance.sort_values('rf_importance', ascending=True)
ax7.barh(rf_imp_sorted['feature'], rf_imp_sorted['rf_importance'], color='steelblue', alpha=0.8)
ax7.set_xlabel('Importance')
ax7.set_title('Random Forest Feature Importance', fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# 8. Feature Importance - XGBoost
ax8 = fig.add_subplot(gs[2, 1])
xgb_imp_sorted = xgb_importance.sort_values('xgb_importance', ascending=True)
ax8.barh(xgb_imp_sorted['feature'], xgb_imp_sorted['xgb_importance'], color='firebrick', alpha=0.8)
ax8.set_xlabel('Importance')
ax8.set_title('XGBoost Feature Importance', fontweight='bold')
ax8.grid(axis='x', alpha=0.3)

# 9. Cross-Validation Scores
ax9 = fig.add_subplot(gs[2, 2])
cv_data = {
    'Fold': list(range(1, 6)) * 2,
    'R² Score': list(rf_cv_scores) + list(xgb_cv_scores),
    'Model': ['Random Forest'] * 5 + ['XGBoost'] * 5
}
cv_df = pd.DataFrame(cv_data)
for model, color in [('Random Forest', 'steelblue'), ('XGBoost', 'firebrick')]:
    model_data = cv_df[cv_df['Model'] == model]
    ax9.plot(model_data['Fold'], model_data['R² Score'], 'o-', label=model, 
             color=color, linewidth=2, markersize=8, alpha=0.8)
ax9.set_xlabel('Fold')
ax9.set_ylabel('R² Score')
ax9.set_title('Cross-Validation Scores', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_xticks(range(1, 6))

plt.suptitle('Random Forest vs XGBoost: Comprehensive Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'model_comparison.png'")

plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)
print(f"\n🎯 Best Model: {winner}")
print(f"   R² Score: {max(rf_opt_r2, xgb_opt_r2):.4f}")
print(f"   RMSE: {min(rf_opt_rmse, xgb_opt_rmse):.4f} ‰")
print(f"   MAE: {min(rf_opt_mae, xgb_opt_mae):.4f} ‰")

print("\n📊 Key Findings:")
print(f"   • XGBoost R² is {((xgb_opt_r2 - rf_opt_r2) / rf_opt_r2 * 100):+.2f}% vs Random Forest")
print(f"   • XGBoost RMSE is {((xgb_opt_rmse - rf_opt_rmse) / rf_opt_rmse * 100):+.2f}% vs Random Forest")
print(f"   • XGBoost trains {(xgb_tuning_time / rf_tuning_time):.2f}x the time of Random Forest")

print("\n🔬 Top 3 Features (XGBoost):")
for idx, row in xgb_importance.head(3).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['xgb_importance']:.4f}")

print("\n💡 Recommendations:")
if xgb_opt_r2 > rf_opt_r2:
    print("   ✓ Use XGBoost for production (better performance)")
    print("   ✓ Consider ensemble stacking for even better results")
else:
    print("   ✓ Use Random Forest for production (better interpretability)")
    print("   ✓ XGBoost didn't improve performance significantly")

print("\n🚀 Next Steps:")
print("   1. Try ensemble stacking (RF + XGBoost + LightGBM)")
print("   2. Experiment with additional feature engineering")
print("   3. Collect more data if possible (current n=123)")
print("   4. Consider SHAP values for detailed feature analysis")
print("   5. Deploy the best model for production use")

print("\n" + "="*80)
print("Comparison complete! Check 'model_comparison.png' for visualizations.")
print("="*80)
