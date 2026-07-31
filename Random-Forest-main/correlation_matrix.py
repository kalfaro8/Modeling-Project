import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load data
df = pd.read_excel("Random Forest Data.xlsx", header=1, usecols="B:L")

# Filter for stream data
stream = df[df["Rain or Stream"].str.lower().eq("stream")].copy()

# Extract temporal features from Date
stream['Month'] = stream['Date'].dt.month
stream['Day_of_Year'] = stream['Date'].dt.dayofyear
stream['Year'] = stream['Date'].dt.year
stream["DOY_sin"] = np.sin(2 * np.pi * stream['Day_of_Year'] / 365.25)
stream["DOY_cos"] = np.cos(2 * np.pi * stream['Day_of_Year'] / 365.25)

# Create interaction features
stream['Temp_Discharge_Interaction'] = stream['Temperature '] * stream['Discharge (m3/s)']

# Add other isotope data as features
stream['d_excess'] = stream['d-excess (‰)']
stream['d2H'] = stream['δ2H']

# Rain Feature
stream['Rain'] = stream['Precipitation (mm)']
stream['Elevation'] = stream['Elevation (ft)']

# Define feature columns (same as in the notebook)
feature_col = [
    "Site ID", 
    "Temperature ", 
    "Discharge (m3/s)",
    "DOY_sin",
    "DOY_cos",
    "Temp_Discharge_Interaction",
    "Rain",
    "Elevation"
]

# Add target variable for correlation analysis
features_with_target = feature_col + ['δ18O']

# Create correlation matrix
correlation_matrix = stream[features_with_target].corr()

# Create a more readable version with renamed columns
feature_labels = {
    'Site ID': 'Site ID',
    'Temperature ': 'Temperature',
    'Discharge (m3/s)': 'Discharge',
    'DOY_sin': 'DOY (Sin)',
    'DOY_cos': 'DOY (Cos)',
    'Temp_Discharge_Interaction': 'Temp x Discharge',
    'Rain': 'Precipitation',
    'Elevation': 'Elevation',
    'δ18O': 'd18O'
}

# Rename for display
correlation_matrix_display = correlation_matrix.rename(columns=feature_labels, index=feature_labels)

# Print correlation matrix
print("="*80)
print("CORRELATION MATRIX FOR RANDOM FOREST FEATURES")
print("="*80)
print("\nCorrelation Matrix:")
print(correlation_matrix_display.to_string())
print("\n" + "="*80)

# Print correlations with target variable (δ18O)
print("\nCorrelations with d18O (sorted by absolute value):")
print("="*80)
target_correlations = correlation_matrix['δ18O'].drop('δ18O').sort_values(key=abs, ascending=False)
for feature, corr in target_correlations.items():
    print(f"{feature_labels[feature]:25s}: {corr:7.4f}")
print("="*80)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 1. Full correlation matrix heatmap
sns.heatmap(correlation_matrix_display, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlation Matrix - All Features (d18O)', fontsize=18, fontweight='bold', pad=20)
axes[0].tick_params(axis='both', which='major', labelsize=11)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[0].get_yticklabels(), rotation=0)

# 2. Correlation with target variable (δ18O) - bar chart
target_corr_sorted = target_correlations.rename(feature_labels).sort_values()
colors = ['#DE8F05' if x > 0 else '#0173B2' for x in target_corr_sorted.values]
bars = axes[1].barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors, alpha=0.8)
axes[1].set_yticks(range(len(target_corr_sorted)))
axes[1].set_yticklabels(target_corr_sorted.index, fontsize=12)
axes[1].set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Feature', fontsize=14, fontweight='bold')
axes[1].set_title('Feature Correlations with d18O', fontsize=18, fontweight='bold', pad=20)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(axis='x', alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add value labels on bars
for i, (idx, val) in enumerate(target_corr_sorted.items()):
    axes[1].text(val + (0.02 if val > 0 else -0.02), i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\nCorrelation matrix visualization saved as 'correlation_matrix.png'")
plt.show()

# Additional statistics
print("\n" + "="*80)
print("CORRELATION STATISTICS")
print("="*80)
print(f"\nStrongest positive correlation with d18O:")
max_corr = target_correlations.max()
max_feature = target_correlations.idxmax()
print(f"  {feature_labels[max_feature]}: {max_corr:.4f}")

print(f"\nStrongest negative correlation with d18O:")
min_corr = target_correlations.min()
min_feature = target_correlations.idxmin()
print(f"  {feature_labels[min_feature]}: {min_corr:.4f}")

print(f"\nWeakest correlation with d18O:")
abs_min_corr = target_correlations.abs().min()
abs_min_feature = target_correlations.abs().idxmin()
print(f"  {feature_labels[abs_min_feature]}: {target_correlations[abs_min_feature]:.4f}")

# Check for multicollinearity (high correlation between features)
print("\n" + "="*80)
print("MULTICOLLINEARITY CHECK")
print("="*80)
print("\nHigh correlations between features (|r| > 0.7):")
high_corr_pairs = []
for i in range(len(feature_col)):
    for j in range(i+1, len(feature_col)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((feature_col[i], feature_col[j], corr_val))

if high_corr_pairs:
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feature_labels[feat1]} <-> {feature_labels[feat2]}: {corr_val:.4f}")
else:
    print("  No high correlations found (all |r| <= 0.7)")

print("\n" + "="*80)
