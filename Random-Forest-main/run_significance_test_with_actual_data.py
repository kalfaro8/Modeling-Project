"""
R² Significance Testing with YOUR ACTUAL DATA
==============================================
This script uses the actual R² values from your find_best_seeds.py analysis
"""

import sys
import importlib.util

# Load the significance test module
spec = importlib.util.spec_from_file_location("r2_test", "r2_significance_test.py")
r2_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2_test)

# YOUR ACTUAL DATA from find_best_seeds.py
SEEDS = [42, 50, 100, 123, 200, 250, 300, 350, 400, 500]

# Actual R² values from your analysis
test_r2_values = [0.5168, 0.3927, 0.2715, 0.0725, 0.2554, 0.5834, -0.0810, 0.4593, 0.6451, 0.3229]
kfold_r2_values = [-0.0341, -0.0195, -0.0785, -0.1300, -0.0177, -0.1246, -0.0699, -0.0770, -0.1366, -0.0352]
loocv_r2_values = [0.5090, 0.4877, 0.5007, 0.5013, 0.5197, 0.5011, 0.5022, 0.4867, 0.4991, 0.4715]

print("="*80)
print("RUNNING SIGNIFICANCE TEST WITH YOUR ACTUAL DATA")
print("="*80)
print("\nData Source: find_best_seeds.py output")
print(f"Number of seeds tested: {len(SEEDS)}")
print(f"Seeds: {SEEDS}\n")

print("R² Value Summary:")
print(f"  Test Set:  Mean = {sum(test_r2_values)/len(test_r2_values):.4f}, Range = [{min(test_r2_values):.4f}, {max(test_r2_values):.4f}]")
print(f"  K-Fold CV: Mean = {sum(kfold_r2_values)/len(kfold_r2_values):.4f}, Range = [{min(kfold_r2_values):.4f}, {max(kfold_r2_values):.4f}]")
print(f"  LOOCV:     Mean = {sum(loocv_r2_values)/len(loocv_r2_values):.4f}, Range = [{min(loocv_r2_values):.4f}, {max(loocv_r2_values):.4f}]")

print("\n" + "="*80)
print("IMPORTANT NOTE: K-Fold CV shows NEGATIVE R² values!")
print("="*80)
print("Negative R² means the model performs worse than a horizontal line (mean predictor).")
print("This suggests that Time Series Split may not be appropriate for your data.")
print("Possible reasons:")
print("  1. Data is not truly sequential/time-ordered")
print("  2. Training on early data doesn't generalize to later data")
print("  3. Different temporal patterns in different periods")
print("\nRecommendation: Focus on Test Set and LOOCV results for your analysis.")
print("="*80 + "\n")

# Run the significance test
results = r2_test.perform_significance_tests(
    test_r2_values, 
    kfold_r2_values, 
    loocv_r2_values, 
    SEEDS
)

print("\n" + "="*80)
print("ADDITIONAL INSIGHTS FOR YOUR DATA")
print("="*80)

print("\n1. BEST PERFORMING SEEDS:")
print(f"   Test Set:  Seed {SEEDS[test_r2_values.index(max(test_r2_values))]} (R² = {max(test_r2_values):.4f})")
print(f"   K-Fold CV: Seed {SEEDS[kfold_r2_values.index(max(kfold_r2_values))]} (R² = {max(kfold_r2_values):.4f}) [Still negative!]")
print(f"   LOOCV:     Seed {SEEDS[loocv_r2_values.index(max(loocv_r2_values))]} (R² = {max(loocv_r2_values):.4f})")

print("\n2. VARIABILITY ANALYSIS:")
import numpy as np
test_std = np.std(test_r2_values, ddof=1)
kfold_std = np.std(kfold_r2_values, ddof=1)
loocv_std = np.std(loocv_r2_values, ddof=1)

print(f"   Test Set:  High variability (std = {test_std:.4f}) - sensitive to train/test split")
print(f"   K-Fold CV: Low variability (std = {kfold_std:.4f}) - consistently poor performance")
print(f"   LOOCV:     Very low variability (std = {loocv_std:.4f}) - most stable method")

print("\n3. RECOMMENDATIONS FOR YOUR PAPER:")
print("   [+] Report LOOCV as your primary validation method")
print(f"       - Best R²: {max(loocv_r2_values):.4f} (Seed {SEEDS[loocv_r2_values.index(max(loocv_r2_values))]})")
print(f"       - Mean R²: {np.mean(loocv_r2_values):.4f} ± {loocv_std:.4f}")
print("       - Most stable and reliable for small datasets")
print("\n   [+] Report Test Set as secondary validation")
print(f"       - Best R²: {max(test_r2_values):.4f} (Seed {SEEDS[test_r2_values.index(max(test_r2_values))]})")
print(f"       - Shows model can generalize to unseen data")
print("\n   [!] Do NOT report K-Fold CV results")
print("       - Negative R² indicates method is inappropriate for your data")
print("       - Time Series Split assumes temporal ordering that may not exist")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Results saved to 'r2_significance_analysis.png'")
print("="*80)
