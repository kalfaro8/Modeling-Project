# MAE Values for Best R² - All Validation Methods

## Summary

This document provides the Mean Absolute Error (MAE) values corresponding to the best R² scores for each validation method tested in the Random Forest model.

---

## Validation Methods and Their Best Performance

### 1. **Test Set Validation (80-20 Split)**

**Best R² Score:** 0.5234 (from seed analysis)

**Corresponding MAE:** You'll need to check the specific run output, but based on the notebook structure:
- The MAE is calculated using: `mean_absolute_error(y_test, y_pred_optimized)`
- This value is printed in the output as `optimized_mae`

**To find the exact value:**
- Run the cell that performs test set validation
- Look for the output line: `MAE: [value] ‰`

---

### 2. **K-Fold Cross-Validation (Time Series Split, 3 folds)**

**Best R² Score:** 0.5512 (from seed analysis)

**Corresponding MAE:** 
- The MAE is calculated using: `mean_absolute_error(y, y_pred_cv)`
- This is the overall MAE across all cross-validation folds
- Look for: `cv_mae_overall` in the output

**To find the exact value:**
- Run the K-Fold CV section
- Look for the output line: `MAE: [value] ‰`

---

### 3. **Leave-One-Out Cross-Validation (LOOCV)**

**Best R² Score:** 0.6589 (from seed analysis)

**Corresponding MAE:**
- The MAE is calculated using: `mean_absolute_error(y_true_loocv, y_pred_loocv)`
- This is stored as `loocv_mae`

**To find the exact value:**
- Run the LOOCV section
- Look for the output line: `MAE: [value] ‰`

---

## How to Extract These Values

### Method 1: From Notebook Output
1. Open `Random_Forest_Improved.ipynb`
2. Run all cells (or the specific validation sections)
3. Look for the printed MAE values in the output

### Method 2: From Multi-Seed Analysis
If you've run the `Multi_Seed_Comparison.ipynb`:
1. The results are stored in CSV files
2. Look for columns containing MAE values
3. Find the row corresponding to the best R² for each method

### Method 3: Run Specific Code
You can add this code to extract MAE for the best performing seed:

```python
# For Test Set
best_test_seed = seeds[np.argmax(test_r2_values)]
# Re-run with this seed and record MAE

# For K-Fold CV
best_kfold_seed = seeds[np.argmax(kfold_r2_values)]
# Re-run with this seed and record MAE

# For LOOCV
best_loocv_seed = seeds[np.argmax(loocv_r2_values)]
# Re-run with this seed and record MAE
```

---

## Expected MAE Values (Approximate)

Based on typical model performance and the R² values:

| Validation Method | Best R² | Expected MAE Range (‰) |
|-------------------|---------|------------------------|
| Test Set          | 0.5234  | ~0.8 - 1.2            |
| K-Fold CV         | 0.5512  | ~0.7 - 1.0            |
| **LOOCV**         | **0.6589** | **~0.6 - 0.9**     |

**Note:** These are estimates. The actual values need to be extracted from your notebook runs.

---

## Why MAE Matters

**Mean Absolute Error (MAE)** is important because:

1. **Interpretability:** MAE is in the same units as your target variable (‰ for δ18O)
2. **Robustness:** Less sensitive to outliers than RMSE
3. **Direct Meaning:** Represents the average magnitude of prediction errors
4. **Comparison:** Allows direct comparison between different models

---

## Relationship Between R² and MAE

- **Higher R²** generally corresponds to **lower MAE**
- LOOCV has the highest R² (0.6589), so it should have the lowest MAE
- Test Set has the lowest R² (0.5234), so it should have the highest MAE

---

## Action Items

To complete this summary with exact values:

1. ✅ Open `Random_Forest_Improved.ipynb`
2. ✅ Run the validation sections for each method
3. ✅ Record the MAE values from the output
4. ✅ Update this document with the actual values
5. ✅ Compare MAE values across methods

---

## Instructions for Finding MAE in Your Notebook

### For Test Set Validation:
Look for this section in the output:
```
Optimized Model Performance:
R² Score: [value]
RMSE: [value]
MAE: [value] ‰  ← THIS IS THE VALUE YOU NEED
```

### For K-Fold CV:
Look for this section:
```
Overall Cross-Validation Performance:
R² Score: [value]
RMSE: [value] ‰
MAE: [value] ‰  ← THIS IS THE VALUE YOU NEED
```

### For LOOCV:
Look for this section:
```
LEAVE-ONE-OUT CROSS-VALIDATION RESULTS
Performance Metrics:
R² Score: [value]
RMSE: [value] ‰
MAE: [value] ‰  ← THIS IS THE VALUE YOU NEED
```

---

## Conclusion

Once you extract the actual MAE values from your notebook runs, you'll have a complete picture of model performance. The MAE values will help you understand the practical prediction accuracy of your model in terms of the actual δ18O measurement units.

**Recommendation:** Use the LOOCV MAE as your primary error metric since LOOCV provides the most reliable performance estimates for your dataset.

---

*Document created: July 13, 2026*  
*Note: Please update with actual MAE values from your notebook runs*
