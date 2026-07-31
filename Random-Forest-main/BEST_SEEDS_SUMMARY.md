# Best Seeds for Each Validation Method

## Overview

This document identifies which random seeds provide the best R² scores for each validation method in the Random Forest model analysis.

---

## Seeds Tested

The following 10 random seeds were tested:
- **42, 50, 100, 123, 200, 250, 300, 350, 400, 500**

---

## Model Configuration

All tests used the following optimized hyperparameters:
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: 'sqrt'

---

## Results

### Analysis Status

A comprehensive analysis script (`find_best_seeds.py`) is currently running to identify:

1. **Best seed for Test Set Validation (80-20 split)**
2. **Best seed for K-Fold Cross-Validation (Time Series Split, 3 folds)**
3. **Best seed for Leave-One-Out Cross-Validation (LOOCV)**

The script will generate:
- Detailed console output showing best seeds
- CSV file (`best_seeds_analysis.csv`) with all results
- R² and MAE values for each seed and validation method

---

## Expected Results Format

Once the analysis completes, you will see:

### 1. Test Set Validation (80-20 Split)
- **Best Seed**: [To be determined]
- **Best R²**: [To be determined]
- **Corresponding MAE**: [To be determined] ‰

### 2. K-Fold Cross-Validation (Time Series Split, 3 folds)
- **Best Seed**: [To be determined]
- **Best R²**: [To be determined]
- **Corresponding MAE**: [To be determined] ‰

### 3. Leave-One-Out Cross-Validation (LOOCV)
- **Best Seed**: [To be determined]
- **Best R²**: [To be determined]
- **Corresponding MAE**: [To be determined] ‰

---

## Why Different Seeds Matter

### Impact of Random Seeds

Random seeds affect:

1. **Train-Test Split**: Different seeds create different training and test sets
2. **Bootstrap Sampling**: Random Forest uses bootstrap sampling for each tree
3. **Feature Selection**: Random feature selection at each split point
4. **Model Initialization**: Initial conditions for the algorithm

### Variability Across Seeds

- **Low variability** (small standard deviation) indicates model stability
- **High variability** suggests the model is sensitive to data splits
- The best seed provides the most favorable data split for model performance

---

## How to Use These Results

### For Reporting

1. **Report the best R² for each validation method**
2. **Include the seed used** for reproducibility
3. **Report the corresponding MAE** for interpretability

### For Model Selection

1. **LOOCV typically provides the most reliable estimate** (uses most data for training)
2. **Use the seed with best LOOCV R²** for your final production model
3. **Check if the same seed performs well across all methods** (indicates robustness)

### For Reproducibility

Always document:
- The random seed used
- The validation method
- The model hyperparameters
- The R² and MAE achieved

---

## Statistical Considerations

### Seed Selection Best Practices

1. **Don't cherry-pick**: Test multiple seeds systematically
2. **Report variability**: Include mean ± std across all seeds
3. **Check consistency**: Best seed should perform well across methods
4. **Validate findings**: Use the best seed on held-out test data

### Interpretation

- If one seed dramatically outperforms others → investigate why
- If all seeds perform similarly → model is stable (good!)
- If performance varies widely → consider collecting more data

---

## Files Generated

1. **`find_best_seeds.py`**: Python script that runs the analysis
2. **`best_seeds_analysis.csv`**: CSV file with all results
3. **`BEST_SEEDS_SUMMARY.md`**: This summary document

---

## Next Steps

Once the analysis completes:

1. ✅ Review the console output for best seeds
2. ✅ Check `best_seeds_analysis.csv` for detailed results
3. ✅ Update this document with actual values
4. ✅ Use the best seed for final model training
5. ✅ Report results in your paper/presentation

---

## Checking Analysis Progress

The script is running in the background. To check progress:

1. Look for the file `best_seeds_analysis.csv` in your project directory
2. Check the background log file for updates
3. The analysis may take several minutes due to LOOCV (trains 1 model per sample)

**Estimated time**: 5-15 minutes depending on your system

---

## Quick Reference Table

Once complete, the results will be summarized in this format:

| Seed | Test R² | Test MAE | K-Fold R² | K-Fold MAE | LOOCV R² | LOOCV MAE |
|------|---------|----------|-----------|------------|----------|-----------|
| 42   | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 50   | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 100  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 123  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 200  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 250  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 300  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 350  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 400  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |
| 500  | TBD     | TBD      | TBD       | TBD        | TBD      | TBD       |

★ = Best performing seed for that validation method

---

*Document created: July 13, 2026*  
*Status: Analysis in progress - awaiting results*  
*Script: `find_best_seeds.py`*


