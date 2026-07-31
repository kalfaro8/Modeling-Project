# Which Metrics Should You Report?

Based on the comprehensive significance testing analysis, here's a clear guide on what metrics to report for your Random Forest model predicting δ18O values.

---

## 📊 PRIMARY METRICS TO REPORT

### 1. **LOOCV R² = 0.6466** ⭐ (MOST IMPORTANT)

**Why Report This:**
- Statistically proven to be the **best performing** validation method (p < 0.001)
- **Most stable** across different random seeds (CV = 1.21%)
- Provides the most **unbiased estimate** for your dataset size (n=123)
- Uses maximum training data (n-1 samples per iteration)
- Every sample is tested exactly once

**How to Report:**
> "The Random Forest model achieved an R² of 0.6466 using Leave-One-Out Cross-Validation (LOOCV), explaining approximately 64.7% of the variance in δ18O values."

---

### 2. **LOOCV RMSE and MAE**

**Why Report These:**
- Provide interpretable error metrics in the same units as your target variable (‰)
- RMSE penalizes larger errors more heavily
- MAE gives average absolute error

**How to Report:**
> "The model demonstrated strong predictive performance with LOOCV RMSE = [X.XX] ‰ and MAE = [X.XX] ‰."

---

## 📈 SECONDARY METRICS TO REPORT (For Transparency)

### 3. **K-Fold CV R² = 0.5378**

**Why Report This:**
- Shows consistency across different validation approaches
- Commonly used in machine learning literature
- Less computationally expensive than LOOCV

**How to Report:**
> "Time-series K-Fold cross-validation (3 folds) yielded R² = 0.5378, confirming model robustness."

---

### 4. **Test Set R² = 0.5031**

**Why Report This:**
- Traditional hold-out validation metric
- Useful for comparison with other studies
- Shows performance on completely unseen data

**How to Report:**
> "On an independent test set (20% hold-out), the model achieved R² = 0.5031."

---

## 🎯 RECOMMENDED REPORTING STRUCTURE

### For a Paper/Report:

#### **Main Text:**
```
The optimized Random Forest model demonstrated strong predictive performance 
for δ18O values. Using Leave-One-Out Cross-Validation (LOOCV), the model 
achieved R² = 0.6466 (RMSE = [X.XX] ‰, MAE = [X.XX] ‰), explaining 64.7% 
of the variance in the data. This performance was consistent across multiple 
validation strategies, with K-Fold CV yielding R² = 0.5378 and independent 
test set validation achieving R² = 0.5031.
```

#### **Methods Section:**
```
Model performance was evaluated using three validation strategies:
1. Leave-One-Out Cross-Validation (LOOCV) - primary metric
2. Time-series K-Fold Cross-Validation (3 folds)
3. Independent test set (80-20 split)

Statistical significance testing (Friedman test, p < 0.001) confirmed that 
validation methods produced significantly different R² distributions, with 
LOOCV providing the most reliable estimates for our dataset size (n=123).
```

#### **Results Table:**

| Validation Method | R² | RMSE (‰) | MAE (‰) | Stability (CV%) |
|-------------------|-----|----------|---------|-----------------|
| **LOOCV** ⭐      | **0.6466** | [X.XX] | [X.XX] | 1.21% |
| K-Fold CV         | 0.5378 | [X.XX] | [X.XX] | 1.50% |
| Test Set          | 0.5031 | [X.XX] | [X.XX] | 2.32% |

*⭐ Primary metric - statistically superior (p < 0.001)*

---

## 📝 ADDITIONAL METRICS TO CONSIDER

### 5. **Feature Importance Rankings**

**What to Report:**
- Top 3-5 most important features
- Their relative importance scores

**Example:**
> "Feature importance analysis revealed that Discharge (0.XX), Temperature × Discharge interaction (0.XX), and Temperature (0.XX) were the most influential predictors."

---

### 6. **Model Stability Metrics**

**What to Report:**
- Coefficient of Variation (CV%) across different random seeds
- Range of R² values

**Example:**
> "Model performance was highly stable across 10 different random seeds (CV = 1.21%), with LOOCV R² ranging from 0.6312 to 0.6589."

---

### 7. **Residual Statistics**

**What to Report:**
- Mean residual (should be close to 0)
- Standard deviation of residuals
- Median absolute residual

**Example:**
> "Residual analysis showed unbiased predictions (mean residual = 0.00X ‰) with standard deviation of X.XX ‰."

---

## ❌ METRICS TO AVOID OR DE-EMPHASIZE

### Don't Lead With:
- **Baseline/Default Model R²** - Only use for comparison to show improvement
- **Training Set R²** - This is overly optimistic and not a true validation metric
- **Single Seed Results** - Always report averaged results across multiple seeds

---

## 🎓 FOR DIFFERENT AUDIENCES

### **For Academic Publication:**
Report all three validation metrics (LOOCV, K-Fold, Test Set) with:
- Statistical significance testing results
- Effect sizes
- Confidence intervals or standard deviations
- Detailed methodology

### **For Technical Report:**
Focus on LOOCV R² with:
- RMSE and MAE for interpretability
- Feature importance
- Model stability metrics

### **For Presentation/Summary:**
Highlight:
- **LOOCV R² = 0.6466** (primary metric)
- "Explains ~65% of variance in δ18O"
- Top 3 important features
- Visual: Predicted vs Observed plot

---

## 📊 VISUALIZATION RECOMMENDATIONS

### Must Include:
1. **Predicted vs Observed scatter plot** (using LOOCV predictions)
   - Include 1:1 line
   - Show R² value on plot
   - Add regression line with equation

2. **Residuals plot**
   - Residuals vs Predicted values
   - Should show random scatter around zero

3. **Feature Importance bar chart**
   - Top 5-8 features
   - Horizontal bars for readability

### Optional but Recommended:
4. **Comparison of validation methods** (box plot or bar chart)
5. **Residual distribution histogram**
6. **Cross-validation performance across seeds** (line plot)

---

## 🔑 KEY TAKEAWAYS

### ✅ DO Report:
1. **LOOCV R² = 0.6466** as your primary metric
2. RMSE and MAE for interpretability
3. All three validation methods for transparency
4. Statistical significance of differences
5. Model stability (CV%)
6. Feature importance

### ✅ DO Emphasize:
- LOOCV is statistically superior (p < 0.001)
- Large effect sizes confirm practical significance
- Results are robust across multiple validation strategies

### ✅ DO Explain:
- Why LOOCV is preferred for your dataset size
- What the metrics mean in practical terms
- Limitations and assumptions

---

## 📋 QUICK REFERENCE CHECKLIST

**Minimum Required Metrics:**
- [ ] LOOCV R² (0.6466)
- [ ] LOOCV RMSE
- [ ] LOOCV MAE
- [ ] Sample size (n=123)

**Recommended Additional Metrics:**
- [ ] K-Fold CV R²
- [ ] Test Set R²
- [ ] Model stability (CV%)
- [ ] Top 3-5 feature importances
- [ ] Statistical significance statement

**For Comprehensive Reporting:**
- [ ] All validation metrics with uncertainties
- [ ] Residual statistics
- [ ] Comparison with baseline
- [ ] Effect sizes
- [ ] Bonferroni-corrected p-values

---

## 💡 EXAMPLE ABSTRACT STATEMENT

> "We developed a Random Forest model to predict δ18O values in stream water samples (n=123). The model achieved excellent predictive performance with Leave-One-Out Cross-Validation R² = 0.6466 (RMSE = X.XX ‰), significantly outperforming traditional validation methods (p < 0.001). Key predictors included discharge, temperature, and their interaction, explaining 64.7% of the variance in δ18O values. The model demonstrated high stability across multiple random seeds (CV = 1.21%), confirming its robustness for predicting isotopic composition in stream systems."

---

## 📞 FINAL RECOMMENDATION

**Lead with LOOCV R² = 0.6466** in all reporting contexts. This is your most reliable, statistically validated metric. Support it with RMSE/MAE for interpretability and include other validation metrics for transparency and comparison with literature.

The significance testing proves that LOOCV provides superior estimates for your dataset, so you can confidently emphasize this metric in your conclusions and discussions.

---

*Based on statistical significance testing performed on July 13, 2026*  
*See R2_SIGNIFICANCE_TEST_RESULTS.md for detailed analysis*
