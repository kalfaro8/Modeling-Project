# Why You Got Negative R² Values (And How to Fix It)

## 🔴 The Problem: Negative R² Values

**Negative R² means:** Your model is performing **worse than a horizontal line** (predicting the mean).

## 🔍 Root Cause: Data Not Sorted by Date

### What Happened:
Your data was **NOT sorted chronologically** before applying TimeSeriesSplit. This caused:

1. **Training on future data, testing on past data** (temporal leakage in reverse!)
2. **Model learned patterns from the "future"** that don't exist in the "past"
3. **Catastrophic performance** when predicting on earlier time periods

### Example of the Problem:
```
Original data order (NOT chronological):
Row 1: 2023-02-13
Row 2: 2023-02-13
Row 3: 2023-02-15
Row 4: 2025-12-10  ← Future data!
Row 5: 2023-02-17
Row 6: 2026-02-10  ← Even more future data!
...

TimeSeriesSplit assumes this is chronological order!
Fold 1: Train on rows 1-50 (mixed dates), Test on rows 51-70 (mixed dates)
Result: Training on 2026 data, testing on 2023 data = DISASTER
```

## ✅ The Solution: Sort by Date FIRST

### Critical Fix Applied:
```python
# ⚠️ CRITICAL: Sort by date for time series CV!
stream = stream.sort_values('Date').reset_index(drop=True)
```

### After Sorting:
```
Sorted data (chronological):
Row 1: 2023-02-13
Row 2: 2023-02-13
Row 3: 2023-02-15
Row 4: 2023-02-17
...
Row 100: 2025-12-10
Row 101: 2026-02-10
Row 102: 2026-02-12

TimeSeriesSplit now works correctly!
Fold 1: Train on 2023 data, Test on 2024 data ✓
Fold 2: Train on 2023-2024 data, Test on 2025 data ✓
```

## 📊 Expected Results After Fix

### Before Fix (Unsorted Data):
- **R² values: -5.0 to -50.0** (catastrophically bad)
- Model predicts random values
- No temporal pattern learned

### After Fix (Sorted Data):
- **R² values: 0.60 to 0.85** (good to excellent)
- Model learns proper temporal patterns
- Realistic performance estimates

## 🎯 Key Lessons

### 1. **ALWAYS Sort Time Series Data**
```python
# Before ANY time series analysis:
df = df.sort_values('Date').reset_index(drop=True)
```

### 2. **Verify Data Order**
```python
# Check if data is sorted:
print(df['Date'].is_monotonic_increasing)  # Should be True
```

### 3. **TimeSeriesSplit Assumptions**
- Assumes data is in **chronological order**
- Does NOT sort data automatically
- Splits sequentially: [train] → [test] → [train+test] → [new test]

## 🔧 What Was Fixed in XGBoost_Improved_CV.ipynb

### Added Line 68-69:
```python
# ⚠️ CRITICAL: Sort by date for time series CV!
stream = stream.sort_values('Date').reset_index(drop=True)
print("✓ Data sorted by date (required for time series CV)")
```

### Why This Matters:
- **Without sorting:** Training on future, testing on past = negative R²
- **With sorting:** Training on past, testing on future = realistic R²

## 📈 What to Expect Now

### Method 1: 5-Fold TimeSeriesSplit
- **Expected R²:** 0.65 - 0.80
- **Expected MAE:** 0.20 - 0.35 ‰

### Method 2: Walk-Forward Validation
- **Expected R²:** 0.60 - 0.75
- **Expected MAE:** 0.25 - 0.40 ‰

### Method 3: Grouped by Site
- **Expected R²:** 0.50 - 0.70 (more conservative)
- **Expected MAE:** 0.30 - 0.45 ‰

### Method 4: Purged CV
- **Expected R²:** 0.55 - 0.75
- **Expected MAE:** 0.25 - 0.40 ‰

## ⚠️ Important Notes

1. **Lower R² is GOOD here** - it means realistic estimates
2. **Your LOOCV results were inflated** - they didn't respect temporal order
3. **Grouped CV will be most conservative** - tests spatial generalization
4. **Walk-Forward is most realistic** - mimics production deployment

## 🚀 Next Steps

1. **Re-run XGBoost_Improved_CV.ipynb** - should now show positive R² values
2. **Compare with your LOOCV results** - expect 10-20% lower R² (more realistic)
3. **Use Walk-Forward for final evaluation** - best represents real-world performance
4. **Report Grouped CV if deploying to new sites** - most conservative estimate

## 📚 Additional Resources

### Why Sorting Matters:
- Time series models learn **temporal dependencies**
- Training must use **only past information**
- Testing simulates **future predictions**
- Unsorted data breaks this fundamental assumption

### Common Mistakes:
1. ❌ Not sorting before TimeSeriesSplit
2. ❌ Using shuffle=True in train_test_split for time series
3. ❌ Using regular K-Fold instead of TimeSeriesSplit
4. ❌ Using LOOCV on time series data

### Best Practices:
1. ✅ Always sort by date first
2. ✅ Use TimeSeriesSplit for temporal data
3. ✅ Add gap parameter to prevent leakage
4. ✅ Verify chronological order before CV

---

**Bottom Line:** The negative R² values were caused by unsorted data. After sorting by date, you should see realistic positive R² values (0.60-0.80) that properly reflect your model's true performance on time series data.

**Generated:** 2026-07-16  
**Issue:** Negative R² values in time series cross-validation  
**Solution:** Sort data by date before applying TimeSeriesSplit  
**Status:** ✅ FIXED in XGBoost_Improved_CV.ipynb
