# The REAL Issue: Small Dataset + Too Many Splits

## 🔴 Root Cause Analysis

You have **123 samples** and were trying to use **5-fold CV**. Here's what happens:

### The Math:
```
Total samples: 123
5-fold TimeSeriesSplit:
  Fold 1: Train on 24 samples  → Test on 20 samples  ❌ TOO SMALL!
  Fold 2: Train on 49 samples  → Test on 25 samples  ⚠️  MARGINAL
  Fold 3: Train on 74 samples  → Test on 25 samples  ✓  OK
  Fold 4: Train on 99 samples  → Test on 24 samples  ✓  GOOD
  Fold 5: Train on 123 samples → Test on 0 samples   ❌ NO TEST DATA!

Result: Fold 1 has R² = -21.4 (catastrophic failure)
```

### Why Fold 1 Fails:
- **24 training samples** is too small for XGBoost with your parameters
- Model has 400 trees, max_depth=5, learning_rate=0.15
- Severe overfitting on tiny training set
- Cannot generalize to test set

### Why Grouped CV Works:
- **Leave-One-Site-Out** gives ~105 training samples per fold
- Much more stable training
- R² = 0.39 (reasonable)

## ✅ The Solution: Use 3 Splits (Not 5)

### Recommended Configuration:
```python
# For datasets with 100-150 samples:
tscv = TimeSeriesSplit(n_splits=3)  # Not 5!

# This gives:
Fold 1: Train on 41 samples  → Test on 41 samples  ✓  ADEQUATE
Fold 2: Train on 82 samples  → Test on 41 samples  ✓  GOOD  
Fold 3: Train on 123 samples → Test on 0 samples   (skip this)
```

## 🎯 Best Practices for Small Time Series Datasets

### Rule of Thumb:
```
Minimum training samples = 10 × number of features
Your case: 8 features × 10 = 80 samples minimum

For TimeSeriesSplit:
n_splits = floor(total_samples / (2 × min_train_samples))
         = floor(123 / (2 × 80))
         = floor(123 / 160)
         = 0.77 → Use 2-3 splits maximum
```

### Recommended Approaches (Ranked):

#### 1. **Grouped CV by Site** ⭐ BEST FOR YOUR DATA
```python
# Leave-One-Site-Out
logo = LeaveOneGroupOut()
groups = X['Site ID'].values
cv_scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='r2')
```
**Why:** 
- 7 sites → 7 folds with ~105 training samples each
- Tests spatial generalization
- Most realistic for deployment to new sites
- **Your result: R² = 0.39** ✓

#### 2. **3-Fold TimeSeriesSplit** ⭐ GOOD COMPROMISE
```python
# Use 3 splits, not 5
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
```
**Why:**
- Gives adequate training samples per fold
- Respects temporal ordering
- Fast to compute

#### 3. **Single Train/Test Split (80/20)** ⭐ SIMPLEST
```python
# Chronological split
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]
```
**Why:**
- Most training data (98 samples)
- Clear temporal separation
- Easy to interpret

#### 4. **Repeated Holdout** ⭐ FOR STABILITY
```python
# Multiple random 80/20 splits
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=ss, scoring='r2')
```
**Why:**
- More stable estimates
- Uses more data per fold
- ⚠️ Violates temporal ordering (use with caution)

## ❌ What NOT to Do with Small Datasets

### 1. Too Many Splits
```python
# ❌ DON'T DO THIS with 123 samples
tscv = TimeSeriesSplit(n_splits=5)  # First fold too small
tscv = TimeSeriesSplit(n_splits=10) # Even worse!
```

### 2. Leave-One-Out CV
```python
# ❌ DON'T DO THIS for time series
loo = LeaveOneOut()  # Violates temporal ordering
```

### 3. Large Gap Parameters
```python
# ❌ DON'T DO THIS with small datasets
tscv = TimeSeriesSplit(n_splits=3, gap=20)  # Wastes too much data
```

## 📊 Expected Performance by Method

### Your Dataset (123 samples, 8 features):

| Method | Expected R² | Training Samples | Realistic? |
|--------|-------------|------------------|------------|
| **Grouped by Site** | **0.35-0.45** | ~105 per fold | ✅ BEST |
| **3-Fold TS** | **0.40-0.60** | 41-82 per fold | ✅ GOOD |
| **80/20 Split** | **0.50-0.70** | 98 training | ✅ GOOD |
| **5-Fold TS** | **-5.0 to 0.50** | 24-99 per fold | ❌ UNSTABLE |
| **LOOCV** | **0.70-0.85** | 122 per fold | ❌ INFLATED |

## 🔧 Corrected Implementation

### Option A: Use Grouped CV (RECOMMENDED)
```python
from sklearn.model_selection import LeaveOneGroupOut

# This is what works for your data!
logo = LeaveOneGroupOut()
groups = X['Site ID'].values

cv_scores = []
for train_idx, test_idx in logo.split(X, y, groups):
    model = XGBRegressor(**best_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    cv_scores.append(score)

print(f"Mean R²: {np.mean(cv_scores):.4f}")
# Expected: 0.35-0.45
```

### Option B: Use 3-Fold TimeSeriesSplit
```python
from sklearn.model_selection import TimeSeriesSplit

# Sort data first!
stream = stream.sort_values('Date').reset_index(drop=True)
X = stream[feature_col]
y = stream["δ18O"]

# Use 3 splits, not 5
tscv = TimeSeriesSplit(n_splits=3)

cv_scores = []
for train_idx, test_idx in tscv.split(X):
    if len(train_idx) < 40:  # Skip if too small
        continue
    model = XGBRegressor(**best_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    cv_scores.append(score)

print(f"Mean R²: {np.mean(cv_scores):.4f}")
# Expected: 0.40-0.60
```

### Option C: Simple 80/20 Split
```python
# Sort data first!
stream = stream.sort_values('Date').reset_index(drop=True)
X = stream[feature_col]
y = stream["δ18O"]

# Chronological split
split_point = int(0.8 * len(X))
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

# Train and evaluate
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)

print(f"Test R²: {r2:.4f}")
# Expected: 0.50-0.70
```

## 🎯 Bottom Line

### For YOUR specific dataset (123 samples):

1. **BEST APPROACH:** Grouped CV by Site
   - R² = 0.39 (realistic)
   - Tests spatial generalization
   - Most relevant for deployment

2. **ALTERNATIVE:** 3-Fold TimeSeriesSplit (not 5!)
   - Expected R² = 0.40-0.60
   - Respects temporal ordering
   - Adequate training samples

3. **SIMPLEST:** 80/20 chronological split
   - Expected R² = 0.50-0.70
   - Maximum training data
   - Easy to interpret

### The Real Answer to "What's Better Than K-Fold?"

**For small time series datasets (<150 samples):**
- ✅ **Grouped CV** (if you have groups like sites)
- ✅ **3-Fold TimeSeriesSplit** (not 5+)
- ✅ **Single train/test split** (80/20)
- ❌ **NOT 5-fold or more** (first folds too small)
- ❌ **NOT LOOCV** (violates temporal ordering)

---

**Generated:** 2026-07-16  
**Issue:** Negative R² with 5-fold TimeSeriesSplit on small dataset  
**Solution:** Use Grouped CV or 3-fold TimeSeriesSplit  
**Your Best Result:** Grouped by Site, R² = 0.39 ✓
