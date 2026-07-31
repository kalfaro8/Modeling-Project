# Multi-Seed Random Forest Comparison

## Overview
This tool runs the Random Forest Improved model with multiple different random seeds and provides comprehensive comparison analysis across all seeds, including average performance metrics.

## Files Created
- **`multi_seed_comparison.py`** - Main script to run multi-seed experiments
- **`multi_seed_comparison.png`** - Comprehensive visualization comparing all seeds
- **`multi_seed_results.json`** - Detailed results in JSON format

## Features

### 1. **Multiple Seed Testing**
- Runs the same Random Forest model with different random seeds
- Tests how sensitive the model is to the train/test split
- Provides statistical analysis across all seeds

### 2. **Comprehensive Metrics**
For each seed, the script calculates:
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **Cross-Validation R²** - 5-fold CV performance

### 3. **Statistical Summary**
- Mean ± Standard Deviation for all metrics
- Best and worst performing seeds
- Performance range analysis
- Individual seed comparison table

### 4. **Visualizations**
The script generates a comprehensive plot with:
1. R² Score comparison across seeds
2. RMSE comparison across seeds
3. MAE comparison across seeds
4. Box plot of metric distributions
5. Predicted vs Observed overlay for all seeds
6. Baseline vs Optimized comparison
7. Feature importance heatmap across seeds

## Usage

### Basic Usage
```bash
python multi_seed_comparison.py
```

### Customizing Seeds
Edit the `main()` function in `multi_seed_comparison.py`:

```python
# Default seeds
seeds = [42, 50, 100, 123, 200, 250, 300, 350, 400, 500]

# Or use your own
seeds = [1, 2, 3, 4, 5]  # Quick test with 5 seeds
seeds = list(range(0, 100, 10))  # Test 10 seeds from 0 to 90
```

### Hyperparameter Tuning
By default, the script uses pre-optimized parameters for speed. To enable full hyperparameter tuning:

```python
# In main() function, change:
experiment.run_all_seeds(tune_hyperparameters=False)  # Fast (default)

# To:
experiment.run_all_seeds(tune_hyperparameters=True)   # Slow but optimized per seed
```

**Note:** Hyperparameter tuning can take several minutes per seed!

## Output Interpretation

### Console Output
```
======================================================================
MULTI-SEED EXPERIMENT SUMMARY
======================================================================
Number of seeds tested: 10
Seeds: [42, 50, 100, 123, 200, 250, 300, 350, 400, 500]

======================================================================
AVERAGE PERFORMANCE ACROSS ALL SEEDS
======================================================================

Baseline Model:
  R² Score:  0.8234 ± 0.0156
  RMSE:      0.4521 ± 0.0234 ‰
  MAE:       0.3456 ± 0.0189 ‰

Optimized Model:
  R² Score:  0.8567 ± 0.0123
  RMSE:      0.4123 ± 0.0198 ‰
  MAE:       0.3123 ± 0.0156 ‰
```

### Understanding the Results

1. **Mean ± Std Dev**: Shows average performance and variability
   - Lower std dev = more stable model
   - Higher std dev = model sensitive to train/test split

2. **Best/Worst Seeds**: Identifies which random seeds give best/worst results
   - Useful for understanding model stability
   - Can use best seed for final model

3. **Individual Results Table**: Compare each seed side-by-side

### JSON Output Structure
```json
{
  "experiment_info": {
    "date": "2026-07-13 09:00:00",
    "seeds": [42, 50, 100, ...],
    "num_seeds": 10
  },
  "summary_statistics": {
    "optimized_r2": {
      "mean": 0.8567,
      "std": 0.0123,
      "min": 0.8345,
      "max": 0.8789
    },
    ...
  },
  "individual_results": [...],
  "best_parameters_per_seed": {...}
}
```

## Requirements
```bash
pip install pandas scikit-learn openpyxl matplotlib seaborn numpy scipy
```

## Advanced Usage

### Using as a Module
```python
from multi_seed_comparison import MultiSeedRandomForest

# Create experiment
experiment = MultiSeedRandomForest(seeds=[1, 2, 3, 4, 5])

# Run experiment
experiment.run_all_seeds(tune_hyperparameters=False)

# Get statistics
stats = experiment.calculate_statistics()
print(f"Average R²: {stats['optimized_r2']['mean']:.4f}")

# Access individual models
best_seed = 42
model = experiment.models[best_seed]

# Make predictions with best model
predictions = model.predict(X_new)
```

### Custom Data Path
```python
experiment = MultiSeedRandomForest(
    seeds=[42, 50, 100],
    data_path="path/to/your/data.xlsx"
)
```

## Interpreting Visualizations

### 1. R²/RMSE/MAE Bar Charts
- Each bar represents one seed
- Orange dashed line shows the mean
- Look for consistency across seeds

### 2. Box Plot
- Shows distribution of metrics
- Outliers indicate problematic seeds
- Tight boxes = stable model

### 3. Predicted vs Observed Overlay
- All seeds plotted together
- Should cluster around diagonal line
- Spread indicates variability

### 4. Baseline vs Optimized
- Shows improvement from optimization
- Consistent improvement = good optimization

### 5. Feature Importance Heatmap
- Shows which features are consistently important
- Darker colors = higher importance
- Consistency across seeds = reliable features

## Tips for Best Results

1. **Start with 5-10 seeds** to get a sense of variability
2. **Use tune_hyperparameters=False** for quick experiments
3. **Check the standard deviation** - if high, consider:
   - Collecting more data
   - Using stratified splitting
   - Ensemble methods
4. **Look at feature importance consistency** - features that are important across all seeds are most reliable
5. **Use the best performing seed** for your final model

## Troubleshooting

### "ModuleNotFoundError"
Install required packages:
```bash
pip install pandas scikit-learn openpyxl matplotlib seaborn
```

### "FileNotFoundError: Random Forest Data.xlsx"
Ensure the data file is in the same directory as the script, or provide the full path:
```python
experiment = MultiSeedRandomForest(
    seeds=[42, 50],
    data_path="C:/full/path/to/Random Forest Data.xlsx"
)
```

### Script runs slowly
- Set `tune_hyperparameters=False` (default)
- Reduce number of seeds
- Use fewer cross-validation folds

### Memory issues
- Reduce number of seeds
- Reduce n_estimators in the model
- Close other applications

## Example Workflow

```python
# 1. Quick test with 3 seeds
experiment = MultiSeedRandomForest(seeds=[42, 50, 100])
experiment.run_all_seeds(tune_hyperparameters=False)
experiment.print_summary()

# 2. If results look good, run full experiment
experiment_full = MultiSeedRandomForest(seeds=list(range(0, 100, 10)))
experiment_full.run_all_seeds(tune_hyperparameters=False)
experiment_full.plot_comparison()
experiment_full.save_results()

# 3. Use best seed for final model
stats = experiment_full.calculate_statistics()
best_seed_idx = np.argmax(stats['optimized_r2']['values'])
best_seed = experiment_full.seeds[best_seed_idx]
best_model = experiment_full.models[best_seed]

print(f"Best seed: {best_seed}")
print(f"Best R²: {stats['optimized_r2']['max']:.4f}")
```

## Citation
If you use this tool in your research, please cite:
```
Multi-Seed Random Forest Comparison Tool
Nuclear Reactor Modeling Project
Pacific Northwest National Laboratory (PNNL)
2026
```

## Support
For questions or issues, please contact the project team or refer to the main project documentation.
