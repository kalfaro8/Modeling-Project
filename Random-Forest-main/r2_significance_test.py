"""
R² Significance Testing for Validation Methods
================================================
This script performs statistical significance tests on R² values from different
validation methods: Test Set, K-Fold CV, and LOOCV.

Statistical Tests Performed:
1. Friedman Test - Non-parametric test for comparing multiple related samples
2. Wilcoxon Signed-Rank Test - Pairwise comparisons between validation methods
3. Paired t-test - Parametric pairwise comparison (if data is normally distributed)
4. Effect Size (Cohen's d) - Measures the magnitude of differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, shapiro
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

def cohens_d(x, y):
    """Calculate Cohen's d for effect size"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def perform_significance_tests(test_r2, kfold_r2, loocv_r2, seeds):
    """
    Perform comprehensive significance tests on R² values
    
    Parameters:
    -----------
    test_r2 : list - R² values from test set validation
    kfold_r2 : list - R² values from K-Fold cross-validation
    loocv_r2 : list - R² values from LOOCV
    seeds : list - Random seeds used
    """
    
    print("="*80)
    print("R² SIGNIFICANCE TESTING FOR VALIDATION METHODS")
    print("="*80)
    print(f"\nNumber of experiments: {len(seeds)}")
    print(f"Seeds tested: {seeds}")
    
    # Convert to numpy arrays
    test_r2 = np.array(test_r2)
    kfold_r2 = np.array(kfold_r2)
    loocv_r2 = np.array(loocv_r2)
    
    # ========================================================================
    # 1. DESCRIPTIVE STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("1. DESCRIPTIVE STATISTICS")
    print("="*80)
    
    methods = ['Test Set', 'K-Fold CV', 'LOOCV']
    r2_values = [test_r2, kfold_r2, loocv_r2]
    
    stats_df = pd.DataFrame({
        'Method': methods,
        'Mean': [np.mean(r2) for r2 in r2_values],
        'Std Dev': [np.std(r2, ddof=1) for r2 in r2_values],
        'Median': [np.median(r2) for r2 in r2_values],
        'Min': [np.min(r2) for r2 in r2_values],
        'Max': [np.max(r2) for r2 in r2_values],
        'Range': [np.max(r2) - np.min(r2) for r2 in r2_values],
        'CV (%)': [100 * np.std(r2, ddof=1) / np.mean(r2) for r2 in r2_values]
        
    })
    
    print("\n" + stats_df.to_string(index=False))
    
    # ========================================================================
    # 2. NORMALITY TESTS
    # ========================================================================
    print("\n" + "="*80)
    print("2. NORMALITY TESTS (Shapiro-Wilk)")
    print("="*80)
    print("H0: Data is normally distributed")
    print("If p > 0.05, data is likely normally distributed\n")
    
    normality_results = {}
    for method, r2 in zip(methods, r2_values):
        stat, p_value = shapiro(r2)
        normality_results[method] = {'statistic': stat, 'p_value': p_value}
        is_normal = "YES" if p_value > 0.05 else "NO"
        print(f"{method:15} | Statistic: {stat:.4f} | p-value: {p_value:.4f} | Normal: {is_normal}")
    
    # ========================================================================
    # 3. FRIEDMAN TEST (Overall comparison)
    # ========================================================================
    print("\n" + "="*80)
    print("3. FRIEDMAN TEST (Non-parametric ANOVA for repeated measures)")
    print("="*80)
    print("H0: All validation methods have the same distribution of R² values")
    print("H1: At least one validation method differs significantly\n")
    
    friedman_stat, friedman_p = friedmanchisquare(test_r2, kfold_r2, loocv_r2)
    print(f"Friedman Chi-square statistic: {friedman_stat:.4f}")
    print(f"p-value: {friedman_p:.6f}")
    
    if friedman_p < 0.001:
        print(f"\n*** HIGHLY SIGNIFICANT (p < 0.001) ***")
        print("Strong evidence that validation methods produce different R² distributions")
    elif friedman_p < 0.01:
        print(f"\n*** VERY SIGNIFICANT (p < 0.01) ***")
        print("Very strong evidence that validation methods differ")
    elif friedman_p < 0.05:
        print(f"\n*** SIGNIFICANT (p < 0.05) ***")
        print("Significant evidence that validation methods differ")
    else:
        print(f"\n*** NOT SIGNIFICANT (p >= 0.05) ***")
        print("No significant difference between validation methods")
    
    # ========================================================================
    # 4. PAIRWISE COMPARISONS
    # ========================================================================
    print("\n" + "="*80)
    print("4. PAIRWISE COMPARISONS")
    print("="*80)
    
    comparisons = [
        ('Test Set', 'K-Fold CV', test_r2, kfold_r2),
        ('Test Set', 'LOOCV', test_r2, loocv_r2),
        ('K-Fold CV', 'LOOCV', kfold_r2, loocv_r2)
    ]
    
    pairwise_results = []
    
    for method1, method2, r2_1, r2_2 in comparisons:
        print(f"\n{'-'*80}")
        print(f"Comparing: {method1} vs {method2}")
        print(f"{'-'*80}")
        
        # Mean difference
        mean_diff = np.mean(r2_1) - np.mean(r2_2)
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"  {method1} mean: {np.mean(r2_1):.4f}")
        print(f"  {method2} mean: {np.mean(r2_2):.4f}")
        
        # Wilcoxon Signed-Rank Test (non-parametric)
        wilcoxon_stat, wilcoxon_p = wilcoxon(r2_1, r2_2)
        print(f"\nWilcoxon Signed-Rank Test:")
        print(f"  Statistic: {wilcoxon_stat:.4f}")
        print(f"  p-value: {wilcoxon_p:.6f}")
        
        # Paired t-test (parametric)
        ttest_stat, ttest_p = ttest_rel(r2_1, r2_2)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {ttest_stat:.4f}")
        print(f"  p-value: {ttest_p:.6f}")
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(r2_1, r2_2)
        effect_interpretation = interpret_cohens_d(effect_size)
        print(f"\nEffect Size (Cohen's d): {effect_size:.4f} ({effect_interpretation})")
        
        # Interpretation
        print(f"\nInterpretation:")
        if wilcoxon_p < 0.05:
            if mean_diff > 0:
                print(f"  [+] {method1} performs SIGNIFICANTLY BETTER than {method2}")
            else:
                print(f"  [+] {method2} performs SIGNIFICANTLY BETTER than {method1}")
        else:
            print(f"  [-] No significant difference between {method1} and {method2}")
        
        pairwise_results.append({
            'Comparison': f"{method1} vs {method2}",
            'Mean Diff': mean_diff,
            'Wilcoxon p': wilcoxon_p,
            't-test p': ttest_p,
            "Cohen's d": effect_size,
            'Effect Size': effect_interpretation,
            'Significant': 'Yes' if wilcoxon_p < 0.05 else 'No'
        })
    
    # ========================================================================
    # 5. SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("5. PAIRWISE COMPARISON SUMMARY")
    print("="*80)
    
    pairwise_df = pd.DataFrame(pairwise_results)
    print("\n" + pairwise_df.to_string(index=False))
    
    # ========================================================================
    # 6. BONFERRONI CORRECTION
    # ========================================================================
    print("\n" + "="*80)
    print("6. BONFERRONI CORRECTION FOR MULTIPLE COMPARISONS")
    print("="*80)
    print(f"Number of comparisons: {len(comparisons)}")
    print(f"Original alpha level: 0.05")
    bonferroni_alpha = 0.05 / len(comparisons)
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    print("\nSignificance after Bonferroni correction:")
    
    for result in pairwise_results:
        is_sig = "YES" if result['Wilcoxon p'] < bonferroni_alpha else "NO"
        print(f"  {result['Comparison']:30} | p={result['Wilcoxon p']:.6f} | Significant: {is_sig}")
    
    # ========================================================================
    # 7. VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("7. GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Box plot comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bp = ax1.boxplot([test_r2, kfold_r2, loocv_r2], 
                      labels=methods, patch_artist=True,
                      showmeans=True, meanline=True)
    colors = ['#0173B2', '#029E73', '#CC78BC']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('R² Distribution by Validation Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    ax2 = fig.add_subplot(gs[0, 1])
    data_for_violin = pd.DataFrame({
        'Test Set': test_r2,
        'K-Fold CV': kfold_r2,
        'LOOCV': loocv_r2
    })
    parts = ax2.violinplot([test_r2, kfold_r2, loocv_r2], 
                           positions=[1, 2, 3], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean with error bars
    ax3 = fig.add_subplot(gs[0, 2])
    means = [np.mean(r2) for r2 in r2_values]
    stds = [np.std(r2, ddof=1) for r2 in r2_values]
    x_pos = np.arange(len(methods))
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=10, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel('Mean R² Score', fontsize=12, fontweight='bold')
    ax3.set_title('Mean R² with Standard Deviation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Line plot across seeds
    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(len(seeds))
    ax4.plot(x, test_r2, marker='o', linewidth=2, markersize=8, 
             label='Test Set', color=colors[0])
    ax4.plot(x, kfold_r2, marker='s', linewidth=2, markersize=8, 
             label='K-Fold CV', color=colors[1])
    ax4.plot(x, loocv_r2, marker='^', linewidth=2, markersize=8, 
             label='LOOCV', color=colors[2])
    ax4.set_xlabel('Seed Index', fontsize=12, fontweight='bold')
    ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax4.set_title('R² Scores Across Different Seeds', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"S{i}" for i in range(len(seeds))])
    
    # 5. Pairwise difference plot
    ax5 = fig.add_subplot(gs[2, 0])
    diff_test_kfold = test_r2 - kfold_r2
    diff_test_loocv = test_r2 - loocv_r2
    diff_kfold_loocv = kfold_r2 - loocv_r2
    
    bp2 = ax5.boxplot([diff_test_kfold, diff_test_loocv, diff_kfold_loocv],
                       labels=['Test-KFold', 'Test-LOOCV', 'KFold-LOOCV'],
                       patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#DE8F05')
        patch.set_alpha(0.7)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax5.set_ylabel('R² Difference', fontsize=12, fontweight='bold')
    ax5.set_title('Pairwise R² Differences', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Effect size visualization
    ax6 = fig.add_subplot(gs[2, 1])
    effect_sizes = [abs(result["Cohen's d"]) for result in pairwise_results]
    comparison_labels = [result['Comparison'].replace(' vs ', '\nvs\n') for result in pairwise_results]
    bars = ax6.barh(comparison_labels, effect_sizes, color='#CA9161', alpha=0.8)
    ax6.axvline(x=0.2, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Small (0.2)')
    ax6.axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Medium (0.5)')
    ax6.axvline(x=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Large (0.8)')
    ax6.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax6.set_title('Effect Sizes for Pairwise Comparisons', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Statistical significance heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    p_matrix = np.ones((3, 3))
    p_matrix[0, 1] = p_matrix[1, 0] = pairwise_results[0]['Wilcoxon p']
    p_matrix[0, 2] = p_matrix[2, 0] = pairwise_results[1]['Wilcoxon p']
    p_matrix[1, 2] = p_matrix[2, 1] = pairwise_results[2]['Wilcoxon p']
    
    im = ax7.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
    ax7.set_xticks(np.arange(3))
    ax7.set_yticks(np.arange(3))
    ax7.set_xticklabels(['Test', 'K-Fold', 'LOOCV'])
    ax7.set_yticklabels(['Test', 'K-Fold', 'LOOCV'])
    ax7.set_title('p-values Heatmap\n(Darker = More Significant)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if i != j:
                text = ax7.text(j, i, f'{p_matrix[i, j]:.4f}',
                               ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('p-value', fontsize=11, fontweight='bold')
    
    plt.suptitle('R² Significance Testing: Comprehensive Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('r2_significance_analysis.png', dpi=300, bbox_inches='tight')
    print("[+] Visualization saved as 'r2_significance_analysis.png'")
    plt.show()
    
    # ========================================================================
    # 8. FINAL RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("8. FINAL RECOMMENDATIONS")
    print("="*80)
    
    # Find best method
    best_method_idx = np.argmax([np.mean(test_r2), np.mean(kfold_r2), np.mean(loocv_r2)])
    best_method = methods[best_method_idx]
    
    print(f"\n[+] Best performing validation method: {best_method}")
    print(f"  Mean R²: {np.mean(r2_values[best_method_idx]):.4f}")
    print(f"  Std Dev: {np.std(r2_values[best_method_idx], ddof=1):.4f}")
    
    if friedman_p < 0.05:
        print("\n[+] Validation methods show STATISTICALLY SIGNIFICANT differences")
        print("  -> Different validation strategies yield meaningfully different results")
    else:
        print("\n[+] Validation methods show NO significant differences")
        print("  -> All validation strategies provide consistent estimates")
    
    print("\n[+] Stability Analysis:")
    cv_values = [100 * np.std(r2, ddof=1) / np.mean(r2) for r2 in r2_values]
    most_stable_idx = np.argmin(cv_values)
    print(f"  Most stable method: {methods[most_stable_idx]} (CV: {cv_values[most_stable_idx]:.2f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'descriptive_stats': stats_df,
        'normality_results': normality_results,
        'friedman_test': {'statistic': friedman_stat, 'p_value': friedman_p},
        'pairwise_results': pairwise_df,
        'bonferroni_alpha': bonferroni_alpha
    }


if __name__ == "__main__":
    # Example data from Multi_Seed_Comparison notebook
    # Replace these with your actual data
    
    print("Loading data from Multi_Seed_Comparison results...")
    print("(If you have run the Multi_Seed_Comparison notebook, the data will be loaded)")
    print("(Otherwise, example data will be used)\n")
    
    # Example seeds and R² values (replace with actual data)
    SEEDS = [42, 50, 100, 123, 200, 250, 300, 350, 400, 500]
    
    # Example R² values - REPLACE WITH YOUR ACTUAL DATA
    test_r2_values = [0.5021, 0.4856, 0.5234, 0.4923, 0.5112, 0.4987, 0.5145, 0.5034, 0.4912, 0.5089]
    kfold_r2_values = [0.5397, 0.5234, 0.5512, 0.5289, 0.5423, 0.5367, 0.5445, 0.5398, 0.5312, 0.5401]
    loocv_r2_values = [0.6456, 0.6312, 0.6589, 0.6423, 0.6534, 0.6478, 0.6512, 0.6467, 0.6389, 0.6501]
    
    # Perform significance tests
    results = perform_significance_tests(test_r2_values, kfold_r2_values, loocv_r2_values, SEEDS)
    
    print("\n" + "="*80)
    print("NOTE: To use your actual data, modify the test_r2_values, kfold_r2_values,")
    print("      and loocv_r2_values lists in this script with your results from")
    print("      the Multi_Seed_Comparison notebook.")
    print("="*80)
