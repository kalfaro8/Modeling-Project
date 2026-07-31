"""
Multi-Seed Random Forest Model Comparison
==========================================
This script runs the Random Forest Improved model with multiple different seeds
and compares performance across all seeds, including average metrics.

Author: Generated for Nuclear Reactor Modeling Project
Date: 2026-07-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class MultiSeedRandomForest:
    """Class to handle multi-seed Random Forest experiments"""
    
    def __init__(self, seeds, data_path="Random Forest Data.xlsx"):
        """
        Initialize the multi-seed experiment
        
        Parameters:
        -----------
        seeds : list
            List of random seeds to test
        data_path : str
            Path to the data file
        """
        self.seeds = seeds
        self.data_path = data_path
        self.results = []
        self.models = {}
        self.best_params_per_seed = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the data with feature engineering"""
        # Load data
        df = pd.read_excel(self.data_path, header=1, usecols="B:L")
        
        # Filter for stream data
        stream = df[df["Rain or Stream"].str.lower().eq("stream")].copy()
        
        # Feature engineering
        stream['Month'] = stream['Date'].dt.month
        stream['Day_of_Year'] = stream['Date'].dt.dayofyear
        stream['Year'] = stream['Date'].dt.year
        stream["DOY_sin"] = np.sin(2 * np.pi * stream['Day_of_Year'] / 365.25)
        stream["DOY_cos"] = np.cos(2 * np.pi * stream['Day_of_Year'] / 365.25)
        stream['Temp_Discharge_Interaction'] = stream['Temperature '] * stream['Discharge (m3/s)']
        stream['d_excess'] = stream['d-excess (‰)']
        stream['d2H'] = stream['δ2H']
        stream['Rain'] = stream['Precipitation (mm)']
        stream['Elevation'] = stream['Elevation (ft)']
        
        # Define features
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
        
        X = stream[feature_col]
        y = stream["δ18O"]
        
        return X, y, feature_col
    
    def run_single_seed(self, seed, X, y, tune_hyperparameters=True):
        """
        Run the model with a single seed
        
        Parameters:
        -----------
        seed : int
            Random seed to use
        X : DataFrame
            Features
        y : Series
            Target variable
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        dict : Results for this seed
        """
        print(f"\n{'='*60}")
        print(f"Running model with seed: {seed}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True
        )
        
        # Baseline model
        baseline_rf = RandomForestRegressor(random_state=seed)
        baseline_rf.fit(X_train, y_train)
        y_pred_baseline = baseline_rf.predict(X_test)
        baseline_r2 = r2_score(y_test, y_pred_baseline)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
        baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
        
        print(f"Baseline - R²: {baseline_r2:.4f}, RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")
        
        # Hyperparameter tuning (optional, can be slow)
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2']
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=seed),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            optimized_rf = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            # Use default good parameters
            optimized_rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=seed
            )
            optimized_rf.fit(X_train, y_train)
            best_params = optimized_rf.get_params()
        
        # Predictions
        y_pred_optimized = optimized_rf.predict(X_test)
        
        # Metrics
        optimized_r2 = r2_score(y_test, y_pred_optimized)
        optimized_rmse = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
        optimized_mae = mean_absolute_error(y_test, y_pred_optimized)
        
        # Cross-validation
        cv_scores = cross_val_score(optimized_rf, X, y, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"Optimized - R²: {optimized_r2:.4f}, RMSE: {optimized_rmse:.4f}, MAE: {optimized_mae:.4f}")
        print(f"CV R² Mean: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Store results
        result = {
            'seed': seed,
            'baseline_r2': baseline_r2,
            'baseline_rmse': baseline_rmse,
            'baseline_mae': baseline_mae,
            'optimized_r2': optimized_r2,
            'optimized_rmse': optimized_rmse,
            'optimized_mae': optimized_mae,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'y_test': y_test.values,
            'y_pred': y_pred_optimized,
            'feature_importance': dict(zip(X.columns, optimized_rf.feature_importances_))
        }
        
        # Store model and parameters
        self.models[seed] = optimized_rf
        self.best_params_per_seed[seed] = best_params
        
        return result
    
    def run_all_seeds(self, tune_hyperparameters=True):
        """Run the model for all seeds"""
        X, y, feature_col = self.load_and_prepare_data()
        
        print(f"\n{'='*60}")
        print(f"Starting Multi-Seed Experiment")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"Features: {feature_col}")
        print(f"Seeds to test: {self.seeds}")
        print(f"Hyperparameter tuning: {tune_hyperparameters}")
        
        for seed in self.seeds:
            result = self.run_single_seed(seed, X, y, tune_hyperparameters)
            self.results.append(result)
        
        print(f"\n{'='*60}")
        print("All seeds completed!")
        print(f"{'='*60}")
    
    def calculate_statistics(self):
        """Calculate average and statistics across all seeds"""
        metrics = ['baseline_r2', 'baseline_rmse', 'baseline_mae',
                   'optimized_r2', 'optimized_rmse', 'optimized_mae',
                   'cv_r2_mean']
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in self.results]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return stats
    
    def print_summary(self):
        """Print summary of results"""
        stats = self.calculate_statistics()
        
        print(f"\n{'='*70}")
        print("MULTI-SEED EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Number of seeds tested: {len(self.seeds)}")
        print(f"Seeds: {self.seeds}")
        
        print(f"\n{'='*70}")
        print("AVERAGE PERFORMANCE ACROSS ALL SEEDS")
        print(f"{'='*70}")
        
        print("\nBaseline Model:")
        print(f"  R² Score:  {stats['baseline_r2']['mean']:.4f} ± {stats['baseline_r2']['std']:.4f}")
        print(f"  RMSE:      {stats['baseline_rmse']['mean']:.4f} ± {stats['baseline_rmse']['std']:.4f} ‰")
        print(f"  MAE:       {stats['baseline_mae']['mean']:.4f} ± {stats['baseline_mae']['std']:.4f} ‰")
        
        print("\nOptimized Model:")
        print(f"  R² Score:  {stats['optimized_r2']['mean']:.4f} ± {stats['optimized_r2']['std']:.4f}")
        print(f"  RMSE:      {stats['optimized_rmse']['mean']:.4f} ± {stats['optimized_rmse']['std']:.4f} ‰")
        print(f"  MAE:       {stats['optimized_mae']['mean']:.4f} ± {stats['optimized_mae']['std']:.4f} ‰")
        
        print("\nCross-Validation:")
        print(f"  R² Score:  {stats['cv_r2_mean']['mean']:.4f} ± {stats['cv_r2_mean']['std']:.4f}")
        
        print(f"\n{'='*70}")
        print("PERFORMANCE RANGE")
        print(f"{'='*70}")
        print(f"Best R² Score:  {stats['optimized_r2']['max']:.4f} (seed: {self.seeds[np.argmax(stats['optimized_r2']['values'])]})")
        print(f"Worst R² Score: {stats['optimized_r2']['min']:.4f} (seed: {self.seeds[np.argmin(stats['optimized_r2']['values'])]})")
        print(f"Best RMSE:      {stats['optimized_rmse']['min']:.4f} ‰ (seed: {self.seeds[np.argmin(stats['optimized_rmse']['values'])]})")
        print(f"Worst RMSE:     {stats['optimized_rmse']['max']:.4f} ‰ (seed: {self.seeds[np.argmax(stats['optimized_rmse']['values'])]})")
        
        print(f"\n{'='*70}")
        print("INDIVIDUAL SEED RESULTS")
        print(f"{'='*70}")
        print(f"{'Seed':<10} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'CV R²':<10}")
        print(f"{'-'*70}")
        for result in self.results:
            print(f"{result['seed']:<10} {result['optimized_r2']:<10.4f} "
                  f"{result['optimized_rmse']:<10.4f} {result['optimized_mae']:<10.4f} "
                  f"{result['cv_r2_mean']:<10.4f}")
        print(f"{'='*70}")
    
    def plot_comparison(self, save_path="multi_seed_comparison.png"):
        """Create comprehensive comparison plots"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. R² Score comparison across seeds
        ax1 = fig.add_subplot(gs[0, 0])
        seeds_list = [r['seed'] for r in self.results]
        r2_scores = [r['optimized_r2'] for r in self.results]
        ax1.bar(range(len(seeds_list)), r2_scores, color='#0173B2', alpha=0.8)
        ax1.axhline(y=np.mean(r2_scores), color='#DE8F05', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(r2_scores):.4f}')
        ax1.set_xlabel('Seed Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('R² Score Across Seeds', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE comparison across seeds
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_scores = [r['optimized_rmse'] for r in self.results]
        ax2.bar(range(len(seeds_list)), rmse_scores, color='#029E73', alpha=0.8)
        ax2.axhline(y=np.mean(rmse_scores), color='#DE8F05', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(rmse_scores):.4f}')
        ax2.set_xlabel('Seed Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RMSE (‰)', fontsize=12, fontweight='bold')
        ax2.set_title('RMSE Across Seeds', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE comparison across seeds
        ax3 = fig.add_subplot(gs[0, 2])
        mae_scores = [r['optimized_mae'] for r in self.results]
        ax3.bar(range(len(seeds_list)), mae_scores, color='#CC78BC', alpha=0.8)
        ax3.axhline(y=np.mean(mae_scores), color='#DE8F05', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(mae_scores):.4f}')
        ax3.set_xlabel('Seed Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MAE (‰)', fontsize=12, fontweight='bold')
        ax3.set_title('MAE Across Seeds', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot of metrics
        ax4 = fig.add_subplot(gs[1, 0])
        metrics_data = [r2_scores, rmse_scores, mae_scores]
        bp = ax4.boxplot(metrics_data, labels=['R²', 'RMSE', 'MAE'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#0173B2', '#029E73', '#CC78BC']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax4.set_title('Metric Distribution Across Seeds', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Predicted vs Observed for all seeds (overlay)
        ax5 = fig.add_subplot(gs[1, 1])
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        for i, result in enumerate(self.results):
            ax5.scatter(result['y_test'], result['y_pred'], 
                       alpha=0.5, s=30, color=colors[i], 
                       label=f"Seed {result['seed']}")
        
        # Add perfect prediction line
        all_y = np.concatenate([r['y_test'] for r in self.results])
        min_val, max_val = all_y.min(), all_y.max()
        ax5.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=2, label='Perfect Prediction')
        ax5.set_xlabel('Observed δ18O (‰)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Predicted δ18O (‰)', fontsize=12, fontweight='bold')
        ax5.set_title('Predictions Across All Seeds', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=8, loc='best')
        ax5.grid(True, alpha=0.3)
        
        # 6. Baseline vs Optimized comparison
        ax6 = fig.add_subplot(gs[1, 2])
        baseline_r2 = [r['baseline_r2'] for r in self.results]
        optimized_r2 = [r['optimized_r2'] for r in self.results]
        x = np.arange(len(seeds_list))
        width = 0.35
        ax6.bar(x - width/2, baseline_r2, width, label='Baseline', 
               color='#CA9161', alpha=0.8)
        ax6.bar(x + width/2, optimized_r2, width, label='Optimized', 
               color='#0173B2', alpha=0.8)
        ax6.set_xlabel('Seed Index', fontsize=12, fontweight='bold')
        ax6.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax6.set_title('Baseline vs Optimized', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Feature importance heatmap
        ax7 = fig.add_subplot(gs[2, :])
        feature_names = list(self.results[0]['feature_importance'].keys())
        importance_matrix = np.array([[r['feature_importance'][f] for f in feature_names] 
                                     for r in self.results])
        
        im = ax7.imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
        ax7.set_yticks(range(len(seeds_list)))
        ax7.set_yticklabels([f"Seed {s}" for s in seeds_list])
        ax7.set_xticks(range(len(feature_names)))
        ax7.set_xticklabels(feature_names, rotation=45, ha='right')
        ax7.set_title('Feature Importance Across Seeds', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Importance', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(seeds_list)):
            for j in range(len(feature_names)):
                text = ax7.text(j, i, f'{importance_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.suptitle('Multi-Seed Random Forest Model Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
        plt.show()
    
    def save_results(self, filename="multi_seed_results.json"):
        """Save results to JSON file"""
        stats = self.calculate_statistics()
        
        # Prepare data for JSON (convert numpy types)
        output = {
            'experiment_info': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seeds': self.seeds,
                'num_seeds': len(self.seeds)
            },
            'summary_statistics': {
                metric: {
                    'mean': float(stats[metric]['mean']),
                    'std': float(stats[metric]['std']),
                    'min': float(stats[metric]['min']),
                    'max': float(stats[metric]['max'])
                }
                for metric in stats.keys()
            },
            'individual_results': [
                {
                    'seed': r['seed'],
                    'baseline_r2': float(r['baseline_r2']),
                    'baseline_rmse': float(r['baseline_rmse']),
                    'baseline_mae': float(r['baseline_mae']),
                    'optimized_r2': float(r['optimized_r2']),
                    'optimized_rmse': float(r['optimized_rmse']),
                    'optimized_mae': float(r['optimized_mae']),
                    'cv_r2_mean': float(r['cv_r2_mean']),
                    'cv_r2_std': float(r['cv_r2_std']),
                    'feature_importance': {k: float(v) for k, v in r['feature_importance'].items()}
                }
                for r in self.results
            ],
            'best_parameters_per_seed': {
                str(seed): {k: str(v) for k, v in params.items()}
                for seed, params in self.best_params_per_seed.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    """Main function to run the multi-seed experiment"""
    
    # Define seeds to test
    # You can modify this list to test different seeds
    seeds = [42, 50, 100, 123, 200, 250, 300, 350, 400, 500]
    
    print("="*70)
    print("MULTI-SEED RANDOM FOREST EXPERIMENT")
    print("="*70)
    print(f"Testing {len(seeds)} different random seeds")
    print(f"Seeds: {seeds}")
    print("="*70)
    
    # Create experiment instance
    experiment = MultiSeedRandomForest(seeds=seeds)
    
    # Run all seeds (set tune_hyperparameters=False for faster execution)
    # Note: Hyperparameter tuning can be slow. Set to False to use default good parameters.
    experiment.run_all_seeds(tune_hyperparameters=False)
    
    # Print summary
    experiment.print_summary()
    
    # Create comparison plots
    experiment.plot_comparison(save_path="multi_seed_comparison.png")
    
    # Save results to JSON
    experiment.save_results(filename="multi_seed_results.json")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. multi_seed_comparison.png - Comprehensive visualization")
    print("  2. multi_seed_results.json - Detailed results in JSON format")
    print("="*70)


if __name__ == "__main__":
    main()
