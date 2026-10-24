"""Diagnostics module inspired by Andrew Ng's Machine Learning Specialization.

Features:
- Learning curve plotting (Course 2 Week 3: Bias vs Variance analysis).
- Error analysis on out-of-fold validation predictions (systematic failure inspection).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

def plot_learning_curves(models_dict: dict, X, y, cv, output_path: str):
    """Plot learning curves for multiple models to diagnose Bias vs Variance.
    
    Andrew Ng's rule of thumb:
    - High Bias (Underfitting): Training and CV curves plateau early at a low score with small gap.
    - High Variance (Overfitting): Training score stays very high, CV score lags with large gap.
    - Balanced: High CV score with small generalization gap.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
        
    train_sizes = np.linspace(0.15, 1.0, 6)
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(sizes, train_mean, 'o-', color='#e74c3c', label='Training Score', lw=2)
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#e74c3c')
        
        ax.plot(sizes, val_mean, 's-', color='#2980b9', label='Cross-Val Score', lw=2)
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2980b9')
        
        ax.set_title(f"{name}\nFinal CV: {val_mean[-1]:.4f} (Gap: {train_mean[-1]-val_mean[-1]:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Training Examples (m)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.65, 0.98)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic learning curves to: {output_path}")

def run_error_analysis(raw_train_df: pd.DataFrame, oof_preds: np.ndarray, y_true: np.ndarray):
    """Perform Andrew Ng-style error analysis to categorize misclassifications."""
    df_err = raw_train_df.copy()
    df_err['Pred'] = oof_preds
    df_err['True'] = y_true
    df_err['Error'] = (df_err['Pred'] != df_err['True']).astype(int)
    
    total_samples = len(df_err)
    total_errors = df_err['Error'].sum()
    error_rate = total_errors / total_samples
    
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS (Total Errors: {total_errors}/{total_samples} = {error_rate*100:.2f}%)")
    print("=" * 60)
    
    false_positives = df_err[(df_err['Pred'] == 1) & (df_err['True'] == 0)]
    false_negatives = df_err[(df_err['Pred'] == 0) & (df_err['True'] == 1)]
    
    print(f"False Positives (Predicted Survived, Actually Died): {len(false_positives)}")
    print(f"False Negatives (Predicted Died, Actually Survived): {len(false_negatives)}")
    
    print("\nFalse Positives breakdown by Sex & Class:")
    print(false_positives.groupby(['Sex', 'Pclass']).size())
    
    print("\nFalse Negatives breakdown by Sex & Class:")
    print(false_negatives.groupby(['Sex', 'Pclass']).size())
    
    return {
        'total_errors': total_errors,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }
