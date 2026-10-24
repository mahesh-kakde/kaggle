"""Comprehensive Diagnostics and Visualization Suite for Titanic ML.

Follows Andrew Ng's pedagogical & diagnostic standards:
- Bias vs Variance Learning Curves (Course 2 Week 3).
- Model Comparison (Accuracy, ROC-AUC, F1-Score).
- ROC Curves & Precision-Recall Curves.
- Confusion Matrix & Probability Calibration Curves (Reliability Diagrams).
- Multi-Model Feature Importances (Tree ensembles).
- Demographic Error Analysis Deep-Dive.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, roc_auc_score, accuracy_score
)
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", palette="muted")

def plot_learning_curves(models_dict: dict, X, y, cv, output_path: str):
    """Diagnose Bias vs Variance across training set sizes."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), sharey=True)
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
    print(f"Saved learning curves to: {output_path}")

def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str):
    """Plot multi-metric comparison bar chart across all candidate models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_melted = metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'ROC-AUC', 'F1-Score'], var_name='Metric', value_name='Score')
    
    sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', ax=ax, palette=['#2b5c8f', '#e67e22', '#27ae60'])
    ax.set_title("Model Performance Comparison (5-Fold Stratified OOF)", fontsize=13, fontweight='bold')
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha='right')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                        textcoords='offset points')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved model comparison to: {output_path}")

def plot_roc_and_pr_curves(oof_probs_dict: dict, y_true: np.ndarray, output_path: str):
    """Plot ROC curves and Precision-Recall curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = sns.color_palette("tab10", len(oof_probs_dict))
    
    for (name, probs), color in zip(oof_probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", lw=2, color=color)
        
        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_val = average_precision_score(y_true, probs)
        ax2.plot(rec, prec, label=f"{name} (AP = {pr_val:.3f})", lw=2, color=color)
        
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    base_pos = np.mean(y_true)
    ax2.axhline(base_pos, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({base_pos:.2f})')
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved ROC & PR curves to: {output_path}")

def plot_confusion_and_calibration(y_true: np.ndarray, final_preds: np.ndarray, final_probs: np.ndarray, output_path: str):
    """Plot confusion matrix heatmap and reliability / probability calibration curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, final_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title(f"Confusion Matrix (Acc: {accuracy_score(y_true, final_preds):.4f})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticklabels(['Perished (0)', 'Survived (1)'])
    ax1.set_yticklabels(['Perished (0)', 'Survived (1)'])
    
    # Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, final_probs, n_bins=8)
    ax2.plot(mean_predicted_value, fraction_of_positives, "s-", color="#2980b9", lw=2, label="Final Ensemble")
    ax2.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax2.set_title("Probability Calibration (Reliability Diagram)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Observed Fraction of Positives")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix & calibration diagram to: {output_path}")

def plot_feature_importances(importance_series_list: list, labels: list, output_path: str, top_n: int = 15):
    """Plot multi-model top feature importances horizontal bar chart."""
    combined = pd.DataFrame()
    for s, name in zip(importance_series_list, labels):
        top = s.head(top_n).reset_index()
        top.columns = ['Feature', 'Importance']
        top['Model'] = name
        combined = pd.concat([combined, top])
        
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=combined, y='Feature', x='Importance', hue='Model', ax=ax, palette='Blues_r')
    ax.set_title(f"Top {top_n} Feature Importances Across Tree Models", fontsize=13, fontweight='bold')
    ax.set_xlabel("Feature Importance Weight")
    ax.set_ylabel("")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved feature importances plot to: {output_path}")

def plot_error_analysis_deepdive(raw_train_df: pd.DataFrame, oof_preds: np.ndarray, y_true: np.ndarray, output_path: str):
    """Plot deep-dive distributions of misclassifications across demographic slices."""
    df = raw_train_df.copy()
    df['Pred'] = oof_preds
    df['True'] = y_true
    df['Result'] = 'Correct'
    df.loc[(df['Pred'] == 1) & (df['True'] == 0), 'Result'] = 'False Positive (Predicted Alive, Died)'
    df.loc[(df['Pred'] == 0) & (df['True'] == 1), 'Result'] = 'False Negative (Predicted Died, Alive)'
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    # 1. Error by Pclass
    sns.countplot(data=df[df['Result'] != 'Correct'], x='Pclass', hue='Result', ax=axes[0, 0], palette=['#e74c3c', '#3498db'])
    axes[0, 0].set_title("Errors by Passenger Class (Pclass)", fontweight='bold')
    axes[0, 0].set_xlabel("Pclass")
    axes[0, 0].set_ylabel("Count")
    
    # 2. Error by Sex
    sns.countplot(data=df[df['Result'] != 'Correct'], x='Sex', hue='Result', ax=axes[0, 1], palette=['#e74c3c', '#3498db'])
    axes[0, 1].set_title("Errors by Gender (Sex)", fontweight='bold')
    axes[0, 1].set_xlabel("Sex")
    axes[0, 1].set_ylabel("Count")
    
    # 3. Error by Age KDE
    sns.kdeplot(data=df, x='Age', hue='Result', ax=axes[1, 0], common_norm=False, fill=True, alpha=0.3)
    axes[1, 0].set_title("Age Distribution by Classification Outcome", fontweight='bold')
    axes[1, 0].set_xlabel("Age")
    
    # 4. Error by Fare Boxplot
    sns.boxplot(data=df, x='Result', y='Fare', ax=axes[1, 1], showfliers=False, palette=['#2ecc71', '#e74c3c', '#3498db'])
    axes[1, 1].set_title("Fare Distribution by Classification Outcome", fontweight='bold')
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Fare")
    axes[1, 1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved error analysis deep-dive to: {output_path}")

def run_error_analysis(raw_train_df: pd.DataFrame, oof_preds: np.ndarray, y_true: np.ndarray):
    """Categorize misclassifications in stdout."""
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
    
    print(f"False Positives: {len(false_positives)}, False Negatives: {len(false_negatives)}")
    return {
        'total_errors': total_errors,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }
