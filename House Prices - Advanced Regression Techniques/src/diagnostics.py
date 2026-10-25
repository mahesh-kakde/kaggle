import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import learning_curve
from sklearn.metrics import root_mean_squared_error

sns.set_theme(style="whitegrid")

def plot_model_comparison(metrics_dict: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame(list(metrics_dict.items()), columns=['model', 'log_rmse']).sort_values('log_rmse')
    
    sns.barplot(data=df, x='model', y='log_rmse', ax=ax, hue='model', palette='Blues_r', legend=False)
    ax.set_title("5-Fold Cross-Validation Log-RMSE Comparison", fontsize=12)
    ax.set_ylabel("Log-RMSE (lower is better)")
    ax.set_xlabel("")
    ax.set_ylim(0.09, max(df['log_rmse']) * 1.08)
    plt.xticks(rotation=20, ha='right')
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_learning_curves(models_dict: dict, X, y, cv, output_path: str):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5.5 * len(models_dict), 4.5), sharey=True)
    if len(models_dict) == 1:
        axes = [axes]
        
    train_sizes = np.linspace(0.15, 1.0, 6)
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(sizes, train_mean, 'o-', color='#e74c3c', label='train rmse', lw=2)
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#e74c3c')
        
        ax.plot(sizes, val_mean, 's-', color='#2980b9', label='cv rmse', lw=2)
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2980b9')
        
        ax.set_title(f"{name}\nfinal cv: {val_mean[-1]:.4f} (gap: {val_mean[-1]-train_mean[-1]:.4f})", fontsize=11)
        ax.set_xlabel("training examples")
        ax.set_ylabel("log-rmse")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_residuals_and_qq(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # residuals vs fitted
    ax1.scatter(y_pred, residuals, alpha=0.5, color='#2980b9', edgecolors='none')
    ax1.axhline(0, color='#e74c3c', linestyle='--', lw=2)
    ax1.set_title("Residuals vs Fitted Values", fontsize=12)
    ax1.set_xlabel("Fitted Values (log scale)")
    ax1.set_ylabel("Residuals")
    
    # normal q-q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q Plot of Residuals", fontsize=12)
    ax2.get_lines()[0].set_color('#2980b9')
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[1].set_color('#e74c3c')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5, color='#2c3e50', edgecolors='none')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='ideal (y = x)')
    
    rmse = root_mean_squared_error(y_true, y_pred)
    ax.set_title(f"Predicted vs Actual log(SalePrice)\n(OOF RMSE: {rmse:.4f})", fontsize=12)
    ax.set_xlabel("Actual log(SalePrice)")
    ax.set_ylabel("Predicted log(SalePrice)")
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_feature_importances(importance_series_list: list, labels: list, output_path: str, top_n: int = 15):
    combined = pd.DataFrame()
    for s, name in zip(importance_series_list, labels):
        top = s.head(top_n).reset_index()
        top.columns = ['feature', 'importance']
        top['model'] = name
        combined = pd.concat([combined, top])
        
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=combined, y='feature', x='importance', hue='model', ax=ax, palette='Blues_r')
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=12)
    ax.set_xlabel("Importance Weight")
    ax.set_ylabel("")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()