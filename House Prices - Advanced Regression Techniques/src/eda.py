import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def run_eda(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    print(f"train shape: {train_df.shape}, test shape: {test_df.shape}")
    print(f"raw saleprice skew: {train_df['SalePrice'].skew():.4f}")
    log_saleprice = np.log1p(train_df['SalePrice'])
    print(f"log1p saleprice skew: {log_saleprice.skew():.4f}")
    
    # check numerical correlations
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    corrs = train_df[num_cols].corr()['SalePrice'].sort_values(ascending=False)
    print("\ntop 10 correlated features:")
    print(corrs.head(11))
    
    # generate visual summary
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # raw target distribution
    sns.histplot(train_df['SalePrice'], kde=True, ax=axes[0, 0], color='#2980b9')
    axes[0, 0].set_title(f"SalePrice Distribution (skew: {train_df['SalePrice'].skew():.2f})", fontsize=11)
    
    # log transformed target distribution
    sns.histplot(log_saleprice, kde=True, ax=axes[0, 1], color='#27ae60')
    axes[0, 1].set_title(f"log1p(SalePrice) Distribution (skew: {log_saleprice.skew():.2f})", fontsize=11)
    
    # grlivarea vs saleprice scatter and outliers
    axes[1, 0].scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.5, color='#34495e', edgecolors='none')
    outliers = train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)]
    axes[1, 0].scatter(outliers['GrLivArea'], outliers['SalePrice'], color='#e74c3c', s=60, label='outliers (grlivarea > 4000)')
    axes[1, 0].set_title("GrLivArea vs SalePrice", fontsize=11)
    axes[1, 0].set_xlabel("GrLivArea (sq ft)")
    axes[1, 0].set_ylabel("SalePrice")
    axes[1, 0].legend()
    
    # top 10 correlations
    top_corrs = corrs.iloc[1:11]
    sns.barplot(x=top_corrs.values, y=top_corrs.index, ax=axes[1, 1], hue=top_corrs.index, palette='Blues_r', legend=False)
    axes[1, 1].set_title("Top Correlated Features with SalePrice", fontsize=11)
    axes[1, 1].set_xlabel("Pearson Correlation")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_distributions.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"saved eda plot to: {plot_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_eda(os.path.join(base_dir, "data"), os.path.join(base_dir, "reports"))