"""Exploratory Data Analysis (EDA) for Titanic: Machine Learning from Disaster.

Follows Andrew Ng's intuition-driven approach:
- Build intuition on raw distributions and feature relationships first.
- Visualize univariate, bivariate, and interaction patterns with the target (Survived).
- Identify key survival drivers and potential data leakage / imputation traps.
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def run_eda(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("=" * 60)
    print("1. DATASET OVERVIEW & SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    base_survival = train_df['Survived'].mean()
    print(f"Overall Survival Rate in Train: {base_survival:.4f} ({train_df['Survived'].sum()}/{len(train_df)})")
    
    # Missing values
    print("\nMissing values in Train:")
    for col, count in train_df.isnull().sum()[train_df.isnull().sum() > 0].items():
        print(f"  {col}: {count} ({count/len(train_df)*100:.1f}%)")
        
    print("\nMissing values in Test:")
    for col, count in test_df.isnull().sum()[test_df.isnull().sum() > 0].items():
        print(f"  {col}: {count} ({count/len(test_df)*100:.1f}%)")
        
    print("\n" + "=" * 60)
    print("2. CORE SURVIVAL DRIVERS: GENDER & CLASS")
    print("=" * 60)
    gender_stat = train_df.groupby('Sex')['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Sex:")
    print(gender_stat)
    
    pclass_stat = train_df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Pclass:")
    print(pclass_stat)
    
    sex_pclass_stat = train_df.groupby(['Sex', 'Pclass'])['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Sex & Pclass:")
    print(sex_pclass_stat)
    
    print("\n" + "=" * 60)
    print("3. EXTRACTING TITLES FROM NAMES")
    print("=" * 60)
    def extract_title(name):
        match = re.search(r',\s*([^\.]+)\.', name)
        return match.group(1).strip() if match else 'Unknown'
        
    train_df['Title'] = train_df['Name'].apply(extract_title)
    test_df['Title'] = test_df['Name'].apply(extract_title)
    
    print("Train Title counts and survival rates:")
    title_stat = train_df.groupby('Title')['Survived'].agg(['count', 'mean'])
    print(title_stat.sort_values(by='count', ascending=False))
    
    # Check median age per title across train + test
    combined = pd.concat([train_df, test_df], ignore_index=True)
    title_age = combined.groupby('Title')['Age'].agg(['count', 'median', 'mean'])
    print("\nMedian Age by Title (Train + Test):")
    print(title_age.sort_values(by='count', ascending=False))

    print("\n" + "=" * 60)
    print("4. FAMILY DYNAMICS & GROUP SIZES")
    print("=" * 60)
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    fam_stat = train_df.groupby('FamilySize')['Survived'].agg(['count', 'mean'])
    print("Survival by FamilySize:")
    print(fam_stat)
    
    # Plotting Key Insights
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Survival by Sex and Pclass
    sns.barplot(ax=axes[0, 0], data=train_df, x='Pclass', y='Survived', hue='Sex', errorbar=None)
    axes[0, 0].set_title("Survival Rate by Class and Sex", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].axhline(base_survival, color='red', linestyle='--', label=f'Baseline ({base_survival:.2f})')
    axes[0, 0].legend()
    
    # Plot 2: Age Distribution by Survival Status
    sns.kdeplot(ax=axes[0, 1], data=train_df, x='Age', hue='Survived', fill=True, common_norm=False, alpha=0.4)
    axes[0, 1].set_title("Age Distribution by Survival (KDE)", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Age")
    
    # Plot 3: Survival by Family Size
    sns.barplot(ax=axes[1, 0], data=train_df, x='FamilySize', y='Survived', color='#3470a3', errorbar=None)
    axes[1, 0].set_title("Survival Rate by Family Size", fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel("Survival Rate")
    axes[1, 0].axhline(base_survival, color='red', linestyle='--')
    
    # Plot 4: Fare distribution by Pclass (log scale)
    sns.boxplot(ax=axes[1, 1], data=train_df, x='Pclass', y='Fare', hue='Survived', showfliers=False)
    axes[1, 1].set_title("Fare Distribution by Class and Survival (without extreme fliers)", fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel("Fare")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_key_insights.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved key insights visualization to: {plot_path}")
    print("EDA completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "reports")
    run_eda(data_dir, output_dir)
