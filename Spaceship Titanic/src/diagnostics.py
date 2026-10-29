import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import learning_curve


def plot_model_comparison(results_df, output_path="reports/model_comparison.png"):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    results_sorted = results_df.sort_values(by="cv_accuracy_mean", ascending=True)
    y_pos = np.arange(len(results_sorted))

    ax.barh(
        y_pos,
        results_sorted["cv_accuracy_mean"],
        xerr=results_sorted["cv_accuracy_std"],
        align="center",
        color="#3498db",
        ecolor="#2c3e50",
        capsize=5,
        alpha=0.85
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted["model"])
    ax.set_xlabel("5-Fold Stratified CV Accuracy")
    ax.set_title("Model Comparison: 5-Fold Cross-Validation Accuracy")
    ax.set_xlim(0.70, 0.85)

    for i, row in enumerate(results_sorted.itertuples()):
        ax.text(
            row.cv_accuracy_mean + 0.003,
            i,
            f"{row.cv_accuracy_mean:.4f} ± {row.cv_accuracy_std:.4f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_curves(estimator, X, y, cv, output_path="reports/diagnostic_learning_curves.png"):
    sns.set_theme(style="whitegrid")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, "o-", color="#e74c3c", label="Training Accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#e74c3c")
    ax.plot(train_sizes, val_mean, "s-", color="#2ecc71", label="Validation Accuracy")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#2ecc71")

    ax.set_title("Learning Curves (Training vs Validation Accuracy)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Classification Accuracy")
    ax.legend(loc="lower right")
    ax.set_ylim(0.75, 0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_and_calibration(y_true, y_prob, output_path="reports/confusion_matrix_calibration.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Not Transported", "Transported"],
        yticklabels=["Not Transported", "Transported"]
    )
    axes[0].set_title("Normalized Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    axes[1].plot(prob_pred, prob_true, marker="o", color="#8e44ad", label="Ensemble")
    axes[1].set_title("Probability Calibration Curve")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of Positives")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_pr_curves(y_true, y_prob, output_path="reports/roc_pr_curves.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color="#27ae60", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(importance_df, top_n=20, output_path="reports/feature_importances.png"):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance_df.sort_values(by="importance", ascending=False).head(top_n)
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_analysis(raw_df, y_true, y_prob, output_path="reports/error_analysis_deepdive.png"):
    sns.set_theme(style="whitegrid")
    df = raw_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = (y_prob >= 0.5).astype(int)
    df["prob"] = y_prob
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(
        data=df,
        x="CryoSleep",
        y="error",
        hue="CryoSleep",
        legend=False,
        ax=axes[0, 0],
        palette="Blues_d"
    )
    axes[0, 0].set_title("Error Rate by CryoSleep Status")
    axes[0, 0].set_ylabel("Misclassification Rate")

    sns.barplot(
        data=df,
        x="HasSpent",
        y="error",
        hue="HasSpent",
        legend=False,
        ax=axes[0, 1],
        palette="Reds_d"
    )
    axes[0, 1].set_title("Error Rate by Luxury Spending (0 vs >0)")
    axes[0, 1].set_ylabel("Misclassification Rate")

    deck_order = ["A", "B", "C", "D", "E", "F", "G", "T", "Missing"]
    deck_df = df[df["CabinDeck"].isin(deck_order)]
    sns.barplot(
        data=deck_df,
        x="CabinDeck",
        y="error",
        hue="CabinDeck",
        legend=False,
        order=deck_order,
        ax=axes[1, 0],
        palette="viridis"
    )
    axes[1, 0].set_title("Error Rate by Cabin Deck")
    axes[1, 0].set_ylabel("Misclassification Rate")

    sns.histplot(data=df[df["error"] == 1], x="prob", bins=30, ax=axes[1, 1], color="#e74c3c", alpha=0.5, label="Misclassified")
    sns.histplot(data=df[df["error"] == 0], x="prob", bins=30, ax=axes[1, 1], color="#2ecc71", alpha=0.3, label="Correct")
    axes[1, 1].set_title("Predicted Probability Distribution (Correct vs Errors)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()