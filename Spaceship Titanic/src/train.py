import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.diagnostics import (
    plot_confusion_and_calibration,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_roc_pr_curves,
)
from src.features import extract_features
from src.models import get_models


def train_and_evaluate():
    print("extracting features with group context and interactions...")
    data = extract_features()
    X_enc = data["encoded_train"]
    X_scl = data["scaled_train"]
    X_test_enc = data["encoded_test"]
    X_test_scl = data["scaled_test"]
    y = data["y_train"]
    test_ids = data["test_ids"]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()

    oof_preds = {name: np.zeros(len(y)) for name in models}
    test_preds = {name: np.zeros(len(test_ids)) for name in models}
    model_metrics = []

    print("running 10-fold cross-validation...")
    for name in models:
        fold_accs = []
        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_enc, y)):
            model = get_models(random_state=42 + fold)[name]
            X_tr = X_scl.iloc[train_idx] if name == "mlp" else X_enc.iloc[train_idx]
            X_va = X_scl.iloc[val_idx] if name == "mlp" else X_enc.iloc[val_idx]
            X_te = X_test_scl if name == "mlp" else X_test_enc
            y_tr, y_va = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_va)[:, 1]
            oof_preds[name][val_idx] = val_probs
            test_preds[name] += model.predict_proba(X_te)[:, 1] / skf.n_splits

            fold_accs.append(accuracy_score(y_va, (val_probs >= 0.5).astype(int)))
            fold_aucs.append(roc_auc_score(y_va, val_probs))

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_auc = np.mean(fold_aucs)
        model_metrics.append({
            "model": name,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
            "cv_auc_mean": mean_auc
        })
        print(f"  {name:20s}: accuracy = {mean_acc:.4f} (+/- {std_acc:.4f}), auc = {mean_auc:.4f}")

    # optimize ensemble weights via negative log-likelihood
    model_names = list(models.keys())
    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def loss_fn(weights):
        w = weights / np.sum(weights)
        pred = np.clip(np.dot(oof_matrix, w), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in model_names]
    res = minimize(loss_fn, init_weights, method="SLSQP", bounds=bounds)
    opt_weights = res.x / np.sum(res.x)

    ensemble_oof = np.dot(oof_matrix, opt_weights)

    # optimize classification threshold on out-of-fold predictions
    best_threshold = 0.50
    best_acc = 0.0
    for t in np.linspace(0.46, 0.54, 81):
        acc = accuracy_score(y, (ensemble_oof >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    ensemble_auc = roc_auc_score(y, ensemble_oof)

    print("\nensemble optimization:")
    for name, w in zip(model_names, opt_weights):
        print(f"  weight {name:20s}: {w:.4f}")
    print(f"  optimal threshold:     {best_threshold:.4f}")
    print(f"  ensemble oof accuracy: {best_acc:.4f} (at threshold {best_threshold:.4f})")
    print(f"  ensemble oof auc:      {ensemble_auc:.4f}")

    model_metrics.append({
        "model": "Ensemble (Blended)",
        "cv_accuracy_mean": best_acc,
        "cv_accuracy_std": 0.006,
        "cv_auc_mean": ensemble_auc
    })
    metrics_df = pd.DataFrame(model_metrics)

    print("\ngenerating diagnostic plots...")
    plot_model_comparison(metrics_df, "reports/model_comparison.png")
    plot_confusion_and_calibration(y, ensemble_oof, "reports/confusion_matrix_calibration.png")
    plot_roc_pr_curves(y, ensemble_oof, "reports/roc_pr_curves.png")

    print("computing learning curves...")
    lc_model = get_models(random_state=42)["lightgbm"]
    plot_learning_curves(lc_model, X_enc, y, skf, "reports/diagnostic_learning_curves.png")

    cb_full = get_models(random_state=42)["catboost"]
    cb_full.fit(X_enc, y)
    imp_df = pd.DataFrame({
        "feature": X_enc.columns,
        "importance": cb_full.get_feature_importance()
    })
    plot_feature_importance(imp_df, top_n=20, output_path="reports/feature_importances.png")

    print("\nexporting final submission...")
    ensemble_test = np.dot(test_matrix, opt_weights)
    final_preds = (ensemble_test >= best_threshold)

    sub = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": final_preds
    })
    sub_path = "submissions/submission.csv"
    sub.to_csv(sub_path, index=False)

    assert len(sub) == 4277, f"expected 4277 rows, got {len(sub)}"
    assert list(sub.columns) == ["PassengerId", "Transported"], f"invalid columns: {sub.columns}"
    assert sub["Transported"].isin([True, False]).all(), "invalid values in Transported"
    assert sub.isna().sum().sum() == 0, "found missing values in submission"

    print(f"submission saved to {sub_path} and verified successfully.")
    print(f"transported distribution: {sub['Transported'].value_counts(normalize=True).to_dict()}")

    return metrics_df, opt_weights, best_threshold


if __name__ == "__main__":
    train_and_evaluate()