"""Training, Cross-Validation, Diagnostics, and Submission Pipeline.

Follows Andrew Ng's end-to-end ML iterative loop:
1. Stratified K-Fold Cross Validation (honest generalization estimate).
2. Diagnostic checks (learning curves: bias vs variance).
3. Error analysis on Out-Of-Fold (OOF) misclassifications.
4. Soft-Voting Ensembling across linear and tree architectures.
5. Final verified submission generation.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from features import engineer_features
from models import get_candidate_models, build_voting_ensemble
from diagnostics import plot_learning_curves, run_error_analysis

def evaluate_models_oof(models: dict, X_train: pd.DataFrame, X_train_scaled: pd.DataFrame, y_train: np.ndarray, cv):
    """Run Stratified K-Fold CV to compute Out-Of-Fold (OOF) predictions and accuracy for each model."""
    oof_predictions = {}
    oof_probabilities = {}
    cv_scores = {}
    
    for name, model in models.items():
        # LogisticRegression uses scaled inputs; trees use unscaled
        X = X_train_scaled.values if name == 'LogisticRegression' else X_train.values
        
        oof_pred = np.zeros(len(y_train))
        oof_prob = np.zeros(len(y_train))
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X, y_train):
            X_tr, y_tr = X[train_idx], y_train[train_idx]
            X_va, y_va = X[val_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            oof_prob[val_idx] = probs
            oof_pred[val_idx] = preds
            fold_scores.append(accuracy_score(y_va, preds))
            
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        overall_acc = accuracy_score(y_train, oof_pred)
        
        oof_predictions[name] = oof_pred
        oof_probabilities[name] = oof_prob
        cv_scores[name] = (cv_mean, cv_std, overall_acc)
        
        print(f"Model: {name:<20} | 5-Fold CV Mean: {cv_mean:.4f} (±{cv_std:.4f}) | OOF Acc: {overall_acc:.4f}")
        
    return oof_predictions, oof_probabilities, cv_scores

def evaluate_ensemble_oof(oof_probabilities: dict, y_train: np.ndarray, weights: dict):
    """Evaluate soft voting blend using Out-Of-Fold probability predictions."""
    total_weight = sum(weights.values())
    blended_probs = np.zeros(len(y_train))
    
    for name, weight in weights.items():
        blended_probs += oof_probabilities[name] * weight
    blended_probs /= total_weight
    
    ensemble_preds = (blended_probs >= 0.5).astype(int)
    acc = accuracy_score(y_train, ensemble_preds)
    print(f"\nModel: {'Soft-Voting Ensemble':<20} | Blended OOF Accuracy: {acc:.4f}")
    return ensemble_preds, blended_probs, acc

def train_and_generate_submission(base_dir: str):
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    submissions_dir = os.path.join(base_dir, "submissions")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 1: FEATURE ENGINEERING")
    print("=" * 70)
    data = engineer_features(
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "test.csv")
    )
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    test_ids = data['test_passenger_ids']
    raw_train = data['train_df_raw']
    feature_names = data['feature_names']
    
    print(f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    
    print("\n" + "=" * 70)
    print("PHASE 2: CROSS-VALIDATION EVALUATION & BENCHMARKING")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models()
    
    oof_preds, oof_probs, cv_scores = evaluate_models_oof(
        models, X_train, X_train_scaled, y_train, cv
    )
    
    # Soft voting weights based on validation performance
    weights = {
        'LogisticRegression': 1.0,
        'RandomForest': 2.0,
        'ExtraTrees': 1.5,
        'XGBoost': 2.0,
        'LightGBM': 2.0
    }
    ens_preds, ens_probs, ens_acc = evaluate_ensemble_oof(oof_probs, y_train, weights)
    
    print("\n" + "=" * 70)
    print("PHASE 3: ANDREW NG DIAGNOSTICS & LEARNING CURVES")
    print("=" * 70)
    diag_models = {
        'LogisticRegression': models['LogisticRegression'],
        'RandomForest': models['RandomForest'],
        'XGBoost': models['XGBoost']
    }
    plot_learning_curves(
        diag_models,
        X_train_scaled.values,
        y_train,
        cv=cv,
        output_path=os.path.join(reports_dir, "diagnostic_learning_curves.png")
    )
    
    # Phase 4: Error analysis
    print("\n" + "=" * 70)
    print("PHASE 4: ERROR ANALYSIS ON OUT-OF-FOLD PREDICTIONS")
    print("=" * 70)
    run_error_analysis(raw_train, ens_preds, y_train)
    
    # Feature importance inspection (Random Forest)
    rf = models['RandomForest']
    rf.fit(X_train.values, y_train)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 Feature Importances (Random Forest):")
    print(importances.head(10))
    
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL FULL TRAINING & SUBMISSION GENERATION")
    print("=" * 70)
    # Fit each model on the entire training set and predict on test
    test_probs = np.zeros(len(test_ids))
    total_w = sum(weights.values())
    
    for name, model in models.items():
        if name == 'LogisticRegression':
            model.fit(X_train_scaled.values, y_train)
            p = model.predict_proba(X_test_scaled.values)[:, 1]
        else:
            model.fit(X_train.values, y_train)
            p = model.predict_proba(X_test.values)[:, 1]
        test_probs += p * weights[name]
        
    test_probs /= total_w
    final_test_preds = (test_probs >= 0.5).astype(int)
    
    submission_path = os.path.join(submissions_dir, "submission.csv")
    sub_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': final_test_preds
    })
    sub_df.to_csv(submission_path, index=False)
    
    print(f"Generated submission file: {submission_path}")
    print(f"Shape: {sub_df.shape} (Expected: (418, 2))")
    print(f"Missing values: {sub_df.isnull().sum().to_dict()}")
    print("Survival distribution in predictions:")
    print(sub_df['Survived'].value_counts(normalize=True))
    
    # Verify sanity
    assert sub_df.shape == (418, 2), "Error: submission shape is incorrect!"
    assert list(sub_df.columns) == ['PassengerId', 'Survived'], "Error: columns incorrect!"
    assert sub_df.isnull().sum().sum() == 0, "Error: submission contains NaNs!"
    assert set(sub_df['Survived'].unique()).issubset({0, 1}), "Error: non-binary Survived values!"
    print("\nVERIFICATION: ALL SUBMISSION CONTRACT CHECKS PASSED.")
    return ens_acc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_generate_submission(base_dir)
