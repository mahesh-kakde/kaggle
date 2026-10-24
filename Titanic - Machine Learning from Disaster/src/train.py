"""Master Training, Diagnostics, Artifact Generation, and Ensembling Pipeline for Titanic.

Key Highlights:
1. 7 diverse base architectures (LogisticRegression, CalibratedSVC, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost).
2. Advanced Stacking & Soft-Voting Ensembles.
3. 5-Fold Stratified Cross-Validation out-of-fold metrics (Accuracy, ROC-AUC, F1).
4. Full diagnostic artifacts: Learning Curves, ROC/PR Curves, Confusion Matrix, Calibration, Feature Importances, Error Deep-Dive.
5. High-confidence Woman-Child-Group and Adult-Male-Group alignment.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

from features import engineer_features
from models import get_candidate_models
from diagnostics import (
    plot_learning_curves, plot_model_comparison, plot_roc_and_pr_curves,
    plot_confusion_and_calibration, plot_feature_importances,
    plot_error_analysis_deepdive, run_error_analysis
)

def evaluate_models_oof(models: dict, X_train: pd.DataFrame, X_train_scaled: pd.DataFrame, y_train: np.ndarray, cv):
    """Run Stratified K-Fold CV to compute Out-Of-Fold probability and class predictions for all models."""
    oof_predictions = {}
    oof_probabilities = {}
    metrics_records = []
    
    for name, model in models.items():
        X = X_train_scaled.values if name in ['LogisticRegression', 'CalibratedSVC'] else X_train.values
        
        oof_pred = np.zeros(len(y_train))
        oof_prob = np.zeros(len(y_train))
        
        for train_idx, val_idx in cv.split(X, y_train):
            X_tr, y_tr = X[train_idx], y_train[train_idx]
            X_va, y_va = X[val_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            oof_prob[val_idx] = probs
            oof_pred[val_idx] = preds
            
        acc = accuracy_score(y_train, oof_pred)
        auc_val = roc_auc_score(y_train, oof_prob)
        f1 = f1_score(y_train, oof_pred)
        
        oof_predictions[name] = oof_pred
        oof_probabilities[name] = oof_prob
        metrics_records.append({
            'Model': name,
            'Accuracy': acc,
            'ROC-AUC': auc_val,
            'F1-Score': f1
        })
        print(f"Model: {name:<20} | OOF Acc: {acc:.4f} | ROC-AUC: {auc_val:.4f} | F1: {f1:.4f}")
        
    return oof_predictions, oof_probabilities, metrics_records

def train_and_generate_all(base_dir: str):
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    submissions_dir = os.path.join(base_dir, "submissions")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    
    print("=" * 75)
    print("PHASE 1: FEATURE ENGINEERING & DATA PREPARATION")
    print("=" * 75)
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
    raw_test = data['test_df_raw']
    feature_names = data['feature_names']
    combined_df = data['combined_df']
    
    print(f"Total features: {len(feature_names)}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    print("\n" + "=" * 75)
    print("PHASE 2: 5-FOLD STRATIFIED CV BENCHMARKING (7 ARCHITECTURES)")
    print("=" * 75)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models()
    
    oof_preds, oof_probs, metrics_records = evaluate_models_oof(
        models, X_train, X_train_scaled, y_train, cv
    )
    
    # 1. Soft-Voting Blend
    weights = {
        'LogisticRegression': 1.0,
        'CalibratedSVC': 1.5,
        'RandomForest': 2.0,
        'ExtraTrees': 1.5,
        'XGBoost': 2.5,
        'LightGBM': 2.0,
        'CatBoost': 2.0
    }
    tot_w = sum(weights.values())
    voting_probs = sum(oof_probs[k] * weights[k] for k in weights) / tot_w
    voting_preds = (voting_probs >= 0.5).astype(int)
    voting_acc = accuracy_score(y_train, voting_preds)
    voting_auc = roc_auc_score(y_train, voting_probs)
    voting_f1 = f1_score(y_train, voting_preds)
    
    print(f"\nModel: {'Soft-Voting Ensemble':<20} | OOF Acc: {voting_acc:.4f} | ROC-AUC: {voting_auc:.4f} | F1: {voting_f1:.4f}")
    metrics_records.append({
        'Model': 'Soft-Voting',
        'Accuracy': voting_acc,
        'ROC-AUC': voting_auc,
        'F1-Score': voting_f1
    })
    
    # 2. Stacking Meta-Learner (Regularized Logistic Regression on OOF probabilities)
    stack_train = np.column_stack([oof_probs[k] for k in models])
    meta_model = LogisticRegression(C=0.1, solver='lbfgs', random_state=42)
    stacking_probs = np.zeros(len(y_train))
    
    for tr, va in cv.split(stack_train, y_train):
        meta_model.fit(stack_train[tr], y_train[tr])
        stacking_probs[va] = meta_model.predict_proba(stack_train[va])[:, 1]
        
    stacking_preds = (stacking_probs >= 0.5).astype(int)
    stacking_acc = accuracy_score(y_train, stacking_preds)
    stacking_auc = roc_auc_score(y_train, stacking_probs)
    stacking_f1 = f1_score(y_train, stacking_preds)
    
    print(f"Model: {'Stacking Ensemble':<20} | OOF Acc: {stacking_acc:.4f} | ROC-AUC: {stacking_auc:.4f} | F1: {stacking_f1:.4f}")
    metrics_records.append({
        'Model': 'Stacking Meta',
        'Accuracy': stacking_acc,
        'ROC-AUC': stacking_auc,
        'F1-Score': stacking_f1
    })
    
    # Choose best ensemble
    if stacking_acc >= voting_acc:
        final_oof_preds = stacking_preds
        final_oof_probs = stacking_probs
        best_ens_name = 'Stacking Meta-Learner'
        best_ens_acc = stacking_acc
    else:
        final_oof_preds = voting_preds
        final_oof_probs = voting_probs
        best_ens_name = 'Soft-Voting Ensemble'
        best_ens_acc = voting_acc
        
    print(f"\nSelected Best Ensemble: {best_ens_name} (OOF Acc: {best_ens_acc:.4f})")
    
    print("\n" + "=" * 75)
    print("PHASE 3: GENERATING COMPREHENSIVE DIAGNOSTIC PLOTS")
    print("=" * 75)
    metrics_df = pd.DataFrame(metrics_records)
    
    # 1. Model comparison plot
    plot_model_comparison(metrics_df, os.path.join(reports_dir, "model_comparison.png"))
    
    # 2. Andrew Ng Bias vs Variance Learning Curves
    diag_models = {
        'LogisticRegression': models['LogisticRegression'],
        'RandomForest': models['RandomForest'],
        'XGBoost': models['XGBoost'],
        'CatBoost': models['CatBoost']
    }
    plot_learning_curves(diag_models, X_train_scaled.values, y_train, cv, os.path.join(reports_dir, "diagnostic_learning_curves.png"))
    
    # 3. ROC & Precision-Recall curves
    prob_dict_curves = {
        'LogisticRegression': oof_probs['LogisticRegression'],
        'RandomForest': oof_probs['RandomForest'],
        'XGBoost': oof_probs['XGBoost'],
        'CatBoost': oof_probs['CatBoost'],
        'Ensemble': final_oof_probs
    }
    plot_roc_and_pr_curves(prob_dict_curves, y_train, os.path.join(reports_dir, "roc_pr_curves.png"))
    
    # 4. Confusion Matrix & Calibration Diagram
    plot_confusion_and_calibration(y_train, final_oof_preds, final_oof_probs, os.path.join(reports_dir, "confusion_matrix_calibration.png"))
    
    # 5. Feature Importances (Tree ensembles)
    rf = models['RandomForest']
    rf.fit(X_train.values, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    xgb = models['XGBoost']
    xgb.fit(X_train.values, y_train)
    xgb_imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    cb = models['CatBoost']
    cb.fit(X_train.values, y_train)
    cb_imp = pd.Series(cb.get_feature_importance(), index=feature_names).sort_values(ascending=False)
    
    plot_feature_importances([rf_imp, xgb_imp, cb_imp], ['RandomForest', 'XGBoost', 'CatBoost'], os.path.join(reports_dir, "feature_importances.png"))
    
    # 6. Demographic Error Analysis Deep-Dive
    plot_error_analysis_deepdive(raw_train, final_oof_preds, y_train, os.path.join(reports_dir, "error_analysis_deepdive.png"))
    run_error_analysis(raw_train, final_oof_preds, y_train)
    
    print("\n" + "=" * 75)
    print("PHASE 4: FULL DATASET TRAINING & OPTIMIZED SUBMISSION")
    print("=" * 75)
    # Fit each model on entire training set
    test_probs_matrix = []
    
    for name, model in models.items():
        if name in ['LogisticRegression', 'CalibratedSVC']:
            model.fit(X_train_scaled.values, y_train)
            p = model.predict_proba(X_test_scaled.values)[:, 1]
        else:
            model.fit(X_train.values, y_train)
            p = model.predict_proba(X_test.values)[:, 1]
        test_probs_matrix.append(p)
        
    test_probs_matrix = np.column_stack(test_probs_matrix)
    
    if best_ens_name == 'Stacking Meta-Learner':
        meta_model.fit(stack_train, y_train)
        final_test_probs = meta_model.predict_proba(test_probs_matrix)[:, 1]
    else:
        final_test_probs = sum(test_probs_matrix[:, i] * list(weights.values())[i] for i in range(len(weights))) / tot_w
        
    raw_test_preds = (final_test_probs >= 0.5).astype(int)
    
    # High-confidence Woman-Child-Group and Adult-Male-Group alignment
    test_combined = combined_df.iloc[len(raw_train):].copy().reset_index(drop=True)
    aligned_test_preds = raw_test_preds.copy()
    
    flips_to_survive = 0
    flips_to_die = 0
    for i in range(len(test_ids)):
        grp_surv = test_combined.loc[i, 'GroupSurvival']
        if grp_surv == 1.0 and aligned_test_preds[i] == 0:
            aligned_test_preds[i] = 1
            flips_to_survive += 1
            print(f"High-confidence survival override -> 1: ID {test_ids[i]} ({test_combined.loc[i, 'Name']})")
        elif grp_surv == 0.0 and aligned_test_preds[i] == 1:
            aligned_test_preds[i] = 0
            flips_to_die += 1
            print(f"High-confidence mortality override -> 0: ID {test_ids[i]} ({test_combined.loc[i, 'Name']})")
            
    print(f"Group alignment summary: {flips_to_survive} flipped to Survived, {flips_to_die} flipped to Deceased.")
    
    submission_path = os.path.join(submissions_dir, "submission.csv")
    sub_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': aligned_test_preds
    })
    sub_df.to_csv(submission_path, index=False)
    
    print(f"\nFinal submission successfully exported to: {submission_path}")
    print(f"Shape: {sub_df.shape} | Nulls: {sub_df.isnull().sum().to_dict()}")
    print("Class ratio in test predictions:")
    print(sub_df['Survived'].value_counts(normalize=True))
    
    assert sub_df.shape == (418, 2)
    assert list(sub_df.columns) == ['PassengerId', 'Survived']
    assert sub_df.isnull().sum().sum() == 0
    assert set(sub_df['Survived'].unique()).issubset({0, 1})
    print("\nALL CONTRACT & INTEGRITY CHECKS PASSED.")
    return metrics_df, best_ens_acc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_generate_all(base_dir)
