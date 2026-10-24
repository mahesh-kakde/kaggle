"""Comprehensive Model Configurations and Ensembling Factory for Titanic.

Includes:
1. Logistic Regression (L2 regularized baseline)
2. Calibrated SVC (Support Vector Classifier with RBF kernel)
3. Random Forest (Bagged decision trees)
4. Extra Trees (Extremely randomized trees)
5. XGBoost (Tuned gradient boosting)
6. LightGBM (Histogram gradient boosting)
7. CatBoost (Oblivious/symmetric gradient boosting)
8. Stacking Classifier & Soft-Voting Ensemble
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_candidate_models():
    """Return dictionary of 7 tuned candidate models."""
    models = {
        # 1. Regularized Logistic Regression (Andrew Ng Course 1)
        'LogisticRegression': LogisticRegression(
            C=0.3,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ),
        
        # 2. Support Vector Classifier with RBF Kernel (Calibrated)
        'CalibratedSVC': CalibratedClassifierCV(
            SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42),
            ensemble=False
        ),
        
        # 3. Random Forest (Bagging - Andrew Ng Course 2 Week 4)
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        
        # 4. Extra Trees
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        
        # 5. XGBoost (Hyperparameter Tuned)
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='logloss'
        ),
        
        # 6. LightGBM (Regularized Leaf-wise)
        'LightGBM': LGBMClassifier(
            n_estimators=150,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        ),
        
        # 7. CatBoost (Symmetric/Oblivious Trees)
        'CatBoost': CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False
        )
    }
    return models
