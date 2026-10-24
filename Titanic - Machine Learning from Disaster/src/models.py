"""Model configurations and factory for Titanic: Machine Learning from Disaster.

Following Andrew Ng's progressive modeling workflow:
1. Baseline: Logistic Regression with tuned L2 regularization.
2. Random Forest: Bagged trees with depth constraint and min samples leaf.
3. Extra Trees: Extremely randomized trees (reduces variance).
4. XGBoost: Regularized gradient boosted decision trees.
5. LightGBM: Efficient leaf-wise histogram boosting.
6. Soft-Voting Ensemble: High-diversity weighted probability blend.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_candidate_models():
    """Return dictionary of regularized candidate models."""
    models = {
        # 1. Regularized Logistic Regression (Andrew Ng Course 1)
        'LogisticRegression': LogisticRegression(
            C=0.5,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ),
        
        # 2. Random Forest (Bagging - Andrew Ng Course 2 Week 4)
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        ),
        
        # 3. Extra Trees (Extreme Randomization)
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        ),
        
        # 4. XGBoost (Regularized Boosting)
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        ),
        
        # 5. LightGBM (Histogram Boosting)
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
        )
    }
    return models
