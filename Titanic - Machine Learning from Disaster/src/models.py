"""Model configurations and factory for Titanic: Machine Learning from Disaster.

Following Andrew Ng's progressive modeling workflow:
1. Baseline: Logistic Regression with tuned L2 regularization.
2. Random Forest: Bagged trees with feature subsampling (reduces variance).
3. Extra Trees: Extremely randomized trees (further variance reduction).
4. XGBoost: Gradient boosted trees with shrinkage and regularization (reduces bias).
5. LightGBM: Efficient histogram-based boosting with depth and leaf regularization.
6. Soft-Voting Ensemble: Ensembling diverse high-performing models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_candidate_models():
    """Return a dictionary of tuned candidate models."""
    models = {
        # 1. Andrew Ng Baseline: Regularized Logistic Regression
        'LogisticRegression': LogisticRegression(
            C=0.5,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ),
        
        # 2. Bagging: Random Forest with depth constraint and min samples leaf
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        ),
        
        # 3. Extra Trees: Extremely Randomized Trees
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        ),
        
        # 4. XGBoost: Regularized Gradient Boosting
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
        
        # 5. LightGBM: Fast histogram gradient boosting
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

def build_voting_ensemble(models: dict):
    """Build a soft-voting classifier combining the top models."""
    estimators = [
        ('lr', models['LogisticRegression']),
        ('rf', models['RandomForest']),
        ('et', models['ExtraTrees']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
    ]
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[1, 2, 1, 2, 2]
    )
    return voting_clf
