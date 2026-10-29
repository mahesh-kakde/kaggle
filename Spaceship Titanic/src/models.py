from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_models(random_state=42):
    return {
        "catboost": CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=0
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=700,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        ),
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=random_state
        )
    }