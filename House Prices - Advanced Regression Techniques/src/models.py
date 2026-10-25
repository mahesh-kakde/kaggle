from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def get_candidate_models():
    models = {
        'ridge': Ridge(alpha=15.0, random_state=42),
        'lasso': Lasso(alpha=0.0005, random_state=42, max_iter=2000),
        'elastic': ElasticNet(alpha=0.0005, l1_ratio=0.85, random_state=42, max_iter=2000),
        'k_ridge': KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
        'gbr': GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, loss='huber', random_state=42),
        'xgb': XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42),
        'lgb': LGBMRegressor(n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1),
        'cat': CatBoostRegressor(iterations=500, depth=4, learning_rate=0.04, l2_leaf_reg=3.0, random_seed=42, verbose=False)
    }
    return models