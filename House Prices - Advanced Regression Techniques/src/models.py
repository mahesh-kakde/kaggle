from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def get_candidate_models():
    models = {
        'ridge': Ridge(alpha=14.0, random_state=42),
        'lasso': Lasso(alpha=0.0005, random_state=42, max_iter=3000),
        'elastic': ElasticNet(alpha=0.0005, l1_ratio=0.85, random_state=42, max_iter=3000),
        'svr': SVR(C=25.0, epsilon=0.015, gamma='scale'),
        'k_ridge': KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
        'gbr': GradientBoostingRegressor(n_estimators=1000, learning_rate=0.035, max_depth=3, max_features='sqrt', loss='huber', random_state=42),
        'xgb': XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.464, reg_lambda=0.857, random_state=42),
        'lgb': LGBMRegressor(n_estimators=1000, max_depth=3, num_leaves=5, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.4, reg_lambda=0.8, random_state=42, verbose=-1),
        'cat': CatBoostRegressor(iterations=1000, depth=4, learning_rate=0.03, l2_leaf_reg=3.0, random_seed=42, verbose=False)
    }
    return models