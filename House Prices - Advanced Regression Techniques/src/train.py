import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

from features import engineer_features
from models import get_candidate_models
from diagnostics import (
    plot_model_comparison, plot_learning_curves,
    plot_residuals_and_qq, plot_predicted_vs_actual,
    plot_feature_importances
)

def evaluate_models_oof(models: dict, X_scaled_tr: pd.DataFrame, X_raw_tr: pd.DataFrame,
                        X_scaled_te: pd.DataFrame, X_raw_te: pd.DataFrame, y: np.ndarray, cv):
    oof_predictions = {}
    test_predictions = {}
    metrics_dict = {}
    
    linear_family = {'ridge', 'lasso', 'elastic', 'svr', 'k_ridge'}
    n_splits = cv.get_n_splits()
    
    for name, model in models.items():
        inp_tr = X_scaled_tr.values if name in linear_family else X_raw_tr.values
        inp_te = X_scaled_te.values if name in linear_family else X_raw_te.values
        
        oof_pred = np.zeros(len(y))
        test_pred = np.zeros(len(inp_te))
        
        for tr, va in cv.split(inp_tr, y):
            m = clone(model)
            m.fit(inp_tr[tr], y[tr])
            oof_pred[va] = m.predict(inp_tr[va])
            test_pred += m.predict(inp_te) / n_splits
            
        rmse = root_mean_squared_error(y, oof_pred)
        oof_predictions[name] = oof_pred
        test_predictions[name] = test_pred
        metrics_dict[name] = rmse
        print(f"model: {name:<12} | 5-fold log-rmse: {rmse:.4f}")
        
    return oof_predictions, test_predictions, metrics_dict

def train_and_generate_all(base_dir: str):
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    submissions_dir = os.path.join(base_dir, "submissions")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    
    # 1. load and process features
    data = engineer_features(
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "test.csv")
    )
    X_train = data['X_train']
    X_train_scaled = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    test_ids = data['test_ids']
    feature_names = data['feature_names']
    
    print(f"features: {len(feature_names)}, train samples: {len(y_train)}, test samples: {len(test_ids)}")
    
    # 2. cross validation and fold inference
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models()
    
    oof_preds, test_preds, metrics_dict = evaluate_models_oof(
        models, X_train_scaled, X_train, X_test_scaled, X_test, y_train, cv
    )
    
    # 3. optimize ensemble weights with diversity bounds
    ensemble_models = ['ridge', 'lasso', 'elastic', 'svr', 'k_ridge', 'gbr', 'xgb', 'lgb', 'cat']
    oof_matrix = np.column_stack([oof_preds[k] for k in ensemble_models])
    test_matrix = np.column_stack([test_preds[k] for k in ensemble_models])
    
    def loss_func(w):
        return root_mean_squared_error(y_train, np.dot(oof_matrix, w))
        
    init_weights = np.ones(len(ensemble_models)) / len(ensemble_models)
    bounds = [(0.04, 0.30)] * len(ensemble_models)
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1.0}
    opt_res = minimize(loss_func, init_weights, bounds=bounds, constraints=constraints)
    
    best_weights = opt_res.x
    blended_oof = np.dot(oof_matrix, best_weights)
    ensemble_rmse = root_mean_squared_error(y_train, blended_oof)
    
    weight_dict = {m: round(float(w), 3) for m, w in zip(ensemble_models, best_weights)}
    print(f"\nensemble weights: {weight_dict}")
    print(f"blended ensemble log-rmse: {ensemble_rmse:.4f}")
    
    metrics_dict['ensemble'] = ensemble_rmse
    
    # 4. generate diagnostic plots
    plot_model_comparison(metrics_dict, os.path.join(reports_dir, "model_comparison.png"))
    
    diag_models = {
        'ridge': models['ridge'],
        'svr': models['svr'],
        'xgb': models['xgb'],
        'cat': models['cat']
    }
    plot_learning_curves(diag_models, X_train_scaled.values, y_train, cv, os.path.join(reports_dir, "diagnostic_learning_curves.png"))
    plot_residuals_and_qq(y_train, blended_oof, os.path.join(reports_dir, "residuals_and_qq.png"))
    plot_predicted_vs_actual(y_train, blended_oof, os.path.join(reports_dir, "predicted_vs_actual.png"))
    
    # extract feature importances
    xgb_m = models['xgb']
    xgb_m.fit(X_train.values, y_train)
    xgb_imp = pd.Series(xgb_m.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    cat_m = models['cat']
    cat_m.fit(X_train.values, y_train)
    cat_imp = pd.Series(cat_m.get_feature_importance(), index=feature_names).sort_values(ascending=False)
    
    gbr_m = models['gbr']
    gbr_m.fit(X_train.values, y_train)
    gbr_imp = pd.Series(gbr_m.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    plot_feature_importances([xgb_imp, cat_imp, gbr_imp], ['xgb', 'cat', 'gbr'], os.path.join(reports_dir, "feature_importances.png"))
    
    # 5. blend predictions
    final_test_log = np.dot(test_matrix, best_weights)
    final_prices = np.expm1(final_test_log)
    
    # 6. export submission
    submission_path = os.path.join(submissions_dir, "submission.csv")
    sub_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': final_prices
    })
    sub_df.to_csv(submission_path, index=False)
    
    print(f"\nsaved submission to: {submission_path}")
    print(f"shape: {sub_df.shape} | nulls: {sub_df.isnull().sum().to_dict()}")
    print("summary stats of predicted prices:")
    print(sub_df['SalePrice'].describe())
    
    assert sub_df.shape == (1459, 2)
    assert list(sub_df.columns) == ['Id', 'SalePrice']
    assert sub_df.isnull().sum().sum() == 0
    assert (sub_df['SalePrice'] > 0).all()
    print("submission contract checks passed.")
    return ensemble_rmse

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_generate_all(base_dir)