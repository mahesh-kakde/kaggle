import os
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler

def engineer_features(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # drop high leverage outliers
    train_df = train_df[~((train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000))].reset_index(drop=True)
    
    n_train = len(train_df)
    test_ids = test_df['Id'].values
    y_train = np.log1p(train_df['SalePrice'].values)
    
    train_features = train_df.drop(columns=['Id', 'SalePrice'])
    test_features = test_df.drop(columns=['Id'])
    
    df = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
    
    # structural categorical missing values
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        df[col] = df[col].fillna('None')
        
    # structural numeric missing values
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        df[col] = df[col].fillna(0)
        
    # group median imputation
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # frequent category imputation
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
    for col in mode_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    df['Functional'] = df['Functional'].fillna('Typ')
    
    # cast categorical identifiers
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
    df['YrSold'] = df['YrSold'].astype(str)
    
    # ordinal quality mappings
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        df[col] = df[col].map(qual_map).fillna(0).astype(int)
        
    bsmt_exp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exp_map).fillna(0).astype(int)
    
    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map).fillna(0).astype(int)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map).fillna(0).astype(int)
    
    garage_fin_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    df['GarageFinish'] = df['GarageFinish'].map(garage_fin_map).fillna(0).astype(int)
    
    fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    df['Fence'] = df['Fence'].map(fence_map).fillna(0).astype(int)
    
    # domain aggregations
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['HouseAge'] = (df['YrSold'].astype(int) - df['YearBuilt'].astype(int)).clip(lower=0)
    df['RemodelAge'] = (df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)).clip(lower=0)
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['OverallGrade'] = df['OverallQual'] * df['OverallCond']
    df['QualSF'] = df['OverallQual'] * df['TotalSF']
    df['QualGrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['YrBltAndRemod'] = df['YearBuilt'].astype(int) + df['YearRemodAdd'].astype(int)
    
    # transform skewed continuous columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    skewed_feats = df[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
    for feat in high_skew:
        df[feat] = boxcox1p(df[feat], 0.15)
        
    # one-hot encode nominal categories
    df = pd.get_dummies(df, drop_first=True, dtype=float)
    
    # prune near zero variance columns
    low_var = [col for col in df.columns if df[col].nunique() <= 1 or (df[col].dtype == 'float64' and (df[col] != 0).sum() < 2)]
    df = df.drop(columns=low_var)
    
    X_train = df.iloc[:n_train].copy().reset_index(drop=True)
    X_test = df.iloc[n_train:].copy().reset_index(drop=True)
    
    feature_names = list(X_train.columns)
    
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'test_ids': test_ids,
        'feature_names': feature_names,
        'train_df_clean': train_df,
        'test_df_clean': test_df
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = engineer_features(
        os.path.join(base_dir, "data", "train.csv"),
        os.path.join(base_dir, "data", "test.csv")
    )
    print("features processed:", len(data['feature_names']))
    print("train shape:", data['X_train'].shape)
    print("test shape:", data['X_test'].shape)
    print("nulls in train:", data['X_train'].isnull().sum().sum())
    print("nulls in test:", data['X_test'].isnull().sum().sum())