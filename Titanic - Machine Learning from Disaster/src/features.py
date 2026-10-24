"""Feature engineering module for Titanic: Machine Learning from Disaster.

Follows Andrew Ng's principles:
1. Grounded intuition: Extract domain-rich features (Title, Family Dynamics, Ticket Groups, Cabin Decks).
2. Clean preprocessing: No leakage, systematic imputations based on conditional medians.
3. Model-ready outputs: Prepares both scaled features (for regularized Logistic Regression)
   and unscaled/tree-friendly features (for Random Forest, ExtraTrees, XGBoost, LightGBM).
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_title(name: str) -> str:
    """Extract and standardize title from passenger name."""
    match = re.search(r',\s*([^\.]+)\.', name)
    if not match:
        return 'Rare'
    title = match.group(1).strip()
    
    # Standardize French / aristocratic titles
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 'Miss'
    elif title in ['Mrs', 'Mme', 'Lady', 'the Countess', 'Dona', 'Countess']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        # Col, Major, Rev, Dr, Capt, Jonkheer, Don, Sir
        return 'Rare'

def engineer_features(train_path: str, test_path: str):
    """Load and engineer features for train and test sets without data leakage."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    n_train = len(train_df)
    test_passenger_ids = test_df['PassengerId'].values
    y_train = train_df['Survived'].values.astype(int)
    
    # Combine temporarily for consistent group/ticket calculations and encodings
    combined = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    
    # 1. Title Extraction
    combined['Title'] = combined['Name'].apply(extract_title)
    
    # 2. Impute missing Embarked (mode is 'S')
    combined['Embarked'] = combined['Embarked'].fillna('S')
    
    # 3. Family Size & Composition
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    combined['IsSmallFamily'] = ((combined['FamilySize'] >= 2) & (combined['FamilySize'] <= 4)).astype(int)
    combined['IsLargeFamily'] = (combined['FamilySize'] >= 5).astype(int)
    
    # 4. Ticket Group Size and Fare Per Person
    ticket_counts = combined['Ticket'].value_counts()
    combined['TicketGroupSize'] = combined['Ticket'].map(ticket_counts)
    
    # Missing Fare in test: impute with median for Pclass 3 & Embarked S
    fare_median_p3_s = combined[(combined['Pclass'] == 3) & (combined['Embarked'] == 'S')]['Fare'].median()
    combined['Fare'] = combined['Fare'].fillna(fare_median_p3_s)
    
    # Adjust Fare per Person (since group tickets bundle fares together)
    combined['FarePerPerson'] = combined['Fare'] / combined['TicketGroupSize']
    combined['LogFare'] = np.log1p(combined['FarePerPerson'])
    
    # 5. Cabin & Deck
    combined['HasCabin'] = combined['Cabin'].notnull().astype(int)
    combined['Deck'] = combined['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
    # Merge rare 'T' deck with 'A'
    combined['Deck'] = combined['Deck'].replace({'T': 'A'})
    
    # 6. Age Imputation using (Title, Pclass) conditional medians
    age_medians = combined.groupby(['Title', 'Pclass'])['Age'].transform('median')
    combined['Age'] = combined['Age'].fillna(age_medians)
    # If any remaining NaN, fallback to title median
    combined['Age'] = combined['Age'].fillna(combined.groupby('Title')['Age'].transform('median'))
    
    # Child & Mother flags (crucial historical survival rules)
    combined['IsChild'] = (combined['Age'] < 14).astype(int)
    combined['IsMother'] = ((combined['Sex'] == 'female') & (combined['Parch'] > 0) & (combined['Age'] > 18) & (combined['Title'] == 'Mrs')).astype(int)
    
    # 7. Family / Group Survival Signal (Woman-Child-Group heuristic)
    combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip())
    combined['FamilySurvival'] = 0.5  # Neutral default
    
    # Map survival across Ticket and Surname groups
    for grp_key in ['Ticket', 'Surname']:
        for _, grp in combined.groupby(grp_key):
            if len(grp) > 1:
                for idx in grp.index:
                    other_members = grp.loc[grp.index != idx, 'Survived'].dropna()
                    if len(other_members) > 0:
                        if other_members.max() == 1.0:
                            combined.loc[idx, 'FamilySurvival'] = 1.0
                        elif other_members.min() == 0.0:
                            combined.loc[idx, 'FamilySurvival'] = 0.0
                            
    # 8. Categorical Encodings
    combined['Sex_Code'] = (combined['Sex'] == 'female').astype(int)
    
    # One-hot encode Title, Embarked, Deck
    categorical_cols = ['Title', 'Embarked', 'Deck', 'Pclass']
    encoded_df = pd.get_dummies(combined[categorical_cols], drop_first=True, dtype=int)
    
    # Numeric features
    numeric_features = [
        'Sex_Code', 'Age', 'SibSp', 'Parch', 'FamilySize', 'IsAlone',
        'IsSmallFamily', 'IsLargeFamily', 'TicketGroupSize', 'FarePerPerson',
        'LogFare', 'HasCabin', 'IsChild', 'IsMother', 'FamilySurvival'
    ]
    
    # Final feature matrix
    X_all = pd.concat([combined[numeric_features], encoded_df], axis=1)
    
    X_train = X_all.iloc[:n_train].copy().reset_index(drop=True)
    X_test = X_all.iloc[n_train:].copy().reset_index(drop=True)
    
    feature_names = list(X_train.columns)
    
    # Scaled version for Logistic Regression & linear models
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'test_passenger_ids': test_passenger_ids,
        'feature_names': feature_names,
        'train_df_raw': train_df,
        'test_df_raw': test_df
    }

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = engineer_features(
        os.path.join(base_dir, "data", "train.csv"),
        os.path.join(base_dir, "data", "test.csv")
    )
    print(f"Engineered features successfully.")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Features ({len(data['feature_names'])}): {data['feature_names']}")
