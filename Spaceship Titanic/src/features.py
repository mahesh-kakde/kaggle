import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def assign_cabin_zone(num):
    if num < 300:
        return "Z0"
    elif num < 500:
        return "Z1"
    elif num < 700:
        return "Z2"
    elif num < 850:
        return "Z3"
    elif num < 1150:
        return "Z4"
    elif num < 1300:
        return "Z5"
    elif num < 1550:
        return "Z6"
    elif num < 1720:
        return "Z7"
    else:
        return "Z8"


def extract_features(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_ids = test["PassengerId"].copy()

    full = pd.concat(
        [train.assign(is_train=1), test.assign(is_train=0, Transported=np.nan)],
        ignore_index=True
    )

    # passenger travel group structure
    full["GroupId"] = full["PassengerId"].str.split("_").str[0]
    full["GroupNum"] = full["PassengerId"].str.split("_").str[1].astype(int)
    full["GroupSize"] = full.groupby("GroupId")["PassengerId"].transform("count")
    full["IsAlone"] = (full["GroupSize"] == 1).astype(int)

    # passenger family surname structure
    full["Surname"] = full["Name"].str.split().str[-1]
    full["FamilySize"] = full.groupby("Surname")["Surname"].transform("count").fillna(1).astype(int)

    # cabin deck, number, and side
    full["CabinDeck"] = full["Cabin"].str.split("/").str[0]
    full["CabinNum"] = pd.to_numeric(full["Cabin"].str.split("/").str[1], errors="coerce")
    full["CabinSide"] = full["Cabin"].str.split("/").str[2]

    # infer missing cabin side and deck from group members
    group_side = full.groupby("GroupId")["CabinSide"].first().dropna()
    full["CabinSide"] = full["CabinSide"].fillna(full["GroupId"].map(group_side))

    group_deck = full.groupby("GroupId")["CabinDeck"].first().dropna()
    full["CabinDeck"] = full["CabinDeck"].fillna(full["GroupId"].map(group_deck))

    # infer home planet from decks and travel groups
    full.loc[full["HomePlanet"].isna() & full["CabinDeck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
    full.loc[full["HomePlanet"].isna() & (full["CabinDeck"] == "G"), "HomePlanet"] = "Earth"

    group_planet = full.groupby("GroupId")["HomePlanet"].first().dropna()
    full["HomePlanet"] = full["HomePlanet"].fillna(full["GroupId"].map(group_planet))

    surname_planet = full.groupby("Surname")["HomePlanet"].first().dropna()
    full["HomePlanet"] = full["HomePlanet"].fillna(full["Surname"].map(surname_planet))

    # infer luxury amenities and cryosleep status
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spend_sum = full[amenities].sum(axis=1)
    full.loc[full["CryoSleep"].isna() & (spend_sum > 0), "CryoSleep"] = False

    for col in amenities:
        full.loc[(full["CryoSleep"] == True) & full[col].isna(), col] = 0.0
        full.loc[(full["Age"] < 13) & full[col].isna(), col] = 0.0

    # infer vip status
    full.loc[full["Age"] < 18, "VIP"] = False
    full.loc[full["HomePlanet"] == "Earth", "VIP"] = False

    # zero-fill remaining luxury amenities
    for col in amenities:
        full[col] = full[col].fillna(0.0)

    full["TotalSpend"] = full[amenities].sum(axis=1)
    full["HasSpent"] = (full["TotalSpend"] > 0).astype(int)

    # solitary vs communal amenity expenditures
    full["SolitarySpend"] = full["RoomService"] + full["Spa"] + full["VRDeck"]
    full["CommunalSpend"] = full["FoodCourt"] + full["ShoppingMall"]
    full["SolitaryDiff"] = full["CommunalSpend"] - full["SolitarySpend"]

    # impute remaining categoricals and numerics
    full["HomePlanet"] = full["HomePlanet"].fillna("Missing")
    full["CryoSleep"] = full["CryoSleep"].fillna(False).astype(int)
    full["Destination"] = full["Destination"].fillna("Missing")
    full["VIP"] = full["VIP"].fillna(False).astype(int)
    full["CabinDeck"] = full["CabinDeck"].fillna("Missing")
    full["CabinSide"] = full["CabinSide"].fillna("Missing")
    full["Deck_Side"] = full["CabinDeck"] + "_" + full["CabinSide"]

    full["Age"] = full["Age"].fillna(full["Age"].median())
    full["CabinNum"] = full["CabinNum"].fillna(full["CabinNum"].median())

    # spatial breach zones along longitudinal axis
    full["CabinZone"] = full["CabinNum"].apply(assign_cabin_zone)
    full["Side_Zone"] = full["CabinSide"] + "_" + full["CabinZone"]
    full["Cryo_ZeroSpend"] = ((full["CryoSleep"] == 0) & (full["TotalSpend"] == 0)).astype(int)

    train_mask = full["is_train"] == 1
    test_mask = full["is_train"] == 0

    feature_cols = [
        "HomePlanet", "CryoSleep", "CabinDeck", "CabinNum", "CabinSide", "Deck_Side",
        "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "TotalSpend", "HasSpent", "SolitarySpend", "CommunalSpend",
        "SolitaryDiff", "GroupNum", "GroupSize", "IsAlone", "FamilySize",
        "Cryo_ZeroSpend", "CabinZone", "Side_Zone"
    ]

    cat_cols = ["HomePlanet", "CabinDeck", "CabinSide", "Deck_Side", "Destination", "CabinZone", "Side_Zone"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train_df = full.loc[train_mask, feature_cols].copy().reset_index(drop=True)
    y_train = full.loc[train_mask, "Transported"].astype(int).values
    X_test_df = full.loc[test_mask, feature_cols].copy().reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_train = encoder.fit_transform(X_train_df[cat_cols])
    ohe_test = encoder.transform(X_test_df[cat_cols])
    ohe_feature_names = list(encoder.get_feature_names_out(cat_cols))

    X_train_encoded = pd.concat(
        [X_train_df[num_cols].reset_index(drop=True), pd.DataFrame(ohe_train, columns=ohe_feature_names)],
        axis=1
    )
    X_test_encoded = pd.concat(
        [X_test_df[num_cols].reset_index(drop=True), pd.DataFrame(ohe_test, columns=ohe_feature_names)],
        axis=1
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_encoded),
        columns=X_train_encoded.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_encoded),
        columns=X_test_encoded.columns
    )

    return {
        "raw_train": X_train_df,
        "raw_test": X_test_df,
        "encoded_train": X_train_encoded,
        "encoded_test": X_test_encoded,
        "scaled_train": X_train_scaled,
        "scaled_test": X_test_scaled,
        "y_train": y_train,
        "test_ids": test_ids,
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }