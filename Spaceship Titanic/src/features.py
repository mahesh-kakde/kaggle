import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    for col in amenities:
        full[f"log_{col}"] = np.log1p(full[col])
        full[f"ratio_{col}"] = full[col] / (full["TotalSpend"] + 1.0)
    full["log_TotalSpend"] = np.log1p(full["TotalSpend"])

    # solitary vs communal amenity expenditures
    full["SolitarySpend"] = full["RoomService"] + full["Spa"] + full["VRDeck"]
    full["CommunalSpend"] = full["FoodCourt"] + full["ShoppingMall"]
    full["log_SolitarySpend"] = np.log1p(full["SolitarySpend"])
    full["log_CommunalSpend"] = np.log1p(full["CommunalSpend"])
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
    full["IsChild"] = (full["Age"] < 13).astype(int)
    full["IsTeen"] = ((full["Age"] >= 13) & (full["Age"] < 18)).astype(int)
    full["IsAdult"] = (full["Age"] >= 18).astype(int)

    full["CabinNum"] = full["CabinNum"].fillna(full["CabinNum"].median())
    deck_max = full.groupby("CabinDeck")["CabinNum"].transform("max")
    full["RelativeDeckNum"] = full["CabinNum"] / (deck_max + 1)
    full["CabinRegion"] = (full["CabinNum"] // 300).astype(int)
    full["Cryo_ZeroSpend"] = ((full["CryoSleep"] == 0) & (full["TotalSpend"] == 0)).astype(int)

    # group contextual features
    full["Group_Cryo_Ratio"] = full.groupby("GroupId")["CryoSleep"].transform("mean")
    full["Group_TotalSpend"] = full.groupby("GroupId")["TotalSpend"].transform("sum")
    full["Group_MeanSpend"] = full.groupby("GroupId")["TotalSpend"].transform("mean")
    full["Group_SolitarySpend"] = full.groupby("GroupId")["SolitarySpend"].transform("sum")
    full["Group_HasChild"] = full.groupby("GroupId")["IsChild"].transform("max")

    # spatial and planetary interactions
    full["Planet_Dest"] = full["HomePlanet"] + "_" + full["Destination"]
    full["Deck_Side_Region"] = full["CabinDeck"] + "_" + full["CabinSide"] + "_" + full["CabinRegion"].astype(str)

    train_mask = full["is_train"] == 1
    test_mask = full["is_train"] == 0

    feature_cols = [
        "HomePlanet", "CryoSleep", "CabinDeck", "CabinNum", "CabinSide", "Deck_Side",
        "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "TotalSpend", "HasSpent", "log_RoomService", "log_FoodCourt",
        "log_ShoppingMall", "log_Spa", "log_VRDeck", "ratio_RoomService",
        "ratio_FoodCourt", "ratio_ShoppingMall", "ratio_Spa", "ratio_VRDeck",
        "log_TotalSpend", "SolitarySpend", "CommunalSpend", "log_SolitarySpend",
        "log_CommunalSpend", "SolitaryDiff", "CabinRegion", "Cryo_ZeroSpend",
        "GroupNum", "GroupSize", "IsAlone", "FamilySize",
        "IsChild", "IsTeen", "IsAdult", "RelativeDeckNum",
        "Group_Cryo_Ratio", "Group_TotalSpend", "Group_MeanSpend", "Group_SolitarySpend", "Group_HasChild",
        "Planet_Dest", "Deck_Side_Region"
    ]

    cat_cols = ["HomePlanet", "CabinDeck", "CabinSide", "Deck_Side", "Destination", "Planet_Dest", "Deck_Side_Region"]
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