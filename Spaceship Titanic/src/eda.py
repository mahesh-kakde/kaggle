import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_eda(data_path="data/train.csv", output_path="reports/eda_distributions.png"):
    df = pd.read_csv(data_path)

    df["Deck"] = df["Cabin"].apply(lambda x: str(x).split("/")[0] if pd.notna(x) else "Missing")
    df["Side"] = df["Cabin"].apply(lambda x: str(x).split("/")[2] if pd.notna(x) else "Missing")
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[amenities].sum(axis=1)
    df["HasSpent"] = df["TotalSpend"] > 0

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sns.countplot(
        ax=axes[0, 0],
        data=df,
        x="Transported",
        hue="Transported",
        legend=False,
        palette=["#e74c3c", "#2ecc71"]
    )
    axes[0, 0].set_title("Target Distribution (Transported)")
    axes[0, 0].set_xlabel("Transported")
    axes[0, 0].set_ylabel("Count")

    cryo_rate = df.groupby("CryoSleep", dropna=False)["Transported"].mean().reset_index()
    cryo_rate["CryoSleep"] = cryo_rate["CryoSleep"].astype(str)
    sns.barplot(
        ax=axes[0, 1],
        data=cryo_rate,
        x="CryoSleep",
        y="Transported",
        hue="CryoSleep",
        legend=False,
        palette="viridis"
    )
    axes[0, 1].set_title("Transported Rate by CryoSleep Status")
    axes[0, 1].set_ylabel("Transportation Rate")
    axes[0, 1].set_ylim(0, 1)

    spend_rate = df.groupby("HasSpent")["Transported"].mean().reset_index()
    sns.barplot(
        ax=axes[0, 2],
        data=spend_rate,
        x="HasSpent",
        y="Transported",
        hue="HasSpent",
        legend=False,
        palette="crest"
    )
    axes[0, 2].set_title("Transported Rate: Zero vs Non-Zero Luxury Spend")
    axes[0, 2].set_xlabel("Has Any Spend (> 0)")
    axes[0, 2].set_ylabel("Transportation Rate")
    axes[0, 2].set_ylim(0, 1)

    deck_order = ["A", "B", "C", "D", "E", "F", "G", "T", "Missing"]
    deck_rate = df.groupby("Deck", dropna=False)["Transported"].mean().reindex(deck_order).reset_index()
    sns.barplot(
        ax=axes[1, 0],
        data=deck_rate,
        x="Deck",
        y="Transported",
        hue="Deck",
        legend=False,
        palette="magma"
    )
    axes[1, 0].set_title("Transported Rate by Cabin Deck")
    axes[1, 0].set_ylabel("Transportation Rate")
    axes[1, 0].set_ylim(0, 1)

    side_rate = df.groupby("Side", dropna=False)["Transported"].mean().reset_index()
    sns.barplot(
        ax=axes[1, 1],
        data=side_rate,
        x="Side",
        y="Transported",
        hue="Side",
        legend=False,
        palette="mako"
    )
    axes[1, 1].set_title("Transported Rate by Cabin Side (Port / Starboard)")
    axes[1, 1].set_ylabel("Transportation Rate")
    axes[1, 1].set_ylim(0, 1)

    planet_dest = df.groupby(["HomePlanet", "Destination"])["Transported"].mean().reset_index()
    sns.barplot(
        ax=axes[1, 2],
        data=planet_dest,
        x="HomePlanet",
        y="Transported",
        hue="Destination",
        palette="Set2"
    )
    axes[1, 2].set_title("Transported Rate by Origin and Destination")
    axes[1, 2].set_ylabel("Transportation Rate")
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_eda()