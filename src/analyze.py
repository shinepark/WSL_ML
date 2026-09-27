"""
Question to ask: Did WSL team playing style/quality shift measurably from pre-COVID (18/19)
to COVID (20/21 and no fans) to post COVID (23/24 with the post-Euro 22 investment boom)?

3 layers of analysis:
1. statistical testing to see which tactical/performance metrics actually differ significantly, rather than eyeballing means.

2. PCA + KMeans clustering on standardized features to see whether team seasons naturally group by era without being told the era label

3. Random Forest classifier trained to predict era from tactical features, evalyated with Leave-One-Out cross-validation
and a permutation test for statistical significance of the accuracy score. Feature importances showh which stats actually carry the era signal.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "possession_pct",
    "pass_completion_pct",
    "passes_per_90",
    "progressive_passes_per_90",
    "avg_pass_length",
    "long_ball_pct",
    "pressures_per_90",
    "shots_per_90",
    "xg_per_90",
    "xg_per_shot",
    "dribbles_per_90",
    "carries_per_90",
    "avg_carry_length",
]

ERA_ORDER = ["2018/19", "2020/21", "2023/24"]

sns.set_theme(style="whitegrid", palette="viridis")

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "team_season_features.csv")
    df = df.dropna(subset=FEATURE_COLS)
    return df

def run_anova(df: pd.DataFrame, feature_cols: list = FEATURE_COLS, out_name: str = "anova_results.csv") -> pd.DataFrame:
    """
    One way ANOVA per feature: does the feature differ across the 3 seasons?
    """
    results = []
    for col in feature_cols:
        groups = [df[df["season"] == era][col].dropna() for era in ERA_ORDER]
        f_stat, p_val = stats.f_oneway(*groups)
        means = {f"mean_{era}": df[df["season"] == era][col].mean() for era in ERA_ORDER}
        results.append({"feature": col, "f_stat": f_stat, "p_value": p_val, **means})

    result_df = pd.DataFrame(results).sort_values("p_value")
    result_df.to_csv(OUT_DIR / out_name, index=False)
    return result_df

def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA to 2D + KMeans (k=3) on standardized features.
    Does clustering (blind to era labels) recover the era structure?
    """
    X = StandardScaler().fit_transform(df[FEATURE_COLS])

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = km.fit_predict(X)

    df = df.copy()
    df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]
    df["cluster"] = clusters
    df.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    # cross tab - how well do clusters line up with actual era?
    crosstab = pd.crosstab(df["season"], df["cluster"])
    print("\nCluster vs. era cross-tab:")
    print(crosstab)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.round(3)}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="pca1", y="pca2", hue="season", style="cluster",
        s=140, ax=ax, hue_order=ERA_ORDER,
    )
    ax.set_title("WSL Team Seasons: PCA Projection Colored by Era")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_clusters.png", dpi=150)
    plt.close(fig)

    return df

def run_classifier(df: pd.DataFrame):
    """
    Random Forest predicting era from tactical features.
    n=35, 3 classes
    Leave-One-Out CV and a permutation test to check the accuracy is better than chance
    """

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["season"].astype(str).to_numpy()

    clf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=4)

    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo)
    accuracy = scores.mean()

    # permutation test
    # is accuracy meaningfully above chance (33%)?
    score, perm_scores, p_value = permutation_test_score(
        clf, X, y, cv=5, n_permutations=200, random_state=42
    )

    # fit on full data for feature importances
    # interpretation only not for accuracy
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    report_lines = [
        "WSL Era Classifier (Random Forest, Leave One Out CV)",
        "=" * 55,
        f"n = {len(df)} team seasons across {len(ERA_ORDER)} eras",
        f"Leave One Out accuracy: {accuracy:.3f} (chance level = {1/len(ERA_ORDER):.3f})",
        f"5 fold permutation test score: {score:.3f}, p = {p_value:.4f}",
        "",
        "Top features by importance:",
    ]
    for feat, imp in importances.items():
        report_lines.append(f" {feat:<30} {imp:.3f}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (OUT_DIR / "classifier_report.txt").write_text(report_text)

    # feature importance plot
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(importances)))
    ax.set_title("Feature Importance: Predicting WSL Era from Math Stats")
    ax.set_xlabel("Random Forest Importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return accuracy, p_value, importances

def plot_era_boxplots(df: pd.DataFrame, anova_results: pd.DataFrame):
    """
    grid of boxplots for the features w/ the strongest era effect
    """

    top_feats = anova_results.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flat, top_feats):
        sns.boxplot(data=df, x="season", y=feat, order=ERA_ORDER, ax=ax)
        sns.stripplot(data=df, x="season", y=feat, order=ERA_ORDER, ax=ax, color="black", alpha=0.5, size=4)
        p_val = anova_results.loc[anova_results["feature"] == feat, "p_value"].values[0]
        ax.set_title(f"{feat}\n(ANOVA p={p_val:.3f})")
        ax.set_xlabel("")

    fig.suptitle("WSL Team Level Stats by Era: Pre-COVID vs COVID vs Post-COVID Boom", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "era_comparison_boxplots.png", dpi=150)
    plt.close(fig)

NETWORK_FEATURE_COLS = ["betweenness_mean", "centralization", "density", "clustering"]

def merge_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bring pass_networks.py's per-team-era structural stats onto the main
    feature table, joined on (team, era) rather than (team, season) since
    era is the unambiguous key both files share."""
    net_df = pd.read_csv(DATA_DIR / "network_features.csv")
    merged = df.merge(net_df[["team", "era"] + NETWORK_FEATURE_COLS], on=["team", "era"], how="inner")
    dropped = len(df) - len(merged)
    if dropped:
        print(f"[warn] {dropped} team-era rows had no matching network features and were dropped")
    return merged

def main():
    df = load_data()
    print(f"Loaded {len(df)} team-season rows.")

    print("\n Running ANOVA across eras for each feature...")
    anova_results = run_anova(df)
    print(anova_results[["feature", "f_stat", "p_value"]].to_string(index=False))

    print("\n Running ANOVA on passing-network structure features...")
    network_df = merge_network_features(df)
    network_anova = run_anova(network_df, feature_cols=NETWORK_FEATURE_COLS, out_name="network_anova_results.csv")
    print(network_anova[["feature", "f_stat", "p_value"]].to_string(index=False))

    print("\n Running PCA + KMeans clustering...")
    clustered_df = run_clustering(df)

    print("\n Training Random Forest era classifier...")
    run_classifier(df)

    print("\n Generating era comparison boxplots...")
    plot_era_boxplots(df, anova_results)

if __name__ == "__main__":
    main()