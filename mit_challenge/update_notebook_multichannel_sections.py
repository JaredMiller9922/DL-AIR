import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent / "DataAnalysis.ipynb"


def lines(text: str):
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def md(cell_id: str, text: str):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines(text)}


def code(cell_id: str, text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


NEW_IDS = {
    "multichannel-unsup-intro",
    "multichannel-unsup-load",
    "multichannel-unsup-methods-md",
    "multichannel-unsup-exploration",
    "multichannel-unsup-pca-md",
    "multichannel-unsup-pca",
    "multichannel-unsup-linear-latents",
    "multichannel-unsup-manifold-md",
    "multichannel-unsup-manifolds",
    "multichannel-unsup-density-clustering",
    "multichannel-unsup-interpretation-md",
    "multichannel-separator-intro",
    "multichannel-separator-merge",
    "multichannel-separator-plots",
    "multichannel-separator-prediction-md",
    "multichannel-separator-models",
    "multichannel-separator-regime-analysis",
    "multichannel-separator-interpretation-md",
}


CELLS = [
    md(
        "multichannel-unsup-intro",
        r"""
# Part F: Unsupervised Learning On MIT Multichannel Mixtures

This section shifts attention from separator outputs back to the **actual multichannel MIT mixtures themselves**. The goal is to understand whether the four-channel mixture geometry contains latent low-dimensional structure that tracks difficulty, recovery success, and separator behavior.

The working hypothesis is that the dominant variance in these short multichannel RF frames is not purely random. Instead, it may reflect a combination of amplitude balance, spatial covariance geometry, inter-channel coherence, phase alignment, and spectral concentration. If that is true, then unsupervised structure in the mixture space should help explain why some frames are easy to separate and decode while others are persistently difficult.
""",
    ),
    code(
        "multichannel-unsup-load",
        r"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import FactorAnalysis, FastICA as SkFastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:
    umap = None

challenge_root = Path(r"D:\CS 6955\CS-6955\DL-AIR\mit_challenge")
analysis_output_root = challenge_root / "analysis_outputs"
figure_output_root = analysis_output_root / "figures"
analysis_output_root.mkdir(parents=True, exist_ok=True)
figure_output_root.mkdir(parents=True, exist_ok=True)

merge_keys = ["alphaIndex", "frameLen", "setIndex", "frame_number"]
expected_separator_names = ["FastICA", "Learned: Hybrid", "Learned: LSTM", "Learned: Linear", "Learned: IQ_CNN", "Learned: HTDemucs"]


def feature_family(name):
    name = str(name)
    if any(token in name for token in ["amp", "power", "papr", "radius", "iq_kurtosis", "iq_skewness"]):
        return "amplitude"
    if any(token in name for token in ["cov_", "eig", "condition", "participation"]):
        return "covariance"
    if any(token in name for token in ["coherence", "offdiag"]):
        return "coherence"
    if "phase" in name:
        return "phase"
    if "spec_" in name:
        return "spectral"
    if name in merge_keys or name in {"source_path", "separator", "source_file"}:
        return "metadata"
    return "other"


def flatten_columns(frame):
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in col if str(part) != "")
        for col in frame.columns.to_flat_index()
    ]
    return frame


def normalize_key_columns(table):
    table = table.copy()
    for col in merge_keys + ["success", "best_output_index", "output_index"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").astype("Int64")
    return table


def load_debug_tables(debug_sources):
    frame_tables = []
    output_tables = []
    summary_tables = []
    for separator, debug_dir in debug_sources:
        if not debug_dir.exists():
            continue
        per_frame_files = sorted(debug_dir.glob("*_perFrame.csv"))
        per_output_files = sorted(debug_dir.glob("*_perOutput.csv"))
        if per_frame_files:
            per_frames = []
            for path in per_frame_files:
                table = pd.read_csv(path)
                table["separator"] = separator
                table["source_file"] = path.name
                per_frames.append(table)
            frame_tables.append(normalize_key_columns(pd.concat(per_frames, ignore_index=True)))
        if per_output_files:
            per_outputs = []
            for path in per_output_files:
                table = pd.read_csv(path)
                table["separator"] = separator
                table["source_file"] = path.name
                per_outputs.append(table)
            output_tables.append(normalize_key_columns(pd.concat(per_outputs, ignore_index=True)))
        summary_path = debug_dir / "debug_run_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary["separator"] = separator
            summary_tables.append(summary)
    frame_df = pd.concat(frame_tables, ignore_index=True) if frame_tables else pd.DataFrame()
    output_df = pd.concat(output_tables, ignore_index=True) if output_tables else pd.DataFrame()
    summary_df = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    return frame_df, output_df, summary_df


def save_table(frame, filename):
    path = analysis_output_root / filename
    frame.to_csv(path, index=False)
    print(f"Saved table to {path}")
    return path


def save_figure(fig, filename):
    path = figure_output_root / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {path}")
    return path


def numeric_feature_columns(frame, extra_exclude=None, min_non_null=0.98):
    exclude = set(extra_exclude or [])
    cols = []
    for col in frame.select_dtypes(include=[np.number]).columns:
        if col in exclude:
            continue
        if frame[col].notna().mean() >= min_non_null:
            cols.append(col)
    return cols


def top_feature_list(frame, feature_cols, n=18):
    var = frame[feature_cols].replace([np.inf, -np.inf], np.nan).var(numeric_only=True).sort_values(ascending=False)
    return list(var.head(n).index)


extended_feature_path = challenge_root / "separation_frame_features_extended.csv"
fallback_feature_path = challenge_root / "separation_frame_features.csv"
feature_path_in_use = extended_feature_path if extended_feature_path.exists() else fallback_feature_path
mit_feature_table = pd.read_csv(feature_path_in_use)
mit_feature_table = normalize_key_columns(mit_feature_table)

separator_sources = [("FastICA", challenge_root / "debugEval")] + [
    (path.name.replace("debugEval_learned_", "Learned: "), path)
    for path in sorted(challenge_root.glob("debugEval_learned_*"))
]
separator_frame_long_df, separator_output_long_df, separator_summary_df = load_debug_tables(separator_sources)

available_separators = sorted(separator_frame_long_df["separator"].dropna().unique()) if not separator_frame_long_df.empty else []
missing_expected_separators = [name for name in expected_separator_names if name not in available_separators]

fastica_frame_labels_df = (
    separator_frame_long_df[separator_frame_long_df["separator"] == "FastICA"].copy()
    if not separator_frame_long_df.empty else pd.DataFrame()
)
fastica_keep_cols = merge_keys + [col for col in ["best_output_index", "best_num_errors", "success", "separator", "source_file"] if col in fastica_frame_labels_df.columns]
fastica_frame_labels_df = fastica_frame_labels_df[fastica_keep_cols].drop_duplicates(subset=merge_keys) if not fastica_frame_labels_df.empty else pd.DataFrame()

fastica_feature_df = mit_feature_table.merge(
    fastica_frame_labels_df.drop(columns=["separator"], errors="ignore"),
    on=merge_keys,
    how="left",
)
separator_feature_long_df = separator_frame_long_df.merge(mit_feature_table, on=merge_keys, how="left") if not separator_frame_long_df.empty else pd.DataFrame()
separator_output_feature_long_df = separator_output_long_df.merge(mit_feature_table, on=merge_keys, how="left") if not separator_output_long_df.empty else pd.DataFrame()

inventory_rows = []
for col in mit_feature_table.columns:
    series = mit_feature_table[col]
    inventory_rows.append({
        "feature": col,
        "family": feature_family(col),
        "dtype": str(series.dtype),
        "non_null_fraction": float(series.notna().mean()),
        "n_unique": int(series.nunique(dropna=True)),
        "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else np.nan,
        "near_constant": bool(series.nunique(dropna=True) <= 2 or (pd.api.types.is_numeric_dtype(series) and float(series.std()) < 1e-8)),
    })
feature_inventory_df = pd.DataFrame(inventory_rows).sort_values(["family", "near_constant", "non_null_fraction", "feature"], ascending=[True, True, False, True]).reset_index(drop=True)

save_table(feature_inventory_df, "mit_multichannel_feature_inventory.csv")

missingness_df = (
    feature_inventory_df[["feature", "family", "non_null_fraction"]]
    .assign(missing_fraction=lambda d: 1.0 - d["non_null_fraction"])
    .sort_values(["missing_fraction", "feature"], ascending=[False, True])
)
constant_or_near_constant = feature_inventory_df.loc[feature_inventory_df["near_constant"], "feature"].tolist()
family_counts = feature_inventory_df.groupby("family")["feature"].count().rename("n_features").reset_index()

print("Feature path:", feature_path_in_use)
print("mit_feature_table shape:", mit_feature_table.shape)
print("separator_frame_long_df shape:", separator_frame_long_df.shape)
print("separator_output_feature_long_df shape:", separator_output_feature_long_df.shape)
print("fastica_feature_df shape:", fastica_feature_df.shape)
print("Available separators:", available_separators)
print("Missing expected separators:", missing_expected_separators)
print("Near-constant columns:", constant_or_near_constant[:12], "..." if len(constant_or_near_constant) > 12 else "")
print("Top missing columns:")
display(missingness_df.head(15))
print("Feature inventory by family:")
display(family_counts)
print("Feature inventory sample:")
display(feature_inventory_df.head(20))
""",
    ),
    md(
        "multichannel-unsup-methods-md",
        r"""
## Multichannel Geometry Roadmap

The exploratory workflow below deliberately moves from coarse summaries to progressively richer structure discovery.

1. **Feature sanity checks and grouped summaries** tell us whether the four-channel mixtures differ mostly in raw power, spatial covariance, phase relationships, or spectral concentration.
2. **PCA / factor-style methods** test whether a small number of latent axes explains most of the variation. If the first few components are interpretable, that is evidence that the mixtures occupy a structured low-dimensional subspace rather than a feature cloud with no stable geometry.
3. **Nonlinear embeddings** ask whether the data look more like a curved manifold than a linear subspace.
4. **Clustering and density analysis** test whether there are discrete regimes, or whether difficulty changes more smoothly along a trajectory.

In RF terms, the main question is whether "easy" and "hard" mixtures differ in ways that are visible before separation: stronger inter-channel coherence, cleaner covariance geometry, better power balance, or sharper spectral structure would all be plausible markers of recoverability.
""",
    ),
    code(
        "multichannel-unsup-exploration",
        r"""
unsup_feature_cols = numeric_feature_columns(
    mit_feature_table,
    extra_exclude=merge_keys + ["source_path"],
    min_non_null=0.98,
)

candidate_unsup_features = [
    "power_imbalance", "amp_std_all", "papr_mean",
    "cov_condition", "cov_spectral_entropy", "cov_participation_ratio",
    "coherence_mean", "offdiag_cov_abs_mean", "offdiag_cov_phase_std",
    "phase_diff_std_mean", "phase_diff_resultant_mean",
    "spec_entropy_mean", "spec_flatness_mean", "spec_rolloff_85_mean", "spec_bandwidth_mean",
]
selected_unsup_features = [col for col in candidate_unsup_features if col in mit_feature_table.columns][:10]
if len(selected_unsup_features) < 6:
    selected_unsup_features = top_feature_list(mit_feature_table, unsup_feature_cols, n=8)

print("Using", len(unsup_feature_cols), "numeric multichannel features for unsupervised analysis.")
print("Selected exploratory features:", selected_unsup_features)

alpha_summary_features = (
    mit_feature_table.groupby("alphaIndex")[selected_unsup_features]
    .agg(["mean", "std"])
    .pipe(flatten_columns)
    .reset_index()
)
save_table(alpha_summary_features, "mit_alpha_feature_summary.csv")

if fastica_feature_df["success"].notna().any():
    success_summary_features = (
        fastica_feature_df.dropna(subset=["success"]).groupby("success")[selected_unsup_features + ["best_num_errors"]]
        .agg(["mean", "std"])
        .pipe(flatten_columns)
        .reset_index()
    )
    save_table(success_summary_features, "mit_success_feature_summary.csv")
    print("FastICA-labeled success/failure summary:")
    display(success_summary_features)
else:
    print("No FastICA success labels were available for grouped summaries.")

if not separator_feature_long_df.empty:
    separator_summary_features = (
        separator_feature_long_df.groupby("separator")[selected_unsup_features + ["best_num_errors", "success"]]
        .agg(["mean", "std"])
        .pipe(flatten_columns)
        .reset_index()
    )
    save_table(separator_summary_features, "mit_separator_feature_summary.csv")
    print("Separator-level feature summary:")
    display(separator_summary_features)

corr_features = selected_unsup_features[:8]
if corr_features:
    corr_df = mit_feature_table[corr_features].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Correlation Heatmap For Selected Multichannel Features")
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_correlation_heatmap.png")
    plt.show()

scatter_features = selected_unsup_features[:4]
if len(scatter_features) >= 4:
    sampled = mit_feature_table.sample(min(len(mit_feature_table), 500), random_state=42).copy()
    sampled["alpha_group"] = pd.cut(sampled["alphaIndex"].astype(int), bins=[0, 5, 15, 25], labels=["low", "mid", "high"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    pairings = [
        (scatter_features[0], scatter_features[1]),
        (scatter_features[0], scatter_features[2]),
        (scatter_features[1], scatter_features[3]),
        (scatter_features[2], scatter_features[3]),
    ]
    for ax, (xcol, ycol) in zip(axes.flat, pairings):
        sns.scatterplot(data=sampled, x=xcol, y=ycol, hue="alpha_group", alpha=0.7, s=35, ax=ax)
        ax.set_title(f"{xcol} vs {ycol}")
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_scatter_matrix.png")
    plt.show()
""",
    ),
    md(
        "multichannel-unsup-pca-md",
        r"""
## PCA, Factor Structure, And Statistical Interpretation

PCA is the most direct way to ask whether multichannel MIT frames lie near a low-dimensional linear subspace. If the first few components explain a large share of variance and have coherent loadings, those components can often be interpreted in physical terms:

- **amplitude-driven components** suggest that global energy, power imbalance, or PAPR dominates the geometry;
- **covariance-driven components** suggest that spatial anisotropy or channel mixing geometry matters;
- **coherence/phase-driven components** suggest that channel alignment is a strong latent driver;
- **spectral-driven components** suggest different interference textures or time-frequency concentration regimes.

The component summaries printed below are meant to be reusable as paper-style interpretive notes rather than just debugging output.
""",
    ),
    code(
        "multichannel-unsup-pca",
        r"""
X_unsup_raw = mit_feature_table[unsup_feature_cols].replace([np.inf, -np.inf], np.nan)
X_unsup = X_unsup_raw.fillna(X_unsup_raw.median(numeric_only=True))
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(X_unsup)

n_pca_components = int(min(10, X_unsup_scaled.shape[1], X_unsup_scaled.shape[0]))
pca_model = PCA(n_components=n_pca_components, random_state=42)
pca_scores = pca_model.fit_transform(X_unsup_scaled)
pca_cols = [f"PC{i+1}" for i in range(n_pca_components)]

pca_scores_df = mit_feature_table[merge_keys + ["alphaIndex"]].copy()
for idx, col in enumerate(pca_cols):
    pca_scores_df[col] = pca_scores[:, idx]
pca_scores_df = pca_scores_df.merge(
    fastica_feature_df[merge_keys + [col for col in ["success", "best_num_errors"] if col in fastica_feature_df.columns]],
    on=merge_keys,
    how="left",
)
save_table(pca_scores_df, "mit_multichannel_pca_scores.csv")

pca_loadings_df = pd.DataFrame(
    pca_model.components_.T,
    index=unsup_feature_cols,
    columns=pca_cols,
).reset_index().rename(columns={"index": "feature"})
pca_loadings_df["family"] = pca_loadings_df["feature"].map(feature_family)
save_table(pca_loadings_df, "mit_multichannel_pca_loadings.csv")

alpha_centroids_pca = pca_scores_df.groupby("alphaIndex")[[col for col in ["PC1", "PC2", "PC3"] if col in pca_scores_df.columns]].mean().reset_index()
save_table(alpha_centroids_pca, "mit_multichannel_pca_alpha_centroids.csv")

explained_df = pd.DataFrame({
    "component": pca_cols,
    "explained_variance_ratio": pca_model.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
})
save_table(explained_df, "mit_multichannel_pca_explained_variance.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, n_pca_components + 1), pca_model.explained_variance_ratio_)
axes[0].set_title("PCA Scree Plot")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[1].plot(range(1, n_pca_components + 1), np.cumsum(pca_model.explained_variance_ratio_), marker="o")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_ylim(0, 1.05)
fig.tight_layout()
save_figure(fig, "mit_multichannel_pca_scree.png")
plt.show()

plot_specs = [
    ("PC1", "PC2", "alphaIndex", "viridis", "PC1 vs PC2 by alphaIndex"),
    ("PC1", "PC3", "alphaIndex", "viridis", "PC1 vs PC3 by alphaIndex"),
    ("PC2", "PC3", "success", "coolwarm", "PC2 vs PC3 by FastICA success"),
    ("PC1", "PC2", "best_num_errors", "magma_r", "PC1 vs PC2 by FastICA best_num_errors"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (xcol, ycol, hue_col, cmap_name, title) in zip(axes.flat, plot_specs):
    if xcol not in pca_scores_df.columns or ycol not in pca_scores_df.columns or hue_col not in pca_scores_df.columns:
        ax.set_visible(False)
        continue
    plot_df = pca_scores_df.dropna(subset=[xcol, ycol, hue_col]).copy()
    if plot_df.empty:
        ax.set_visible(False)
        continue
    scatter = ax.scatter(plot_df[xcol], plot_df[ycol], c=plot_df[hue_col], cmap=cmap_name, s=24, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    fig.colorbar(scatter, ax=ax)
fig.tight_layout()
save_figure(fig, "mit_multichannel_pca_views.png")
plt.show()


def summarize_pc(pc_name, top_n=12):
    ranked = pca_loadings_df[["feature", "family", pc_name]].copy()
    ranked["abs_loading"] = ranked[pc_name].abs()
    ranked = ranked.sort_values("abs_loading", ascending=False)
    family_summary = (
        ranked.groupby("family")["abs_loading"].sum().sort_values(ascending=False)
    )
    top_families = family_summary.head(3).index.tolist()
    top_features = ranked.head(top_n)["feature"].tolist()
    print(f"{pc_name}: dominant feature families = {top_families}")
    print(f"{pc_name}: strongest features = {top_features[:8]}")


for pc_name in [col for col in ["PC1", "PC2", "PC3"] if col in pca_loadings_df.columns]:
    summarize_pc(pc_name)

print("Top PCA loadings:")
display(pca_loadings_df.head())
""",
    ),
    code(
        "multichannel-unsup-linear-latents",
        r"""
linear_latent_tables = []

try:
    fa_model = FactorAnalysis(n_components=2, random_state=42)
    fa_scores = fa_model.fit_transform(X_unsup_scaled)
    fa_df = mit_feature_table[merge_keys + ["alphaIndex"]].copy()
    fa_df["FA1"] = fa_scores[:, 0]
    fa_df["FA2"] = fa_scores[:, 1]
    fa_df = fa_df.merge(fastica_feature_df[merge_keys + [col for col in ["success", "best_num_errors"] if col in fastica_feature_df.columns]], on=merge_keys, how="left")
    save_table(fa_df, "mit_multichannel_factor_scores.csv")
    fa_loadings_df = pd.DataFrame(fa_model.components_.T, index=unsup_feature_cols, columns=["FA1", "FA2"]).reset_index().rename(columns={"index": "feature"})
    fa_loadings_df["family"] = fa_loadings_df["feature"].map(feature_family)
    save_table(fa_loadings_df, "mit_multichannel_factor_loadings.csv")
    linear_latent_tables.append(("Factor Analysis", fa_df, ["FA1", "FA2"]))
except Exception as exc:
    print("Factor Analysis skipped:", exc)

try:
    feature_ica_model = SkFastICA(n_components=2, random_state=42, max_iter=1200)
    feature_ica_scores = feature_ica_model.fit_transform(X_unsup_scaled)
    feature_ica_df = mit_feature_table[merge_keys + ["alphaIndex"]].copy()
    feature_ica_df["ICA1"] = feature_ica_scores[:, 0]
    feature_ica_df["ICA2"] = feature_ica_scores[:, 1]
    feature_ica_df = feature_ica_df.merge(fastica_feature_df[merge_keys + [col for col in ["success", "best_num_errors"] if col in fastica_feature_df.columns]], on=merge_keys, how="left")
    save_table(feature_ica_df, "mit_multichannel_feature_ica_scores.csv")
    feature_ica_loadings_df = pd.DataFrame(feature_ica_model.components_.T, index=unsup_feature_cols, columns=["ICA1", "ICA2"]).reset_index().rename(columns={"index": "feature"})
    feature_ica_loadings_df["family"] = feature_ica_loadings_df["feature"].map(feature_family)
    save_table(feature_ica_loadings_df, "mit_multichannel_feature_ica_loadings.csv")
    linear_latent_tables.append(("Feature-space ICA", feature_ica_df, ["ICA1", "ICA2"]))
except Exception as exc:
    print("Feature-space ICA skipped:", exc)

if linear_latent_tables:
    fig, axes = plt.subplots(len(linear_latent_tables), 2, figsize=(12, 4 * len(linear_latent_tables)))
    if len(linear_latent_tables) == 1:
        axes = np.array([axes])
    for row_axes, (label, table, cols) in zip(axes, linear_latent_tables):
        ax_alpha, ax_success = row_axes
        sc = ax_alpha.scatter(table[cols[0]], table[cols[1]], c=table["alphaIndex"], cmap="viridis", s=24, alpha=0.75)
        ax_alpha.set_title(f"{label} colored by alphaIndex")
        ax_alpha.set_xlabel(cols[0]); ax_alpha.set_ylabel(cols[1])
        fig.colorbar(sc, ax=ax_alpha)
        if "success" in table.columns and table["success"].notna().any():
            plot_df = table.dropna(subset=["success"])
            sc2 = ax_success.scatter(plot_df[cols[0]], plot_df[cols[1]], c=plot_df["success"], cmap="coolwarm", s=24, alpha=0.75)
            ax_success.set_title(f"{label} colored by FastICA success")
            ax_success.set_xlabel(cols[0]); ax_success.set_ylabel(cols[1])
            fig.colorbar(sc2, ax=ax_success)
        else:
            ax_success.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_linear_latents.png")
    plt.show()
""",
    ),
    md(
        "multichannel-unsup-manifold-md",
        r"""
## Nonlinear Manifolds, Density, And Regime Discovery

If the PCA views look structured but slightly curved, a manifold method is often more revealing than another linear rotation. The nonlinear embeddings below are designed to answer two practical questions:

- Does **alphaIndex** behave like a smooth trajectory through a mixture manifold?
- Do **successful and failed frames** occupy distinct neighborhoods even before separation?

If the answer is yes, then separator performance is partly a problem of navigating an already structured multichannel geometry rather than dealing with frame-to-frame randomness alone.
""",
    ),
    code(
        "multichannel-unsup-manifolds",
        r"""
if len(mit_feature_table) > 1500:
    embed_sample_df = (
        mit_feature_table.groupby("alphaIndex", group_keys=False)
        .apply(lambda grp: grp.sample(min(len(grp), 60), random_state=42), include_groups=False)
        .reset_index(drop=True)
    )
else:
    embed_sample_df = mit_feature_table.copy()

embed_sample_df = embed_sample_df.merge(
    fastica_feature_df[merge_keys + [col for col in ["success", "best_num_errors"] if col in fastica_feature_df.columns]],
    on=merge_keys,
    how="left",
)
X_embed = embed_sample_df[unsup_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(X_unsup_raw.median(numeric_only=True))
X_embed_scaled = scaler_unsup.transform(X_embed)

embedding_frames = {}
embedding_registry_rows = []
embedding_specs = {
    "tSNE": TSNE(n_components=2, perplexity=min(35, max(10, len(embed_sample_df) // 20)), learning_rate="auto", init="pca", random_state=42),
    "Isomap": Isomap(n_components=2, n_neighbors=min(15, max(5, len(embed_sample_df) - 1))),
    "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=min(20, max(6, len(embed_sample_df) - 1)), random_state=42),
}
if umap is not None:
    embedding_specs["UMAP"] = umap.UMAP(n_components=2, n_neighbors=min(25, max(10, len(embed_sample_df) - 1)), min_dist=0.1, metric="euclidean", random_state=42)

for method_name, estimator in embedding_specs.items():
    try:
        coords = estimator.fit_transform(X_embed_scaled)
        emb_df = embed_sample_df[merge_keys + ["alphaIndex"] + [col for col in ["success", "best_num_errors"] if col in embed_sample_df.columns]].copy()
        emb_df["dim1"] = coords[:, 0]
        emb_df["dim2"] = coords[:, 1]
        emb_df["method"] = method_name
        embedding_frames[method_name] = emb_df
        embedding_registry_rows.append({"method": method_name, "status": "ok", "n_rows": len(emb_df)})
    except Exception as exc:
        print(f"{method_name} skipped: {exc}")
        embedding_registry_rows.append({"method": method_name, "status": f"skipped: {exc}", "n_rows": 0})

embedding_registry_df = pd.DataFrame(embedding_registry_rows)
save_table(embedding_registry_df, "mit_multichannel_embedding_registry.csv")

if embedding_frames:
    combined_embeddings_df = pd.concat(embedding_frames.values(), ignore_index=True)
    save_table(combined_embeddings_df, "mit_multichannel_embedding_coordinates.csv")

    alpha_centroid_rows = []
    for method_name, emb_df in embedding_frames.items():
        alpha_centroids = emb_df.groupby("alphaIndex")[["dim1", "dim2"]].mean().reset_index()
        alpha_centroids["method"] = method_name
        alpha_centroid_rows.append(alpha_centroids)
    alpha_centroid_embedding_df = pd.concat(alpha_centroid_rows, ignore_index=True)
    save_table(alpha_centroid_embedding_df, "mit_multichannel_embedding_alpha_centroids.csv")

    for method_name, emb_df in embedding_frames.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        sc = axes[0].scatter(emb_df["dim1"], emb_df["dim2"], c=emb_df["alphaIndex"], cmap="viridis", s=26, alpha=0.8)
        axes[0].set_title(f"{method_name} colored by alphaIndex")
        axes[0].set_xlabel("dim1"); axes[0].set_ylabel("dim2")
        fig.colorbar(sc, ax=axes[0])

        if "success" in emb_df.columns and emb_df["success"].notna().any():
            success_df = emb_df.dropna(subset=["success"])
            sc2 = axes[1].scatter(success_df["dim1"], success_df["dim2"], c=success_df["success"], cmap="coolwarm", s=26, alpha=0.8)
            axes[1].set_title(f"{method_name} colored by FastICA success")
            axes[1].set_xlabel("dim1"); axes[1].set_ylabel("dim2")
            fig.colorbar(sc2, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "No success labels", ha="center", va="center")
            axes[1].set_title(f"{method_name}: success view unavailable")

        if "best_num_errors" in emb_df.columns and emb_df["best_num_errors"].notna().any():
            err_df = emb_df.dropna(subset=["best_num_errors"])
            sc3 = axes[2].scatter(err_df["dim1"], err_df["dim2"], c=err_df["best_num_errors"], cmap="magma_r", s=26, alpha=0.8)
            axes[2].set_title(f"{method_name} colored by best_num_errors")
            axes[2].set_xlabel("dim1"); axes[2].set_ylabel("dim2")
            fig.colorbar(sc3, ax=axes[2])
        else:
            axes[2].text(0.5, 0.5, "No error labels", ha="center", va="center")
            axes[2].set_title(f"{method_name}: error view unavailable")

        fig.tight_layout()
        save_figure(fig, f"mit_multichannel_embedding_{method_name.lower()}.png")
        plt.show()

successful_embedding_methods = list(embedding_frames.keys())
print("Successful embedding methods:", successful_embedding_methods)
""",
    ),
    code(
        "multichannel-unsup-density-clustering",
        r"""
density_features = selected_unsup_features[:4]
if density_features and fastica_feature_df["success"].notna().any():
    density_df = fastica_feature_df.dropna(subset=["success"]).copy()
    fig, axes = plt.subplots(len(density_features), 1, figsize=(9, 3.2 * len(density_features)))
    if len(density_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, density_features):
        sns.kdeplot(data=density_df, x=feat, hue="success", common_norm=False, fill=True, alpha=0.35, ax=ax)
        ax.set_title(f"Density of {feat} by FastICA success")
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_density_success.png")
    plt.show()

if {"PC1", "PC2", "success"}.issubset(pca_scores_df.columns) and pca_scores_df["success"].notna().any():
    density_pca_df = pca_scores_df.dropna(subset=["success"]).copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.kdeplot(data=density_pca_df, x="PC1", y="PC2", hue="success", fill=True, common_norm=False, thresh=0.05, alpha=0.3, ax=ax)
    ax.set_title("Density contours in PCA space by FastICA success")
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_pca_density_success.png")
    plt.show()

cluster_input_cols = [col for col in [f"PC{i+1}" for i in range(min(8, n_pca_components))] if col in pca_scores_df.columns]
cluster_base_df = pca_scores_df[merge_keys + ["alphaIndex"] + cluster_input_cols + [col for col in ["success", "best_num_errors"] if col in pca_scores_df.columns]].copy()
cluster_matrix = cluster_base_df[cluster_input_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

cluster_label_df = cluster_base_df[merge_keys + ["alphaIndex"]].copy()
cluster_summary_frames = []
cluster_registry_rows = []
cluster_methods = {
    "kmeans": KMeans(n_clusters=4, random_state=42, n_init=20),
    "gmm": GaussianMixture(n_components=4, covariance_type="full", random_state=42),
    "agglomerative": AgglomerativeClustering(n_clusters=4),
    "dbscan": DBSCAN(eps=1.25, min_samples=20),
}

for method_name, estimator in cluster_methods.items():
    try:
        labels = estimator.fit_predict(cluster_matrix)
        if len(np.unique(labels)) <= 1:
            raise ValueError("returned one cluster only")
        cluster_label_df[f"{method_name}_cluster"] = labels
        summary = cluster_base_df.copy()
        summary[f"{method_name}_cluster"] = labels
        primary_stats = (
            summary.groupby(f"{method_name}_cluster")
            .agg(
                n_frames=("alphaIndex", "count"),
                alpha_mean=("alphaIndex", "mean"),
                alpha_std=("alphaIndex", "std"),
                success_rate=("success", "mean"),
                mean_best_num_errors=("best_num_errors", "mean"),
            )
            .reset_index()
        )
        selected_feature_means = summary.groupby(f"{method_name}_cluster")[selected_unsup_features[:6]].mean().reset_index()
        method_summary = primary_stats.merge(selected_feature_means, on=f"{method_name}_cluster", how="left")
        method_summary["method"] = method_name
        cluster_summary_frames.append(method_summary)
        cluster_registry_rows.append({"method": method_name, "status": "ok", "n_clusters": int(len(np.unique(labels)) - int((-1 in labels)))})
    except Exception as exc:
        print(f"{method_name} clustering skipped: {exc}")
        cluster_registry_rows.append({"method": method_name, "status": f"skipped: {exc}", "n_clusters": 0})

cluster_registry_df = pd.DataFrame(cluster_registry_rows)
save_table(cluster_registry_df, "mit_multichannel_cluster_registry.csv")
save_table(cluster_label_df, "mit_multichannel_cluster_labels.csv")

if cluster_summary_frames:
    cluster_summary_df = pd.concat(cluster_summary_frames, ignore_index=True)
    save_table(cluster_summary_df, "mit_multichannel_cluster_summary.csv")
    display(cluster_summary_df)

primary_cluster_col = "kmeans_cluster" if "kmeans_cluster" in cluster_label_df.columns else None
if primary_cluster_col is not None and {"PC1", "PC2"}.issubset(pca_scores_df.columns):
    cluster_plot_df = pca_scores_df.merge(cluster_label_df[merge_keys + [primary_cluster_col]], on=merge_keys, how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=cluster_plot_df, x="PC1", y="PC2", hue=primary_cluster_col, palette="tab10", s=28, alpha=0.8, ax=axes[0])
    axes[0].set_title("KMeans clusters in PCA space")
    alpha_centroids = cluster_plot_df.groupby("alphaIndex")[["PC1", "PC2"]].mean().reset_index().sort_values("alphaIndex")
    axes[1].plot(alpha_centroids["PC1"], alpha_centroids["PC2"], marker="o")
    for _, row in alpha_centroids.iterrows():
        axes[1].text(row["PC1"], row["PC2"], int(row["alphaIndex"]), fontsize=8)
    axes[1].set_title("AlphaIndex trajectory in PCA space")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.tight_layout()
    save_figure(fig, "mit_multichannel_clusters_and_alpha_trajectory.png")
    plt.show()
""",
    ),
    md(
        "multichannel-unsup-interpretation-md",
        r"""
## Interpretation Notes

Read the multichannel plots above as a sequence of progressively stronger claims.

- If the grouped feature summaries change smoothly with **alphaIndex**, then the MIT mixtures behave more like a **difficulty trajectory** than a collection of unrelated regimes.
- If success and failure separate in PCA or manifold space, then there is evidence that **recoverability is partly visible from the raw multichannel mixture geometry itself**.
- If clustering produces groups with systematically different coherence, covariance conditioning, or spectral concentration, then the data likely contains **latent mixture regimes** rather than a single homogeneous family.

That distinction matters for the separator story below: a model can fail either because the entire domain is out of distribution, or because only certain geometric regimes remain unsolved.
""",
    ),
    md(
        "multichannel-separator-intro",
        r"""
# Part G: Multichannel Structure, Separator Performance, And Prediction

The next step is to connect the low-dimensional mixture structure to system behavior. We merge the multichannel feature space with MIT recovery labels and separator outputs, then ask three questions:

1. Which mixture regimes are easy or hard for **FastICA**?
2. Where do the **learned separators** fail relative to FastICA?
3. Can simple statistical models predict success, error count, or separator advantage from the mixture geometry alone?
""",
    ),
    code(
        "multichannel-separator-merge",
        r"""
if not separator_summary_df.empty:
    summary_merge_cols = [col for col in ["separator", "frameLen", "setIndex", "alphaIndex", "ber", "frameSuccessRate"] if col in separator_summary_df.columns]
    separator_feature_long_df = separator_feature_long_df.merge(
        separator_summary_df[summary_merge_cols].drop_duplicates(),
        on=[col for col in ["separator", "frameLen", "setIndex", "alphaIndex"] if col in summary_merge_cols],
        how="left",
    )
    separator_output_feature_long_df = separator_output_feature_long_df.merge(
        separator_summary_df[summary_merge_cols].drop_duplicates(),
        on=[col for col in ["separator", "frameLen", "setIndex", "alphaIndex"] if col in summary_merge_cols],
        how="left",
    )

if 'cluster_label_df' in globals() and not separator_feature_long_df.empty:
    separator_feature_long_df = separator_feature_long_df.merge(cluster_label_df, on=merge_keys, how="left")
    separator_output_feature_long_df = separator_output_feature_long_df.merge(cluster_label_df, on=merge_keys, how="left")

separator_comparison_summary = pd.DataFrame()
if not separator_feature_long_df.empty:
    separator_comparison_summary = (
        separator_feature_long_df.groupby(["separator", "alphaIndex"], dropna=False)
        .agg(
            frame_success_rate=("success", "mean"),
            mean_best_num_errors=("best_num_errors", "mean"),
            mean_ber=("ber", "mean"),
            n_frames=("frame_number", "count"),
        )
        .reset_index()
    )
    save_table(separator_comparison_summary, "mit_separator_comparison_summary.csv")

separator_payload_summary = pd.DataFrame()
if not separator_output_feature_long_df.empty:
    separator_payload_summary = (
        separator_output_feature_long_df.groupby(["separator", "output_index", "success"], dropna=False)
        .agg(
            n_rows=("frame_number", "count"),
            mean_numErrors=("numErrors", "mean"),
            mean_payload_std_abs=("payload_std_abs", "mean"),
            mean_payload_phase_std=("payload_phase_std", "mean"),
            mean_payload_power_mean=("payload_power_mean", "mean"),
        )
        .reset_index()
    )
    save_table(separator_payload_summary, "mit_separator_payload_summary.csv")

separator_availability_df = pd.DataFrame({
    "available_separator": available_separators,
})
save_table(separator_availability_df, "mit_available_separators.csv")
print("Available separators:", available_separators)
print("Missing expected separators:", missing_expected_separators)
if not separator_comparison_summary.empty:
    display(separator_comparison_summary.head(12))
""",
    ),
    code(
        "multichannel-separator-plots",
        r"""
if separator_comparison_summary.empty:
    print("Skipping separator-comparison plots because no merged separator table is available.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    sns.lineplot(data=separator_comparison_summary, x="alphaIndex", y="frame_success_rate", hue="separator", marker="o", ax=axes[0])
    axes[0].set_title("Frame success rate by alphaIndex")
    axes[0].set_ylim(-0.05, 1.05)

    sns.lineplot(data=separator_comparison_summary, x="alphaIndex", y="mean_best_num_errors", hue="separator", marker="o", ax=axes[1])
    axes[1].set_title("Mean best_num_errors by alphaIndex")

    sns.lineplot(data=separator_comparison_summary, x="alphaIndex", y="mean_ber", hue="separator", marker="o", ax=axes[2])
    axes[2].set_title("BER by alphaIndex")
    fig.tight_layout()
    save_figure(fig, "mit_separator_comparison_curves.png")
    plt.show()

if not separator_output_feature_long_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=separator_output_feature_long_df, x="separator", y="numErrors", ax=axes[0])
    axes[0].set_title("Output-level error distribution by separator")
    axes[0].tick_params(axis="x", rotation=30)

    sns.scatterplot(
        data=separator_output_feature_long_df,
        x="payload_std_abs",
        y="payload_phase_std",
        hue="separator",
        style="output_index",
        alpha=0.6,
        ax=axes[1],
    )
    axes[1].set_title("Payload dispersion by separator")
    fig.tight_layout()
    save_figure(fig, "mit_separator_payload_dispersion.png")
    plt.show()

if not separator_feature_long_df.empty and {"PC1", "PC2"}.issubset(pca_scores_df.columns):
    embedding_compare_df = separator_feature_long_df.merge(pca_scores_df[merge_keys + ["PC1", "PC2"]], on=merge_keys, how="left")
    sampled_compare_df = embedding_compare_df.sample(min(len(embedding_compare_df), 1400), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=sampled_compare_df, x="PC1", y="PC2", hue="separator", s=28, alpha=0.75, ax=axes[0])
    axes[0].set_title("PCA space colored by separator")
    sns.scatterplot(data=sampled_compare_df.dropna(subset=["success"]), x="PC1", y="PC2", hue="success", style="separator", s=28, alpha=0.75, ax=axes[1])
    axes[1].set_title("PCA space colored by outcome and separator")
    fig.tight_layout()
    save_figure(fig, "mit_separator_embedding_comparison.png")
    plt.show()
""",
    ),
    md(
        "multichannel-separator-prediction-md",
        r"""
## Prediction And Comparator Models

These models are not intended to replace the separators. They answer a different question: **how much of separator performance is statistically predictable from the multichannel mixture geometry before separation happens?**

If simple classical models can predict success, error count, or BER with reasonable accuracy, then the separator problem has a meaningful low-dimensional statistical structure. If they cannot, that suggests that the failures are driven by more local or more nonlinear effects than these summary features can capture.
""",
    ),
    code(
        "multichannel-separator-models",
        r"""
separator_prediction_results = []
separator_model_feature_cols = [col for col in unsup_feature_cols if col not in {"alphaIndex"}]
if len(separator_model_feature_cols) > 24:
    separator_model_feature_cols = top_feature_list(mit_feature_table, separator_model_feature_cols, n=24)

if separator_feature_long_df.empty:
    print("Skipping separator prediction models because separator_feature_long_df is empty.")
else:
    pooled_model_df = separator_feature_long_df.dropna(subset=["success", "best_num_errors"]).copy()
    if not pooled_model_df.empty:
        X_num = pooled_model_df[separator_model_feature_cols].replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(X_num.median(numeric_only=True))
        X_pooled = pd.concat([X_num, pd.get_dummies(pooled_model_df["separator"], prefix="sep")], axis=1)
        y_success = pooled_model_df["success"].astype(int)

        if y_success.nunique() >= 2 and y_success.value_counts().min() >= 2:
            cv = StratifiedKFold(n_splits=min(5, int(y_success.value_counts().min())), shuffle=True, random_state=42)
            classifiers = {
                "LDA": LinearDiscriminantAnalysis(),
                "QDA": QuadraticDiscriminantAnalysis(reg_param=0.05),
                "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
            }
            for name, estimator in classifiers.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = cross_val_score(make_pipeline(StandardScaler(), estimator), X_pooled, y_success, cv=cv, scoring="accuracy")
                    separator_prediction_results.append({
                        "scope": "pooled",
                        "task": "success",
                        "model": name,
                        "mean_cv_score": float(scores.mean()),
                        "std_cv_score": float(scores.std()),
                        "metric": "accuracy",
                        "status": "ok",
                    })
                except Exception as exc:
                    separator_prediction_results.append({
                        "scope": "pooled", "task": "success", "model": name,
                        "mean_cv_score": np.nan, "std_cv_score": np.nan, "metric": "accuracy",
                        "status": f"skipped: {exc}",
                    })
        else:
            print("Success classification skipped: pooled labels are one-class or too imbalanced.")

        regressors = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
        }
        reg_cv = KFold(n_splits=min(5, max(2, len(X_pooled) // 100)), shuffle=True, random_state=42)
        for target in [col for col in ["best_num_errors", "ber"] if col in pooled_model_df.columns and pooled_model_df[col].notna().any()]:
            y_reg = pooled_model_df[target].astype(float)
            for name, estimator in regressors.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r2_scores = cross_val_score(make_pipeline(StandardScaler(), estimator), X_pooled, y_reg, cv=reg_cv, scoring="r2")
                        rmse_scores = -cross_val_score(make_pipeline(StandardScaler(), estimator), X_pooled, y_reg, cv=reg_cv, scoring="neg_root_mean_squared_error")
                    separator_prediction_results.append({
                        "scope": "pooled",
                        "task": target,
                        "model": name,
                        "mean_cv_score": float(r2_scores.mean()),
                        "std_cv_score": float(r2_scores.std()),
                        "metric": "r2",
                        "mean_rmse": float(rmse_scores.mean()),
                        "status": "ok",
                    })
                except Exception as exc:
                    separator_prediction_results.append({
                        "scope": "pooled", "task": target, "model": name,
                        "mean_cv_score": np.nan, "std_cv_score": np.nan, "metric": "r2", "mean_rmse": np.nan,
                        "status": f"skipped: {exc}",
                    })

        for separator_name, group in pooled_model_df.groupby("separator"):
            group_y = group["success"].astype(int)
            if group_y.nunique() < 2 or group_y.value_counts().min() < 2:
                separator_prediction_results.append({
                    "scope": separator_name,
                    "task": "success",
                    "model": "LDA / QDA / Logistic",
                    "mean_cv_score": np.nan,
                    "std_cv_score": np.nan,
                    "metric": "accuracy",
                    "status": "skipped: one-class or too imbalanced",
                })
                continue
            X_group = group[separator_model_feature_cols].replace([np.inf, -np.inf], np.nan)
            X_group = X_group.fillna(X_group.median(numeric_only=True))
            cv_group = StratifiedKFold(n_splits=min(5, int(group_y.value_counts().min())), shuffle=True, random_state=42)
            for name, estimator in {
                "LDA": LinearDiscriminantAnalysis(),
                "QDA": QuadraticDiscriminantAnalysis(reg_param=0.05),
                "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
            }.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = cross_val_score(make_pipeline(StandardScaler(), estimator), X_group, group_y, cv=cv_group, scoring="accuracy")
                    separator_prediction_results.append({
                        "scope": separator_name,
                        "task": "success",
                        "model": name,
                        "mean_cv_score": float(scores.mean()),
                        "std_cv_score": float(scores.std()),
                        "metric": "accuracy",
                        "status": "ok",
                    })
                except Exception as exc:
                    separator_prediction_results.append({
                        "scope": separator_name, "task": "success", "model": name,
                        "mean_cv_score": np.nan, "std_cv_score": np.nan, "metric": "accuracy",
                        "status": f"skipped: {exc}",
                    })

separator_prediction_results_df = pd.DataFrame(separator_prediction_results)
if not separator_prediction_results_df.empty:
    save_table(separator_prediction_results_df, "mit_separator_prediction_results.csv")
    display(separator_prediction_results_df)
""",
    ),
    code(
        "multichannel-separator-regime-analysis",
        r"""
separator_advantage_summary = pd.DataFrame()
cluster_separator_summary = pd.DataFrame()

if not separator_feature_long_df.empty:
    fastica_rows = separator_feature_long_df[separator_feature_long_df["separator"] == "FastICA"][merge_keys + ["best_num_errors", "success", "ber"]].copy()
    advantage_rows = []
    hybrid_advantage_df = pd.DataFrame()
    for separator_name in sorted(separator_feature_long_df["separator"].dropna().unique()):
        if separator_name == "FastICA":
            continue
        model_rows = separator_feature_long_df[separator_feature_long_df["separator"] == separator_name][merge_keys + ["best_num_errors", "success", "ber"]].copy()
        overlap = fastica_rows.merge(model_rows, on=merge_keys, how="inner", suffixes=("_fastica", "_model"))
        if overlap.empty:
            advantage_rows.append({
                "separator": separator_name,
                "n_overlap": 0,
                "mean_error_advantage": np.nan,
                "median_error_advantage": np.nan,
                "pct_model_better": np.nan,
            })
            continue
        overlap["error_advantage"] = overlap["best_num_errors_model"] - overlap["best_num_errors_fastica"]
        overlap["model_better"] = overlap["error_advantage"] < 0
        advantage_rows.append({
            "separator": separator_name,
            "n_overlap": int(len(overlap)),
            "mean_error_advantage": float(overlap["error_advantage"].mean()),
            "median_error_advantage": float(overlap["error_advantage"].median()),
            "pct_model_better": float(overlap["model_better"].mean()),
        })
        if separator_name == "Learned: Hybrid":
            hybrid_advantage_df = overlap.merge(pca_scores_df[merge_keys + [col for col in ["PC1", "PC2"] if col in pca_scores_df.columns]], on=merge_keys, how="left")

    separator_advantage_summary = pd.DataFrame(advantage_rows)
    save_table(separator_advantage_summary, "mit_separator_advantage_summary.csv")
    display(separator_advantage_summary)

    if not hybrid_advantage_df.empty and {"PC1", "PC2"}.issubset(hybrid_advantage_df.columns):
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(hybrid_advantage_df["PC1"], hybrid_advantage_df["PC2"], c=hybrid_advantage_df["error_advantage"], cmap="coolwarm", s=28, alpha=0.8)
        ax.set_title("Hybrid minus FastICA error advantage in PCA space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="Hybrid best_num_errors - FastICA best_num_errors")
        fig.tight_layout()
        save_figure(fig, "mit_hybrid_fastica_advantage_pca.png")
        plt.show()

    if "kmeans_cluster" in separator_feature_long_df.columns:
        cluster_separator_summary = (
            separator_feature_long_df.groupby(["kmeans_cluster", "separator"], dropna=False)
            .agg(
                n_rows=("frame_number", "count"),
                frame_success_rate=("success", "mean"),
                mean_best_num_errors=("best_num_errors", "mean"),
                mean_ber=("ber", "mean"),
                alpha_mean=("alphaIndex", "mean"),
            )
            .reset_index()
        )
        save_table(cluster_separator_summary, "mit_cluster_separator_summary.csv")
        display(cluster_separator_summary)
""",
    ),
    md(
        "multichannel-separator-interpretation-md",
        r"""
## Separator-Focused Interpretation

This final section is where the geometric picture becomes operational.

- If **FastICA** succeeds in the same regions where coherence is high and covariance structure is well-conditioned, then the classical separator is exploiting stable spatial geometry.
- If the **learned separators** fail almost everywhere on MIT, the most likely explanation is not just random bad luck but a **domain-transfer mismatch**: the synthetic training distribution is not landing in the same parts of multichannel feature space as the real MIT mixtures.
- If a learned separator is especially poor in only a subset of clusters or along one portion of the alpha trajectory, that is more encouraging: it implies a **regime-specific generalization gap** rather than total failure.

In other words, the feature-space analysis above is meant to diagnose *where* the domain gap lives, not merely to report that it exists.
""",
    ),
]


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    nb["cells"] = [cell for cell in nb["cells"] if cell.get("id") not in NEW_IDS]
    nb["cells"].extend(CELLS)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Added {len(CELLS)} multichannel MIT analysis cells to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
