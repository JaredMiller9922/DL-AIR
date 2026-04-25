import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent / "DataAnalysis.ipynb"
INSERT_BEFORE_ID = "sep-debug-intro"


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
    "single-channel-review-intro",
    "single-channel-review-audit",
    "single-channel-review-interpretation",
}


NEW_CELLS = [
    md(
        "single-channel-review-intro",
        """
# Part C1: Single-Channel Review And Guardrails

Before moving into the MIT recovery backend, it helps to be explicit about what the earlier single-channel-style section is doing well and where it can mislead.

Those first sections are best interpreted as **signal-family and mixture-description analysis** on the demod training set. They are useful for understanding broad class geometry, feature salience, and whether the data contains recognizable statistical structure. They are **not** the same as end-to-end separator evaluation.

One especially important caveat is that `center_freq` behaves like a near-direct dataset cue in this collection. That makes it a useful descriptive field, but not a fair stand-in for physically meaningful separation difficulty. For that reason, the no-`center_freq` models are the more honest baseline when discussing generalization.
""",
    ),
    code(
        "single-channel-review-audit",
        """
if "df" not in globals():
    print("Single-channel review skipped: base dataframe `df` is not defined yet.")
else:
    audit_feature_sets = {
        "no_center_freq": feature_cols_no_cf if "feature_cols_no_cf" in globals() else [],
        "with_center_freq": feature_cols_with_cf if "feature_cols_with_cf" in globals() else [],
    }
    audit_rows = []
    for feature_set_name, cols in audit_feature_sets.items():
        available = [col for col in cols if col in df.columns]
        if not available:
            continue
        frame = df[available]
        audit_rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": len(available),
                "mean_missing_fraction": float(frame.isna().mean().mean()),
                "near_constant_features": int((frame.nunique(dropna=True) <= 2).sum()),
            }
        )

    single_channel_audit_df = pd.DataFrame(audit_rows)
    single_channel_class_balance_df = (
        df.groupby(["base_class", "kind"]).size().reset_index(name="n_rows")
        if {"base_class", "kind"}.issubset(df.columns)
        else pd.DataFrame()
    )
    single_channel_numeric_inventory_df = (
        pd.DataFrame(
            {
                "feature": df.select_dtypes(include=[np.number]).columns,
                "missing_fraction": [float(df[col].isna().mean()) for col in df.select_dtypes(include=[np.number]).columns],
                "std": [float(df[col].std()) for col in df.select_dtypes(include=[np.number]).columns],
            }
        )
        .sort_values(["missing_fraction", "std", "feature"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    print("Single-channel dataframe shape:", df.shape)
    print("Base-class counts:")
    if "base_class" in df.columns:
        print(df["base_class"].value_counts())
    print("\\nFeature-set audit:")
    display(single_channel_audit_df)
    print("\\nClass balance by signal family and file role:")
    display(single_channel_class_balance_df)
    print("\\nMost variable numeric features:")
    display(single_channel_numeric_inventory_df.head(12))
""",
    ),
    md(
        "single-channel-review-interpretation",
        """
## Single-Channel Takeaways

The early notebook results are still valuable, but they should be read with the right emphasis:

- the strongest conclusions there are about **feature geometry and class structure**,
- the cleanest baseline comparisons are the ones **excluding `center_freq`**,
- and the later MIT sections should be treated as the real recovery-side test of whether those geometric intuitions survive contact with multichannel separation and decoding.

That framing makes the notebook more coherent: the single-channel portion motivates the feature-engineering story, while the MIT sections test whether those same ideas remain useful when the task becomes actual recovery rather than descriptive classification.
""",
    ),
]


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = [cell for cell in nb["cells"] if cell.get("id") not in NEW_IDS]

    insert_at = len(cells)
    for idx, cell in enumerate(cells):
        if cell.get("id") == INSERT_BEFORE_ID:
            insert_at = idx
            break

    cells[insert_at:insert_at] = NEW_CELLS
    nb["cells"] = cells
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted {len(NEW_CELLS)} single-channel review cells into {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
