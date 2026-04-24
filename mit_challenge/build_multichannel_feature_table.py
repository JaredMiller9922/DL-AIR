from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.rf_features import complex_mixture_features

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "mit_challenge" / "rfchallenge_multichannel_starter-main" / "mixtureData"
DEFAULT_BASE_CSV = PROJECT_ROOT / "mit_challenge" / "separation_frame_features.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "mit_challenge" / "separation_frame_features_extended.csv"

FILENAME_RE = re.compile(
    r"input_frameLen_(?P<frameLen>\d+)_setIndex_(?P<setIndex>\d+)_alphaIndex_(?P<alphaIndex>\d+)_frame(?P<frame>\d+)\.iqdata$"
)


def read_mit_iqdata(path: Path, n_rx: int = 4) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % 2 != 0:
        raise ValueError(f"Odd number of float32 values in {path}")
    complex_samples = raw[0::2] + 1j * raw[1::2]
    if complex_samples.size % n_rx != 0:
        raise ValueError(f"Complex sample count {complex_samples.size} is not divisible by n_rx={n_rx} for {path}")
    samples_per_rx = complex_samples.size // n_rx
    return complex_samples.reshape(samples_per_rx, n_rx).T.astype(np.complex64)


def parse_keys(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not parse MIT frame keys from filename {path.name}")
    data = {key: int(value) for key, value in match.groupdict().items()}
    return {
        "alphaIndex": data["alphaIndex"],
        "frameLen": data["frameLen"],
        "setIndex": data["setIndex"],
        "frame_number": data["frame"],
    }


def build_feature_rows(input_root: Path, n_rx: int = 4) -> pd.DataFrame:
    rows = []
    paths = sorted(input_root.rglob("input_frameLen_*_setIndex_*_alphaIndex_*_frame*.iqdata"))
    if not paths:
        raise FileNotFoundError(f"No MIT backend mixture .iqdata files found under {input_root}")

    for idx, path in enumerate(paths, start=1):
        mixture = read_mit_iqdata(path, n_rx=n_rx)
        row = parse_keys(path)
        row.update(complex_mixture_features(mixture))
        row["source_path"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(row)
        if idx % 500 == 0:
            print(f"Processed {idx} mixture frames...")

    feature_df = pd.DataFrame(rows).sort_values(["frameLen", "setIndex", "alphaIndex", "frame_number"]).reset_index(drop=True)
    return feature_df


def merge_with_existing(base_csv: Path, feature_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["alphaIndex", "frameLen", "setIndex", "frame_number"]
    if not base_csv.exists():
        return feature_df

    base_df = pd.read_csv(base_csv)
    overlap = [col for col in feature_df.columns if col in base_df.columns and col not in merge_keys]
    merged = base_df.merge(feature_df, on=merge_keys, how="outer", suffixes=("", "__new"))
    for col in overlap:
        new_col = f"{col}__new"
        if new_col in merged.columns:
            merged[col] = merged[col].where(~merged[col].isna(), merged[new_col])
            merged = merged.drop(columns=[new_col])
    merged = merged.sort_values(merge_keys).reset_index(drop=True)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an extended multichannel MIT feature table.")
    parser.add_argument("--input_root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--base_csv", default=str(DEFAULT_BASE_CSV))
    parser.add_argument("--n_rx", type=int, default=4)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_csv = Path(args.output_csv)
    base_csv = Path(args.base_csv)

    feature_df = build_feature_rows(input_root, n_rx=args.n_rx)
    merged_df = merge_with_existing(base_csv, feature_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Wrote extended multichannel feature table to {output_csv}")
    print(f"Rows: {len(merged_df)}, Columns: {len(merged_df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
