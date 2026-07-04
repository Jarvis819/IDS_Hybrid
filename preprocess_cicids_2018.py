import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from preprocess_cicids import (
    BENIGN_LABEL_NORMALIZED,
    TRAIN_ONLY_ATTACK_LABELS_NORMALIZED,
    ROOT_DIR as _TRAIN_ROOT_DIR,
    assign_synthetic_flow_endpoints_if_missing,
    canonicalize_label_string,
    clean_column_names,
    clean_label_text,
    coerce_numerics_and_handle_infinite,
    drop_identifier_and_duplicates,
    drop_null_and_duplicate_rows,
    normalize_label_key,
)


# Reuse the same root as the original CICIDS 2017 preprocessing script.
ROOT_DIR = _TRAIN_ROOT_DIR
RAW_2018_DIR = ROOT_DIR / "CIC-IDS-2018-Dataset"
PROCESSED_DIR = ROOT_DIR / "processed"


def list_raw_2018_csvs(raw_dir: Path) -> List[Path]:
    """List all CICIDS 2018 CSV files under CIC-IDS-2018-Dataset."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw CICIDS 2018 data directory not found: {raw_dir}")
    # Skip temporary lock files such as .~lock.*
    return sorted(p for p in raw_dir.glob("*.csv") if not p.name.startswith(".~lock"))


def load_and_concatenate_2018(csv_paths: List[Path]) -> pd.DataFrame:
    """Load all CICIDS 2018 CSVs and vertically concatenate them, adding a source-file column."""
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df["SourceFile"] = path.name
        frames.append(df)
    if not frames:
        raise ValueError(f"No CSV files found in {RAW_2018_DIR}")
    return pd.concat(frames, axis=0, ignore_index=True)


def align_to_training_features(
    df: pd.DataFrame,
    artifacts_path: Path,
) -> Tuple[pd.DataFrame, list]:
    """
    Align CICIDS 2018 dataframe with the training feature space.

    - Loads scaler + preprocessing artifacts from training.
    - Ensures the same numeric feature columns exist (adding missing ones if needed).
    - Applies median imputation and scaling using training medians/scaler.
    """
    artifacts = joblib.load(artifacts_path)
    numeric_feature_names = list(artifacts["numeric_feature_names"])
    imputation_medians = artifacts["imputation_medians"]
    scaler = artifacts["scaler"]

    df = df.copy()

    # Ensure all training numeric feature columns exist in the 2018 frame.
    for col in numeric_feature_names:
        if col not in df.columns:
            df[col] = 0.0

    # Drop any extra numeric columns that were not used for training features.
    extra_cols = [c for c in df.columns if c not in numeric_feature_names and c not in {"Label", "Source IP", "Destination IP", "Timestamp", "SourceFile"}]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # Impute using training medians so the distribution seen by the model is consistent.
    for col in numeric_feature_names:
        if col not in df.columns:
            # Should not happen after above loop, but guard just in case.
            df[col] = 0.0
        median_val = imputation_medians.get(col, 0.0)
        df[col] = df[col].fillna(median_val)

    # Apply the training scaler to obtain standardized features.
    numeric_values = df[numeric_feature_names].astype(np.float32).values
    scaled_values = scaler.transform(numeric_values)
    df[numeric_feature_names] = scaled_values

    return df, numeric_feature_names


def filter_version_two_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For version two, keep only BENIGN + the attack classes the model was trained on.
    This mirrors the label filtering step used in training.
    """
    if "Label" not in df.columns:
        return df

    df = df.copy()
    label_norm = df["Label"].astype(str).map(normalize_label_key)
    keep_mask = label_norm.eq(BENIGN_LABEL_NORMALIZED) | label_norm.isin(TRAIN_ONLY_ATTACK_LABELS_NORMALIZED)
    return df.loc[keep_mask].reset_index(drop=True)


def run_preprocessing_2018() -> None:
    """
    Preprocess CICIDS 2018 data into two aligned test sets:

    - Version 1: all attack classes present in CICIDS 2018 (no attack-class filtering).
    - Version 2: only BENIGN + the three attack classes used for training.

    Both versions:
    - Reuse the CICIDS 2017 preprocessing steps (column/label cleaning, ID/duplicate drops, etc.).
    - Are aligned to the training model's numeric feature set and scaler.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = list_raw_2018_csvs(RAW_2018_DIR)
    print(f"[2018] Found {len(csv_paths)} raw CSV files under {RAW_2018_DIR}", flush=True)

    # Process each CSV incrementally to reduce peak memory usage, then concatenate
    # only after basic cleaning / type coercion has been applied.
    cleaned_parts = []
    total_rows_raw = 0
    for i, path in enumerate(csv_paths, start=1):
        print(f"[2018] Loading file {i}/{len(csv_paths)}: {path.name}", flush=True)
        df_part = load_and_concatenate_2018([path])
        total_rows_raw += len(df_part)
        print(f"[2018]  Raw shape for {path.name}: {df_part.shape}", flush=True)

        df_part = clean_column_names(df_part)
        df_part = clean_label_text(df_part)
        df_part, _ = drop_identifier_and_duplicates(df_part)
        df_part, _ = drop_null_and_duplicate_rows(df_part)
        df_part = coerce_numerics_and_handle_infinite(df_part)
        df_part = assign_synthetic_flow_endpoints_if_missing(df_part)

        print(f"[2018]  After cleaning/coercion for {path.name}: {df_part.shape}", flush=True)
        cleaned_parts.append(df_part)

    df = pd.concat(cleaned_parts, axis=0, ignore_index=True)
    print(
        f"[2018] Combined cleaned shape: {df.shape} (total raw rows across files: {total_rows_raw})",
        flush=True,
    )

    # Align with training feature space and scaler.
    artifacts_path = PROCESSED_DIR / "scaler.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Training preprocessing artifacts not found at {artifacts_path}. "
            "Run preprocess_cicids.py first to generate training artifacts."
        )

    df_scaled, feature_names = align_to_training_features(df, artifacts_path)
    print(f"[2018] After alignment to training features: {df_scaled.shape}; num_features={len(feature_names)}", flush=True)

    # Version 1: keep all attack classes as present after cleaning.
    df_v1 = df_scaled.copy()

    # Version 2: keep only BENIGN + the attack classes used in training.
    df_v2 = filter_version_two_labels(df_scaled)
    print(f"[2018] Version 2 (train classes only) shape: {df_v2.shape}", flush=True)

    # Persist both versions as parquet and CSV (CSV is used for some evaluation scripts
    # to avoid parquet engine compatibility issues on some environments).
    v1_parquet = PROCESSED_DIR / "traffic_2018_all_classes.parquet"
    v2_parquet = PROCESSED_DIR / "traffic_2018_train_classes.parquet"
    v1_csv = PROCESSED_DIR / "traffic_2018_all_classes.csv"
    v2_csv = PROCESSED_DIR / "traffic_2018_train_classes.csv"

    df_v1.to_parquet(v1_parquet, index=False)
    df_v2.to_parquet(v2_parquet, index=False)
    df_v1.to_csv(v1_csv, index=False)
    df_v2.to_csv(v2_csv, index=False)

    print(f"[2018] Saved version 1 (all classes) to: {v1_parquet} and {v1_csv}", flush=True)
    print(f"[2018] Saved version 2 (train classes only) to: {v2_parquet} and {v2_csv}", flush=True)


if __name__ == "__main__":
    run_preprocessing_2018()

