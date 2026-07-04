import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocess_cicids import (
    ROOT_DIR,
    clean_column_names,
    clean_label_text,
    drop_identifier_and_duplicates,
    drop_null_and_duplicate_rows,
    coerce_numerics_and_handle_infinite,
)


RAW_DIR = ROOT_DIR / "TrafficLabelling"
PROCESSED_DIR = ROOT_DIR / "processed"


def list_raw_csvs(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    return sorted(raw_dir.glob("*.csv"))


def load_concat(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df["SourceFile"] = path.name
        frames.append(df)
    if not frames:
        raise ValueError("No CSV files found.")
    return pd.concat(frames, axis=0, ignore_index=True)


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csvs = list_raw_csvs(RAW_DIR)
    print(f"[binary] Found {len(csvs)} raw CSV files", flush=True)

    df = load_concat(csvs)
    print(f"[binary] Raw shape: {df.shape}", flush=True)

    df = clean_column_names(df)
    df = clean_label_text(df)
    df, _ = drop_identifier_and_duplicates(df)
    df, _ = drop_null_and_duplicate_rows(df)
    df = coerce_numerics_and_handle_infinite(df)
    print(f"[binary] After base cleaning: {df.shape}", flush=True)

    if "Label" not in df.columns:
        raise ValueError("Label column missing after preprocessing.")

    # Binary mapping: BENIGN -> BENIGN, all attacks -> ATTACK
    lbl = df["Label"].astype(str).str.strip()
    df["LabelOriginal"] = lbl
    df["Label"] = np.where(lbl.str.upper() == "BENIGN", "BENIGN", "ATTACK")

    keep_meta = {"Label", "LabelOriginal", "Source IP", "Destination IP", "Timestamp", "SourceFile"}
    numeric_cols = [c for c in df.columns if c not in keep_meta and pd.api.types.is_numeric_dtype(df[c])]

    medians = {}
    for c in numeric_cols:
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        medians[c] = float(med)
        df[c] = df[c].fillna(med)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(np.float32).values)

    out_parquet = PROCESSED_DIR / "traffic_all_binary_processed.parquet"
    out_csv = PROCESSED_DIR / "traffic_all_binary_processed.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    artifacts = {
        "numeric_feature_names": numeric_cols,
        "imputation_medians": medians,
        "scaler": scaler,
        "pipeline_type": "binary_all_attacks",
    }
    joblib.dump(artifacts, PROCESSED_DIR / "scaler_binary.joblib")

    summary = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "label_counts": df["Label"].value_counts().to_dict(),
        "label_original_top20": df["LabelOriginal"].value_counts().head(20).to_dict(),
        "num_numeric_features": len(numeric_cols),
    }
    (PROCESSED_DIR / "preprocessing_summary_binary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[binary] Saved: {out_parquet}", flush=True)
    print(f"[binary] Saved artifacts: {PROCESSED_DIR / 'scaler_binary.joblib'}", flush=True)


if __name__ == "__main__":
    run()

