"""Primitive data analysis pipeline for combined raw CICIDS CSV files.

This script:
1) Combines all CSVs in TrafficLabelling/ into one CSV file.
2) Runs basic exploratory analysis.
3) Saves all analysis outputs to disk for lightweight dashboard loading.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "TrafficLabelling"
COMBINED_DIR = ROOT / "combined_data"
ANALYSIS_DIR = ROOT / "analysis_outputs" / "primitive"


def list_csvs(raw_dir: Path) -> List[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return files


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def combine_csvs() -> Path:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = list_csvs(RAW_DIR)

    frames = []
    for p in csv_paths:
        df = read_csv_with_fallback(p)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out_path = COMBINED_DIR / "traffic_all_combined.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def save_plot(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_analysis(combined_csv_path: Path) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(combined_csv_path, low_memory=False, encoding="latin-1")
    # Normalize column names similar to preprocessing pipeline.
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    print(f"Loaded combined CSV: {df.shape}", flush=True)

    # Basic dataset understanding
    shape = df.shape
    columns = list(df.columns)

    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_text = info_buf.getvalue()
    (ANALYSIS_DIR / "df_info.txt").write_text(info_text, encoding="utf-8")

    # Expensive operations run on a representative sample for speed.
    describe_sample_n = min(300_000, len(df))
    df_desc = df.sample(n=describe_sample_n, random_state=42) if len(df) > describe_sample_n else df
    describe_df = df_desc.describe(include="all")
    describe_df.to_csv(ANALYSIS_DIR / "df_describe.csv")
    print("Saved df.describe()", flush=True)

    # Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_df = missing.rename("missing_count").reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df.to_csv(ANALYSIS_DIR / "missing_values.csv", index=False)

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    print("Computed missing values and duplicates", flush=True)

    # Label distribution
    label_counts_df = pd.DataFrame(columns=["Label", "count"])
    if "Label" in df.columns:
        label_counts = df["Label"].astype(str).value_counts(dropna=False)
        label_counts_df = label_counts.rename("count").reset_index()
        label_counts_df.columns = ["Label", "count"]
        label_counts_df.to_csv(ANALYSIS_DIR / "label_counts.csv", index=False)

        fig = plt.figure(figsize=(14, 6))
        # Countplot on sampled rows for plotting speed; counts table is exact.
        plot_n = min(300_000, len(df))
        df_plot = df.sample(n=plot_n, random_state=42) if len(df) > plot_n else df
        sns.countplot(x="Label", data=df_plot)
        plt.xticks(rotation=45, ha="right")
        plt.title("Label Distribution")
        save_plot(fig, ANALYSIS_DIR / "label_countplot.png")
        print("Saved label distribution", flush=True)

    # Correlation analysis (numeric only)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_sample_n = min(200_000, len(numeric_df))
    if len(numeric_df) > corr_sample_n:
        numeric_corr = numeric_df.sample(n=corr_sample_n, random_state=42)
    else:
        numeric_corr = numeric_df
    corr = numeric_corr.corr(numeric_only=True) if not numeric_corr.empty else pd.DataFrame()
    corr.to_csv(ANALYSIS_DIR / "correlation_matrix.csv")

    if not corr.empty:
        # Large matrices are slow to render; cap to first 50 columns for heatmap image.
        corr_plot = corr.iloc[:50, :50]
        fig = plt.figure(figsize=(14, 12))
        sns.heatmap(corr_plot, cmap="coolwarm")
        plt.title("Correlation Heatmap (first 50 numeric features)")
        save_plot(fig, ANALYSIS_DIR / "correlation_heatmap.png")
        print("Saved correlation outputs", flush=True)

    # Feature importance (RandomForest)
    feat_imp_df = pd.DataFrame(columns=["feature", "importance"])
    if "Label" in df.columns and not numeric_df.empty:
        X = numeric_df.copy()
        y = df["Label"].astype(str).copy()

        # Keep analysis practical: fit RF on a large sample, not full 3M rows.
        max_rf_rows = 150_000
        if len(X) > max_rf_rows:
            sample_idx = X.sample(n=max_rf_rows, random_state=42).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

        # Fill NaNs before RF
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        y_enc, _ = pd.factorize(y)

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X, y_enc)
        importances = rf.feature_importances_

        feat_imp_df = pd.DataFrame(
            {"feature": X.columns.astype(str), "importance": importances}
        ).sort_values("importance", ascending=False)
        feat_imp_df.to_csv(ANALYSIS_DIR / "feature_importances.csv", index=False)

        top20 = feat_imp_df.head(20).iloc[::-1]
        fig = plt.figure(figsize=(10, 8))
        plt.barh(top20["feature"], top20["importance"])
        plt.title("Top 20 Feature Importances (RandomForest)")
        plt.xlabel("Importance")
        save_plot(fig, ANALYSIS_DIR / "feature_importances_top20.png")
        print("Saved RandomForest feature importances", flush=True)

    summary = {
        "combined_csv_path": str(combined_csv_path),
        "shape": {"rows": int(shape[0]), "cols": int(shape[1])},
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "num_missing_columns": int((missing > 0).sum()),
        "top_missing_columns": missing_df.head(20).to_dict(orient="records"),
        "label_distribution_available": bool("Label" in df.columns),
        "correlation_shape": {"rows": int(corr.shape[0]), "cols": int(corr.shape[1])},
        "feature_importance_rows": int(len(feat_imp_df)),
    }
    with open(ANALYSIS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary.json", flush=True)


def main() -> None:
    combined_csv_path = combine_csvs()
    print(f"Combined CSV saved to: {combined_csv_path}")
    run_analysis(combined_csv_path)
    print(f"Primitive analysis outputs saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()

