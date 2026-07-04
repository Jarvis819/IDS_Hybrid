"""
Create balanced CICIDS 2018 evaluation datasets for:

- Model B (binary): ~1M BENIGN + ALL attacks, label = {BENIGN, ATTACK}
- Model A (multi):  ~1M BENIGN + mapped attacks into {DDoS, DoS Hulk} (+ PortScan absent)

This does NOT reread raw CICIDS 2018 CSVs. It reuses the already-cleaned parquet outputs:
- processed/traffic_2018_all_classes.parquet (multi pipeline, LabelOriginal-style names)
- processed/traffic_2018_binary_all_classes.parquet (binary pipeline, Label + LabelOriginal)

Outputs (written to processed/):
- traffic_2018_binary_eval_balanced.parquet
- traffic_2018_binary_eval_balanced.csv
- traffic_2018_multiclass_eval_balanced.parquet
- traffic_2018_multiclass_eval_balanced.csv
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from preprocess_cicids import (
    canonicalize_label_string,
    normalize_label_key,
    BENIGN_LABEL_NORMALIZED,
    assign_synthetic_flow_endpoints_if_missing,
)


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"


def _read_parquet_any_engine(path: Path, columns: List[str] | None = None) -> pd.DataFrame:
    last: Exception | None = None
    engines = [None]
    if importlib.util.find_spec("fastparquet") is not None:
        engines.append("fastparquet")
    for engine in engines:
        try:
            kw: Dict[str, Any] = {}
            if columns is not None:
                kw["columns"] = columns
            if engine is not None:
                kw["engine"] = engine
            return pd.read_parquet(path, **kw)
        except Exception as e:
            last = e
    raise last if last else RuntimeError(f"Failed to read parquet: {path}")


def _read_table_parquet_or_csv(base_path: Path, columns: List[str]) -> pd.DataFrame:
    """
    Read parquet if possible; otherwise fall back to CSV with selected columns.
    """
    def _filter_cols(available: List[str]) -> List[str]:
        avail_set = set(available)
        keep = [c for c in columns if c in avail_set]
        missing = [c for c in columns if c not in avail_set]
        if missing:
            print(
                f"[reprocess-2018] WARNING: {base_path.name} missing columns {missing}. "
                "Continuing without them (will add placeholders for graph cols later).",
                flush=True,
            )
        return keep

    if base_path.suffix.lower() == ".csv":
        # Use C engine for speed; skip bad lines if any.
        # Read header via pandas so quoted column names are parsed correctly.
        available = list(pd.read_csv(base_path, nrows=0).columns)
        usecols = _filter_cols(available)
        return pd.read_csv(base_path, usecols=usecols, low_memory=False, on_bad_lines="skip")

    try:
        # Parquet engines will error if we request missing columns, so probe columns first.
        try:
            probe = _read_parquet_any_engine(base_path, columns=None)
        except Exception:
            # If parquet itself is unreadable, fall back to CSV below.
            raise
        usecols = _filter_cols(list(probe.columns))
        return probe[usecols]
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        if not csv_path.exists():
            raise
        available = list(pd.read_csv(csv_path, nrows=0).columns)
        usecols = _filter_cols(available)
        return pd.read_csv(csv_path, usecols=usecols, low_memory=False, on_bad_lines="skip")


def _ensure_graph_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some saved 2018 artifacts may not include Source/Destination IP columns.
    HybridIDSDataset needs them for graph construction; we can safely fill placeholders.
    """
    df = df.copy()
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.Timestamp.utcnow()
    if "SourceFile" not in df.columns:
        df["SourceFile"] = "unknown"
    # Also handles the case where IP columns exist but are all placeholder values.
    df = assign_synthetic_flow_endpoints_if_missing(df)
    return df


def _safe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_binary_eval(target_benign: int, seed: int = 42) -> pd.DataFrame:
    art = joblib.load(PROCESSED / "scaler_binary.joblib")
    feat_cols = list(art["numeric_feature_names"])
    # Some binary parquet exports may not include IP columns; we will add placeholders after loading.
    keep_cols = [
        "Label",
        "LabelOriginal",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "SourceFile",
    ] + feat_cols

    src_path = PROCESSED / "traffic_2018_binary_all_classes.parquet"
    if not src_path.exists():
        src_path = PROCESSED / "traffic_2018_binary_all_classes.csv"
    if not src_path.exists():
        raise FileNotFoundError("Missing binary 2018 base file (expected parquet or csv).")

    df = _read_table_parquet_or_csv(src_path, columns=keep_cols)
    df = _ensure_graph_cols(df)
    # Normalize label casing just in case
    df["Label"] = df["Label"].astype(str).map(canonicalize_label_string)

    benign_mask = df["Label"].map(normalize_label_key).eq(BENIGN_LABEL_NORMALIZED)
    df_b = df.loc[benign_mask].reset_index(drop=True)
    df_a = df.loc[~benign_mask].reset_index(drop=True)

    df_b = _safe_sample(df_b, target_benign, seed)
    out = pd.concat([df_b, df_a], axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _map_2018_to_model_a(label: str) -> str:
    """
    Map CICIDS 2018 label strings to the Model A label space.
    """
    s = canonicalize_label_string(label)
    k = normalize_label_key(s)
    if k == BENIGN_LABEL_NORMALIZED:
        return "BENIGN"

    # DDoS family mapping
    if k in {
        normalize_label_key("DDOS attack-HOIC"),
        normalize_label_key("DDoS attacks-LOIC-HTTP"),
        normalize_label_key("DDOS attack-LOIC-UDP"),
    }:
        return "DDoS"

    # DoS Hulk mapping
    if k in {normalize_label_key("DoS attacks-Hulk")}:
        return "DoS Hulk"

    # PortScan is not present in your 2018 label summary.
    return "__DROP__"


def build_multiclass_eval(target_benign: int, seed: int = 42) -> pd.DataFrame:
    art = joblib.load(PROCESSED / "scaler.joblib")
    feat_cols = list(art["numeric_feature_names"])
    # Multi parquet should usually include IP columns, but still guard and fill placeholders.
    keep_cols = ["Label", "Source IP", "Destination IP", "Timestamp", "SourceFile"] + feat_cols

    src_path = PROCESSED / "traffic_2018_all_classes.parquet"
    if not src_path.exists():
        src_path = PROCESSED / "traffic_2018_all_classes.csv"
    if not src_path.exists():
        raise FileNotFoundError("Missing multi 2018 base file (expected parquet or csv).")

    df = _read_table_parquet_or_csv(src_path, columns=keep_cols)
    df = _ensure_graph_cols(df)
    df["Label"] = df["Label"].astype(str).map(canonicalize_label_string)
    df["Label"] = df["Label"].map(_map_2018_to_model_a)

    df = df.loc[df["Label"] != "__DROP__"].reset_index(drop=True)

    benign_mask = df["Label"].map(normalize_label_key).eq(BENIGN_LABEL_NORMALIZED)
    df_b = df.loc[benign_mask].reset_index(drop=True)
    df_a = df.loc[~benign_mask].reset_index(drop=True)

    df_b = _safe_sample(df_b, target_benign, seed)
    out = pd.concat([df_b, df_a], axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> int:
    target_benign = int(os.environ.get("EVAL_TARGET_BENIGN", "1000000"))
    seed = int(os.environ.get("EVAL_SAMPLE_SEED", "42"))
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"[reprocess-2018] target_benign={target_benign:,} seed={seed}", flush=True)

    df_bin = build_binary_eval(target_benign=target_benign, seed=seed)
    out_bin_pq = PROCESSED / "traffic_2018_binary_eval_balanced.parquet"
    out_bin_csv = PROCESSED / "traffic_2018_binary_eval_balanced.csv"
    df_bin.to_parquet(out_bin_pq, index=False)
    df_bin.to_csv(out_bin_csv, index=False)
    print(f"[reprocess-2018] wrote {out_bin_pq.name} rows={len(df_bin):,}", flush=True)
    print(df_bin["Label"].value_counts().to_string(), flush=True)

    df_mc = build_multiclass_eval(target_benign=target_benign, seed=seed)
    out_mc_pq = PROCESSED / "traffic_2018_multiclass_eval_balanced.parquet"
    out_mc_csv = PROCESSED / "traffic_2018_multiclass_eval_balanced.csv"
    df_mc.to_parquet(out_mc_pq, index=False)
    df_mc.to_csv(out_mc_csv, index=False)
    print(f"[reprocess-2018] wrote {out_mc_pq.name} rows={len(df_mc):,}", flush=True)
    print(df_mc["Label"].value_counts().to_string(), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

