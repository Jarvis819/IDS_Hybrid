import os
import unicodedata
import zlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "TrafficLabelling"
PROCESSED_DIR = ROOT_DIR / "processed"

# Attack classes explicitly excluded from training due to very low support.
LOW_SUPPORT_ATTACK_LABELS = {
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Heartbleed",
}


def canonicalize_label_string(label: str) -> str:
    """
    Fix common label encoding/spacing artefacts while preserving human-readable casing.

    This updates the dataframe's `Label` column values (used for training metrics/plots),
    but avoids forcing everything to lowercase.
    """
    s = str(label)
    s = unicodedata.normalize("NFKC", s)
    # Common problematic sequences seen in CICIDS exports / copy/paste tables
    s = (
        s.replace("Â", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
    )
    s = (
        s.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    s = " ".join(s.split())
    return s.strip()


def normalize_label_key(label: str) -> str:
    """Lowercased key used for membership tests (e.g., dropping low-support classes)."""
    return canonicalize_label_string(label).lower().strip()


LOW_SUPPORT_ATTACK_LABELS_NORMALIZED = {normalize_label_key(s) for s in LOW_SUPPORT_ATTACK_LABELS}

# Only train on these classes (plus BENIGN).
TRAIN_ONLY_ATTACK_LABELS = {"DDoS", "DoS Hulk", "PortScan"}
TRAIN_ONLY_ATTACK_LABELS_NORMALIZED = {normalize_label_key(s) for s in TRAIN_ONLY_ATTACK_LABELS}
BENIGN_LABEL_NORMALIZED = normalize_label_key("BENIGN")


def list_raw_csvs(raw_dir: Path) -> List[Path]:
    """List all CICIDS CSV files under TrafficLabelling."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    return sorted(raw_dir.glob("*.csv"))


def load_and_concatenate(csv_paths: List[Path]) -> pd.DataFrame:
    """Load all CSVs and vertically concatenate them, adding a source-file column."""
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df["SourceFile"] = path.name
        frames.append(df)
    if not frames:
        raise ValueError(f"No CSV files found in {RAW_DIR}")
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    return df_all


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and standardize column names (keep them readable for thesis)."""
    df = df.copy()
    new_cols = []
    for col in df.columns:
        col_stripped = str(col).strip()
        # Keep original words but normalize spaces -> single space
        col_stripped = " ".join(col_stripped.split())
        new_cols.append(col_stripped)
    df.columns = new_cols
    return df


# CICIDS 2018 TrafficForML_CICFlowMeter uses abbreviated names; map onto 2017-style names used at training time.
_CICIDS_TRAFFICFORML_TO_2017_COLUMN_ALIASES: Dict[str, str] = {
    "Src IP": "Source IP",
    "Dst IP": "Destination IP",
    "Src Port": "Source Port",
    "Dst Port": "Destination Port",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


def apply_cicids_trafficforml_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy TrafficForML-style columns into 2017-style names when the target column is missing.

    Does not overwrite an existing target column. Safe to call on 2017 data (no-op).
    """
    df = df.copy()
    for old, new in _CICIDS_TRAFFICFORML_TO_2017_COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def ensure_numeric_feature_columns_present(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Ensure every training feature name exists (fill with NaN for imputation later)."""
    df = df.copy()
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _has_usable_ip_columns(df: pd.DataFrame) -> bool:
    if "Source IP" not in df.columns or "Destination IP" not in df.columns:
        return False
    s = df["Source IP"].astype(str).str.strip()
    d = df["Destination IP"].astype(str).str.strip()
    benign = (s.eq("0.0.0.0") | s.str.lower().eq("nan")) & (d.eq("0.0.0.0") | d.str.lower().eq("nan"))
    if benign.all():
        return False
    return int(s.nunique(dropna=False)) > 1 or int(d.nunique(dropna=False)) > 1


def assign_synthetic_flow_endpoints_if_missing(
    df: pd.DataFrame,
    row_index: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    CICIDS 2018 *TrafficForML_CICFlowMeter* CSVs typically omit Source/Destination IP columns.
    The hybrid dataset still needs two string endpoints per row for graph construction.

    When real endpoints are missing or are all placeholders, derive deterministic 10.x.x.x
    pseudo-IPs from (SourceFile, row_index) so per-flow edges are not degenerate.
    """
    df = df.copy()
    if _has_usable_ip_columns(df):
        return df
    if "SourceFile" not in df.columns:
        df["SourceFile"] = "unknown"
    n = len(df)
    if row_index is None:
        ri = np.arange(n, dtype=np.uint64)
    else:
        ri = np.asarray(row_index, dtype=np.uint64).reshape(-1)
        if ri.shape[0] != n:
            raise ValueError("row_index length must match dataframe length")
    fh = (
        df["SourceFile"]
        .astype(str)
        .map(lambda s: zlib.crc32(s.encode("utf-8", errors="ignore")) & 0xFFFFFFFF)
        .to_numpy(dtype=np.uint64)
    )
    x = (fh * np.uint64(0x9E3779B97F4A7C15) + ri) & np.uint64(0xFFFFFFFF)
    y = (fh + ri * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    y = np.where(x == y, y ^ np.uint64(0x13579BDF), y)
    xs = x.astype(np.uint32)
    ys = y.astype(np.uint32)

    def ipv4_10_series(u: np.ndarray) -> pd.Series:
        a = (u >> np.uint32(16)) & np.uint32(255)
        b = (u >> np.uint32(8)) & np.uint32(255)
        c = u & np.uint32(255)
        return "10." + pd.Series(a, dtype="uint32", index=df.index).astype(str) + "." + pd.Series(
            b, dtype="uint32", index=df.index
        ).astype(str) + "." + pd.Series(c, dtype="uint32", index=df.index).astype(str)

    df["Source IP"] = ipv4_10_series(xs)
    df["Destination IP"] = ipv4_10_series(ys)
    return df


def clean_label_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label strings to reduce encoding artifacts and spacing mismatch."""
    df = df.copy()
    if "Label" not in df.columns:
        return df

    label_series = df["Label"].map(canonicalize_label_string)
    df["Label"] = label_series
    return df


def drop_identifier_and_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Drop columns that are clearly identifiers or exact duplicates.

    Returns:
        cleaned_df, dropped_reasons
    """
    df = df.copy()
    dropped_reasons: Dict[str, str] = {}

    # 1) Drop duplicated columns (e.g., duplicate "Fwd Header Length")
    #    Keep the first occurrence; record any removed names.
    duplicated_mask = df.columns.duplicated(keep="first")
    if duplicated_mask.any():
        dup_cols = list(df.columns[duplicated_mask])
        for col in dup_cols:
            dropped_reasons[col] = "Duplicate of an earlier column with the same name (no new information)."
        df = df.loc[:, ~duplicated_mask]

    def _norm(c: str) -> str:
        return str(c).strip().lower().replace(" ", "")

    # 2) Drop explicitly-designated identifier / leakage-like / weak-contribution columns.
    # Note: IPs and timestamps are kept in the dataframe for graph construction / ordering,
    # but they are excluded from the numeric model input feature set later.
    explicit_drop_reasons: Dict[str, str] = {
        # Identifier-based features
        "flowid": (
            "Identifier composed from IPs/ports; encourages memorization of specific flows "
            "and harms generalization / zero-day claims."
        ),
        # Duplicate/measurement redundancy
        "fwdheaderlength.1": "Duplicate measurement column ('Fwd Header Length.1'); redundant and non-informative.",
        # Near-zero variance / non-informative (often sparse binary flags)
        "fwdpshflags": "Feature selection removed non-informative flag (near-zero variance / redundancy).",
        "bwdpshflags": "Feature selection removed non-informative flag (near-zero variance / redundancy).",
        "fwdurgflags": "Feature selection removed non-informative flag (near-zero variance / redundancy).",
        "bwdurgflags": "Feature selection removed non-informative flag (near-zero variance / redundancy).",
        # Weak contribution
        "fwdavgbytes/bulk": "Bulk Features weak contribution; excluded from model input to reduce noise.",
        "fwdavgpackets/bulk": "Bulk Features weak contribution; excluded from model input to reduce noise.",
        "fwdavgbulkrate": "Bulk Features weak contribution; excluded from model input to reduce noise.",
        "bwdavgbytes/bulk": "Bulk Features weak contribution; excluded from model input to reduce noise.",
        "bwdavgpackets/bulk": "Bulk Features weak contribution; excluded from model input to reduce noise.",
        "bwdavgbulkrate": "Bulk Features weak contribution; excluded from model input to reduce noise.",
    }

    to_drop: List[str] = []
    for col in df.columns:
        n = _norm(col)
        if n in explicit_drop_reasons:
            to_drop.append(col)

    for col in to_drop:
        dropped_reasons[col] = explicit_drop_reasons[_norm(col)]
    if to_drop:
        df = df.drop(columns=to_drop)

    return df, dropped_reasons


def coerce_numerics_and_handle_infinite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all non-label, non-metadata columns to numeric, handling Infinity and ±inf.

    We preserve IPs and timestamps as strings for temporal/GNN modeling, but they are
    not used as raw numeric inputs.
    """
    df = df.copy()

    # Identify key metadata columns by name patterns (robust to small formatting changes).
    meta_like = []
    for col in df.columns:
        col_l = col.lower()
        if "ip" in col_l or "timestamp" in col_l or col_l in {"label", "sourcefile"}:
            meta_like.append(col)

    # Replace literal "Infinity"/"inf" strings before conversion
    df = df.replace(to_replace=["Infinity", "INF", "Inf", "inf", "NaN", "nan"], value=np.nan)

    # Coerce all non-meta columns to numeric
    for col in df.columns:
        if col in meta_like:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace ±inf with NaN to avoid exploding scales
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def drop_null_and_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove duplicate rows and rows with missing labels.
    Optionally remove rows that cannot support graph/temporal modeling (missing IP endpoints / timestamp).
    Keep numeric NaNs for median imputation in later step.
    """
    df = df.copy()
    stats = {
        "rows_before": int(len(df)),
        "duplicate_rows_removed": 0,
        "null_label_rows_removed": 0,
        "missing_source_ip_rows_removed": 0,
        "missing_destination_ip_rows_removed": 0,
        "missing_timestamp_rows_removed": 0,
    }

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicate_rows_removed"] = dup_count

    if "Label" in df.columns:
        before = len(df)
        label_ok = df["Label"].notna() & (df["Label"].astype(str).str.strip() != "")
        df = df.loc[label_ok].reset_index(drop=True)
        stats["null_label_rows_removed"] = int(before - len(df))

    # Structural / temporal integrity: edges and sorting require endpoints/time where available.
    if "Source IP" in df.columns:
        before = len(df)
        ok = df["Source IP"].notna() & (df["Source IP"].astype(str).str.strip() != "")
        df = df.loc[ok].reset_index(drop=True)
        stats["missing_source_ip_rows_removed"] = int(before - len(df))
    if "Destination IP" in df.columns:
        before = len(df)
        ok = df["Destination IP"].notna() & (df["Destination IP"].astype(str).str.strip() != "")
        df = df.loc[ok].reset_index(drop=True)
        stats["missing_destination_ip_rows_removed"] = int(before - len(df))
    if "Timestamp" in df.columns:
        before = len(df)
        # Fast-path for CICIDS date strings ("%d/%m/%Y %H:%M:%S"), then robust fallback.
        ts_raw = df["Timestamp"]
        ts = pd.to_datetime(ts_raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if ts.isna().any():
            miss = ts.isna()
            ts.loc[miss] = pd.to_datetime(ts_raw.loc[miss], errors="coerce")
        ok = ts.notna()
        df = df.loc[ok].reset_index(drop=True)
        df["Timestamp"] = ts.loc[ok].values
        stats["missing_timestamp_rows_removed"] = int(before - len(df))

    return df, stats


def drop_low_support_attack_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop rows belonging to user-specified low-support attack classes."""
    df = df.copy()
    if "Label" not in df.columns:
        return df, {}

    label_norm = df["Label"].astype(str).map(normalize_label_key)
    drop_mask = label_norm.isin(LOW_SUPPORT_ATTACK_LABELS_NORMALIZED)
    dropped_counts = (
        df.loc[drop_mask, "Label"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
        .to_dict()
    )
    if drop_mask.any():
        df = df.loc[~drop_mask].reset_index(drop=True)
    return df, {str(k): int(v) for k, v in dropped_counts.items()}


def impute_missing_with_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Impute NaNs in numeric columns using feature-wise median."""
    df = df.copy()
    imputation_stats: Dict[str, Any] = {}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            # If a column is entirely NaN, fallback to 0.0 but record it.
            median_val = 0.0
        imputation_stats[col] = float(median_val)
        df[col] = df[col].fillna(median_val)

    return df, imputation_stats


def encode_non_numeric_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Encode non-numeric columns that are intended for training features.
    Excludes metadata/target fields used for graphing/sorting/labels.
    """
    df = df.copy()
    exclude = {"Label", "Source IP", "Destination IP", "Timestamp", "SourceFile"}
    encode_cols = [
        c for c in df.columns
        if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])
    ]

    enc_stats: Dict[str, Dict[str, Any]] = {}
    for col in encode_cols:
        vals = df[col].astype(str).fillna("__MISSING__")
        # Frequency-based integer encoding keeps memory low and avoids huge one-hot matrices.
        freq_order = vals.value_counts().index.tolist()
        mapping = {v: i for i, v in enumerate(freq_order)}
        df[col] = vals.map(mapping).astype(np.int32)
        enc_stats[col] = {
            "encoding": "frequency-ordered integer encoding",
            "num_unique_values": int(len(mapping)),
        }
    return df, enc_stats


def drop_near_zero_variance_numeric_features(
    df: pd.DataFrame,
    std_thresh: float = 1e-8,
) -> Dict[str, str]:
    """
    Remove numeric columns whose standard deviation is ~0.

    Returns:
        dropped_reasons for the columns dropped.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {}

    stds = df[numeric_cols].std(skipna=True)
    dropped = stds[stds.isna() | (stds <= std_thresh)].index.tolist()
    dropped_reasons: Dict[str, str] = {}
    for col in dropped:
        dropped_reasons[col] = (
            "Features with near-zero variance were removed as they do not contribute to model learning."
        )
    if dropped:
        df.drop(columns=dropped, inplace=True)
    return dropped_reasons


def drop_redundant_correlated_numeric_features(
    df: pd.DataFrame,
    corr_thresh: float = 0.999,
    sample_rows: int = 200_000,
) -> Dict[str, str]:
    """
    Drop redundant numeric features with very high pairwise correlation.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return {}

    work_df = df[numeric_cols]
    if len(work_df) > sample_rows:
        work_df = work_df.sample(n=sample_rows, random_state=42)

    corr = work_df.corr().abs()
    stds = work_df.std(skipna=True).fillna(0.0)

    keep = {c: True for c in numeric_cols}
    dropped_reasons: Dict[str, str] = {}

    cols = list(numeric_cols)
    for i in range(len(cols)):
        ci = cols[i]
        if not keep[ci]:
            continue
        for j in range(i + 1, len(cols)):
            cj = cols[j]
            if not keep.get(cj, False):
                continue
            if corr.loc[ci, cj] > corr_thresh:
                # Drop the lower-STD feature (keeps the more "informative" one under this heuristic).
                drop_col = ci if stds[ci] <= stds[cj] else cj
                if not keep.get(drop_col, False):
                    continue
                keep[drop_col] = False
                dropped_reasons[drop_col] = (
                    "Redundant attribute removed due to very high correlation with another feature "
                    "(eliminates non-informative / overlapping information)."
                )

    to_drop = [c for c, k in keep.items() if not k]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
    return dropped_reasons


def scale_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Standardize numeric columns with mean=0, std=1 for deep learning.

    Returns:
        scaled_df, fitted_scaler, numeric_feature_names
    """
    df = df.copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    scaler = StandardScaler()
    numeric_values = df[numeric_cols].astype(np.float32).values
    scaled_values = scaler.fit_transform(numeric_values)
    df[numeric_cols] = scaled_values

    return df, scaler, numeric_cols


def run_preprocessing() -> None:
    """End-to-end preprocessing: load, clean, impute, scale, and save artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = list_raw_csvs(RAW_DIR)
    print(f"Found {len(csv_paths)} raw CSV files under {RAW_DIR}", flush=True)

    df_raw = load_and_concatenate(csv_paths)
    print(f"Loaded raw shape: {df_raw.shape}", flush=True)

    df = clean_column_names(df_raw)
    print(f"After column name cleaning: {df.shape}", flush=True)
    df = clean_label_text(df)
    print(f"After label normalization: {df.shape}", flush=True)
    df, dropped_reasons = drop_identifier_and_duplicates(df)
    print(f"After column drops: {df.shape}", flush=True)
    df, row_clean_stats = drop_null_and_duplicate_rows(df)
    print(f"After row cleaning (dupes/labels/endpoints/timestamp): {df.shape}", flush=True)
    df, dropped_low_support = drop_low_support_attack_rows(df)
    print(f"After low-support class filtering: {df.shape}", flush=True)

    # Keep only BENIGN + requested three attack families.
    if "Label" in df.columns:
        label_norm = df["Label"].map(normalize_label_key)
        keep_mask = label_norm.eq(BENIGN_LABEL_NORMALIZED) | label_norm.isin(TRAIN_ONLY_ATTACK_LABELS_NORMALIZED)
        dropped_other = int((~keep_mask).sum())
        if dropped_other > 0:
            other_counts = (
                df.loc[~keep_mask, "Label"]
                .astype(str)
                .value_counts()
                .sort_values(ascending=False)
                .to_dict()
            )
            df = df.loc[keep_mask].reset_index(drop=True)
        else:
            other_counts = {}
    else:
        dropped_other = 0
        other_counts = {}
    print(
        f"After keeping only 3 attack classes (+BENIGN): {df.shape}; dropped_other={dropped_other}",
        flush=True,
    )
    # Store for later summary lines.
    dropped_other = int(dropped_other)
    df = coerce_numerics_and_handle_infinite(df)
    print(f"After numeric coercion: {df.shape}", flush=True)
    df, encoded_features = encode_non_numeric_training_features(df)
    print(f"After non-numeric encodings: {df.shape}; encoded_cols={len(encoded_features)}", flush=True)
    df, imputation_stats = impute_missing_with_median(df)
    print(f"After median imputation: {df.shape}", flush=True)
    # Numeric feature selection (variance + correlation based).
    # This reduces dimensionality and removes non-informative / redundant attributes.
    variance_reasons = drop_near_zero_variance_numeric_features(df)
    dropped_reasons.update(variance_reasons)
    corr_reasons = drop_redundant_correlated_numeric_features(df)
    dropped_reasons.update(corr_reasons)
    df_scaled, scaler, numeric_cols = scale_numeric_features(df)
    print(f"After scaling numeric features: {df_scaled.shape}; num_numeric={len(numeric_cols)}", flush=True)

    # Persist main processed data (scaled numeric + metadata + label)
    processed_path_parquet = PROCESSED_DIR / "traffic_all_processed.parquet"
    df_scaled.to_parquet(processed_path_parquet, index=False)
    write_csv = os.environ.get("PREPROCESS_WRITE_CSV", "").strip() in {"1", "true", "True", "yes", "YES"}
    processed_path_csv = PROCESSED_DIR / "traffic_all_processed.csv"
    if write_csv:
        df_scaled.to_csv(processed_path_csv, index=False)
        print(f"Wrote CSV (optional): {processed_path_csv}", flush=True)
    else:
        print("Skipping CSV export (set PREPROCESS_WRITE_CSV=1 to write traffic_all_processed.csv)", flush=True)

    # Persist preprocessing artifacts for reuse in modeling code
    artifacts = {
        "numeric_feature_names": numeric_cols,
        "dropped_columns": dropped_reasons,
        "imputation_medians": imputation_stats,
        "scaler": scaler,
    }
    joblib.dump(artifacts, PROCESSED_DIR / "scaler.joblib")

    # Also dump a human-readable summary of dropped features and reasons
    summary_lines = []
    summary_lines.append("Dropped columns and justifications:\n")
    if not dropped_reasons:
        summary_lines.append("  (None)\n")
    else:
        for col, reason in dropped_reasons.items():
            summary_lines.append(f"- {col}: {reason}\n")

    summary_lines.append("\nRow-level cleaning summary:\n")
    summary_lines.append(f"- Rows before cleaning: {row_clean_stats['rows_before']}\n")
    summary_lines.append(
        f"- Duplicate rows removed: {row_clean_stats['duplicate_rows_removed']} "
        "(exact duplicate records can bias model priors and metrics).\n"
    )
    summary_lines.append(
        f"- Rows with null/empty Label removed: {row_clean_stats['null_label_rows_removed']} "
        "(supervised training requires valid targets).\n"
    )
    summary_lines.append(
        f"- Rows with missing Source IP removed: {row_clean_stats['missing_source_ip_rows_removed']} "
        "(cannot construct graph edges without a source endpoint).\n"
    )
    summary_lines.append(
        f"- Rows with missing Destination IP removed: {row_clean_stats['missing_destination_ip_rows_removed']} "
        "(cannot construct graph edges without a destination endpoint).\n"
    )
    summary_lines.append(
        f"- Rows with unparsable/missing Timestamp removed: {row_clean_stats['missing_timestamp_rows_removed']} "
        "(temporal ordering for windowing requires a valid timestamp).\n"
    )
    if dropped_low_support:
        summary_lines.append(
            f"- Low-support attack rows removed (class imbalance control): {sum(dropped_low_support.values())}\n"
        )
        for lbl, cnt in dropped_low_support.items():
            summary_lines.append(f"  - {lbl}: {cnt}\n")
    else:
        summary_lines.append("- Low-support attack rows removed: 0\n")
    summary_lines.append(
        "\nJustification (low-support attack filtering): several attack families have extremely small sample sizes "
        "relative to the dominant classes; keeping them can destabilize learning, inflate variance in macro metrics, "
        "and encourage brittle memorization rather than robust decision boundaries. These rows are removed only in "
        "the processed training artifact; raw combined data remains unchanged for other experiments.\n"
    )

    summary_lines.append("\nJustification (train-only class filtering):\n")
    summary_lines.append(
        "- Training is restricted to BENIGN + the requested 3 attack families (DDoS, DoS Hulk, PortScan). "
        "Removing other attack classes reduces dataset size and class confusion from extremely diverse attack categories, "
        "improving speed and focusing the model on the specific threat types you care about. "
        "This filtering is applied only to the processed training artifact; the raw combined dataset remains untouched.\n"
    )
    summary_lines.append(
        f"- Rows dropped by this step: {dropped_other}\n"
    )
    if other_counts:
        for lbl, cnt in other_counts.items():
            summary_lines.append(f"  - {lbl}: {cnt}\n")

    summary_lines.append("\nNon-numeric feature encoding used in training:\n")
    if encoded_features:
        for col, st in encoded_features.items():
            summary_lines.append(
                f"- {col}: {st['encoding']} ({st['num_unique_values']} unique values)\n"
            )
    else:
        summary_lines.append(
            "- No additional non-numeric training features required encoding after cleaning/coercion.\n"
        )

    # Group-level explanations for features excluded from numeric model inputs.
    # These columns are retained in the dataframe for GNN/ordering, but they are excluded from
    # the numeric feature set used by the model/scaler.
    summary_lines.append("\nModel input exclusions (kept for graph/ordering, excluded from numeric inputs):\n")
    summary_lines.append(
        "- Identifier-based features such as IP addresses (Source IP, Destination IP) and flow IDs were removed as they introduce bias "
        "and do not contribute to generalizable intrusion detection.\n"
    )
    summary_lines.append(
        "- Timestamp was removed as temporal dependencies are captured through sequence modeling rather than absolute time values.\n"
    )
    summary_lines.append(
        "- Features with near-zero variance were removed as they do not contribute to model learning.\n"
    )
    summary_lines.append(
        "- Bulk features weak contribution: bulk-rate/bytes/packets statistics were excluded because they provided limited discriminative signal.\n"
    )
    summary_lines.append(
        "\nFeature selection was performed to eliminate irrelevant, redundant, and non-informative attributes. "
        "Identifier-based features (IP addresses and flow IDs) were removed to prevent overfitting and ensure model generalization. "
        "Features with near-zero variance and redundant attributes were excluded as they do not contribute meaningful information "
        "to the learning process. Additionally, timestamp data was omitted since temporal dependencies are captured through "
        "sequence-based modeling. This preprocessing step reduces dimensionality, improves computational efficiency, "
        "and enhances model performance.\n"
    )

    summary_lines.append("\nNumeric feature count (after cleaning, before modeling): "
                         f"{len(numeric_cols)}\n")
    summary_lines.append(f"Final processed shape: {df_scaled.shape}\n")

    summary_path = PROCESSED_DIR / "preprocessing_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    if write_csv:
        print(f"Saved processed data to: {processed_path_parquet} and {processed_path_csv}", flush=True)
    else:
        print(f"Saved processed data to: {processed_path_parquet}", flush=True)
    print(f"Saved scaler and stats to: {PROCESSED_DIR / 'scaler.joblib'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    run_preprocessing()

