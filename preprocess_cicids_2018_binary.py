import csv
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from preprocess_cicids import (
    ROOT_DIR,
    assign_synthetic_flow_endpoints_if_missing,
    clean_column_names,
    clean_label_text,
    drop_null_and_duplicate_rows,
    coerce_numerics_and_handle_infinite,
)


RAW_2018_DIR = ROOT_DIR / "CIC-IDS-2018-Dataset"
PROCESSED_DIR = ROOT_DIR / "processed"


def list_csvs(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing folder: {raw_dir}")
    cache_root = raw_dir / "_preprocessed" / "attacks_only"
    out: List[Path] = []
    for p in sorted(raw_dir.glob("*.csv")):
        if p.name.startswith(".~lock"):
            continue
        cached = cache_root / p.name
        out.append(cached if cached.exists() else p)
    return out


def load_one(path: Path, required_cols: set[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            low_memory=False,
            encoding="utf-8",
            usecols=lambda c: c in required_cols,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            low_memory=False,
            encoding="latin-1",
            usecols=lambda c: c in required_cols,
        )
    df["SourceFile"] = path.name
    return df


def run() -> None:
    art_path = PROCESSED_DIR / "scaler_binary.joblib"
    if not art_path.exists():
        raise FileNotFoundError(f"Binary artifacts missing: {art_path}. Run preprocess_cicids_binary.py first.")
    artifacts = joblib.load(art_path)
    feat_cols = list(artifacts["numeric_feature_names"])
    medians = artifacts["imputation_medians"]
    scaler = artifacts["scaler"]
    # Read only columns we actually use (major speed + memory improvement).
    required_cols = set(feat_cols) | {
        "Label",
        "Timestamp",
        "SourceFile",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP",
    }

    out_csv = PROCESSED_DIR / "traffic_2018_binary_all_classes_fresh.csv"
    if out_csv.exists():
        out_csv.unlink()

    wrote_header = False
    paths = list_csvs(RAW_2018_DIR)
    print(f"[2018-binary] files={len(paths)}", flush=True)
    for i, p in enumerate(paths, 1):
        print(f"[2018-binary] {i}/{len(paths)} {p.name}", flush=True)
        d = load_one(p, required_cols)
        d = clean_column_names(d)
        # CIC exports sometimes use short IP column names.
        if "Src IP" in d.columns and "Source IP" not in d.columns:
            d["Source IP"] = d["Src IP"]
        if "Dst IP" in d.columns and "Destination IP" not in d.columns:
            d["Destination IP"] = d["Dst IP"]
        d = clean_label_text(d)
        # Keep IP columns for graph construction in downstream HybridIDSDataset.
        # drop_identifier_and_duplicates() removes identifier fields (including IPs),
        # which destroys graph signal for 2018 evaluation.
        d, _ = drop_null_and_duplicate_rows(d)
        d = coerce_numerics_and_handle_infinite(d)
        # TrafficForML 2018 CSVs have no IP columns; pseudo-IPs keep graph construction non-degenerate.
        d = assign_synthetic_flow_endpoints_if_missing(d)

        if "Label" not in d.columns:
            d["Label"] = "BENIGN"
        d["LabelOriginal"] = d["Label"].astype(str).str.strip()
        d["Label"] = np.where(d["LabelOriginal"].str.upper() == "BENIGN", "BENIGN", "ATTACK")

        for c in feat_cols:
            if c not in d.columns:
                d[c] = 0.0
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(float(medians.get(c, 0.0)))

        d[feat_cols] = scaler.transform(d[feat_cols].astype(np.float32).values)
        keep_meta = {"Label", "LabelOriginal", "Source IP", "Destination IP", "Timestamp", "SourceFile"}
        drop_cols = [c for c in d.columns if c not in keep_meta and c not in feat_cols]
        if drop_cols:
            d = d.drop(columns=drop_cols)
        # Append to CSV incrementally to avoid large in-memory concatenation.
        # Quote non-numeric fields so commas in timestamps/labels cannot break parsers.
        d.to_csv(
            out_csv,
            mode="a",
            header=not wrote_header,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        wrote_header = True
        print(f"[2018-binary] appended {len(d):,} rows from {p.name}", flush=True)

    final_csv = PROCESSED_DIR / "traffic_2018_binary_all_classes.csv"
    try:
        out_csv.replace(final_csv)
        print(f"[2018-binary] saved {final_csv}", flush=True)
    except Exception:
        print(f"[2018-binary] saved {out_csv} (could not overwrite {final_csv})", flush=True)


if __name__ == "__main__":
    run()

