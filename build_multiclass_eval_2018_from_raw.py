"""
Build a *balanced* CICIDS 2018 evaluation dataset for Model A (multi-attack),
directly from raw CICIDS 2018 CSVs, in a memory-safe chunked pipeline.

Keeps:
- BENIGN: downsample to target (default 1,000,000)
- DDoS:   map {HOIC, LOIC-HTTP, LOIC-UDP} -> 'DDoS' (keep all)
- DoS Hulk: map {DoS attacks-Hulk} -> 'DoS Hulk' (keep all)
- PortScan: none in CICIDS 2018 (kept as 0 rows)

Writes:
- processed/traffic_2018_multiclass_eval_balanced.parquet
- processed/traffic_2018_multiclass_eval_balanced.csv

This script also aligns numeric features to Model A preprocessing artifacts:
- loads processed/scaler.joblib
- imputes using training medians
- scales using training scaler
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd

from preprocess_cicids import (
    assign_synthetic_flow_endpoints_if_missing,
    clean_column_names,
    clean_label_text,
    drop_identifier_and_duplicates,
    coerce_numerics_and_handle_infinite,
    canonicalize_label_string,
    normalize_label_key,
    BENIGN_LABEL_NORMALIZED,
)


ROOT = Path(__file__).resolve().parent
RAW_2018_DIR = ROOT / "CIC-IDS-2018-Dataset"
PROCESSED = ROOT / "processed"

CHUNK = int(os.environ.get("RAW_CHUNK_ROWS", "200000"))


def list_csvs() -> List[Path]:
    return sorted(p for p in RAW_2018_DIR.glob("*.csv") if not p.name.startswith(".~lock"))


def map_label_to_model_a(label: str) -> str:
    s = canonicalize_label_string(label)
    k = normalize_label_key(s)
    if k == BENIGN_LABEL_NORMALIZED:
        return "BENIGN"

    if k in {
        normalize_label_key("DDOS attack-HOIC"),
        normalize_label_key("DDoS attacks-LOIC-HTTP"),
        normalize_label_key("DDOS attack-LOIC-UDP"),
    }:
        return "DDoS"

    if k == normalize_label_key("DoS attacks-Hulk"):
        return "DoS Hulk"

    return "__DROP__"


def load_model_a_artifacts() -> Tuple[List[str], Dict[str, float], Any]:
    art = joblib.load(PROCESSED / "scaler.joblib")
    feat_cols = list(art["numeric_feature_names"])
    med = dict(art.get("imputation_medians", {}))
    scaler = art["scaler"]
    return feat_cols, med, scaler


def count_benign_total(files: List[Path]) -> int:
    total = 0
    for p in files:
        for chunk in pd.read_csv(p, usecols=["Label"], chunksize=CHUNK, low_memory=False):
            # normalize like training
            lab = chunk["Label"].astype(str).map(canonicalize_label_string)
            total += int(lab.map(normalize_label_key).eq(BENIGN_LABEL_NORMALIZED).sum())
    return total


def main() -> int:
    target_benign = int(os.environ.get("EVAL_TARGET_BENIGN", "1000000"))
    seed = int(os.environ.get("EVAL_SAMPLE_SEED", "42"))
    rng = np.random.default_rng(seed)

    files = list_csvs()
    if not files:
        raise FileNotFoundError(f"No CSVs found in {RAW_2018_DIR}")

    feat_cols, medians, scaler = load_model_a_artifacts()
    keep_meta = ["Label", "Source IP", "Destination IP", "Timestamp", "SourceFile"]
    keep_cols = keep_meta + feat_cols

    print(f"[mc-2018] files={len(files)} chunk_rows={CHUNK}", flush=True)
    print(f"[mc-2018] target_benign={target_benign:,} seed={seed}", flush=True)

    print("[mc-2018] Pass 1: counting total BENIGN rows in raw 2018...", flush=True)
    benign_total = count_benign_total(files)
    if benign_total <= 0:
        raise ValueError("No BENIGN rows found in raw 2018 Label column.")
    print(f"[mc-2018] raw BENIGN total={benign_total:,}", flush=True)

    out_pq = PROCESSED / "traffic_2018_multiclass_eval_balanced.parquet"
    out_csv = PROCESSED / "traffic_2018_multiclass_eval_balanced.csv"
    for f in (out_pq, out_csv):
        if f.exists():
            f.unlink()

    # We'll append to CSV in chunks; parquet we write at the end from concatenated pieces on disk
    # using a lightweight approach: collect chunk outputs as parquet parts and then concat.
    parts_dir = PROCESSED / "tmp_mc2018_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("part_*.parquet"):
        old.unlink()

    benign_remaining = benign_total
    benign_needed = target_benign
    part_idx = 0

    kept_counts = {"BENIGN": 0, "DDoS": 0, "DoS Hulk": 0}

    print("[mc-2018] Pass 2: streaming raw 2018 -> mapped+scaled -> write parts...", flush=True)
    for fi, path in enumerate(files, 1):
        print(f"[mc-2018] file {fi}/{len(files)} {path.name}", flush=True)
        # Read full rows but only required columns if they exist; raw files might have extra cols.
        # We'll read everything then drop early to keep implementation robust to schema drift.
        reader = pd.read_csv(path, chunksize=CHUNK, low_memory=False)
        for chunk in reader:
            chunk["SourceFile"] = path.name
            chunk = clean_column_names(chunk)
            chunk = clean_label_text(chunk)
            chunk, _ = drop_identifier_and_duplicates(chunk)
            chunk = coerce_numerics_and_handle_infinite(chunk)

            # CICIDS 2018 raw files commonly use Src/Dst IP column names.
            # Preserve real IPs for graph construction by mapping aliases.
            if "Src IP" in chunk.columns and "Source IP" not in chunk.columns:
                chunk["Source IP"] = chunk["Src IP"]
            if "Dst IP" in chunk.columns and "Destination IP" not in chunk.columns:
                chunk["Destination IP"] = chunk["Dst IP"]

            # Ensure required columns exist
            for c in feat_cols:
                if c not in chunk.columns:
                    chunk[c] = 0.0
            if "Timestamp" not in chunk.columns:
                chunk["Timestamp"] = pd.Timestamp.utcnow()
            # TrafficForML exports omit IPs; synthesize endpoints for graphs (see preprocess_cicids).
            chunk = assign_synthetic_flow_endpoints_if_missing(chunk)
            if "Label" not in chunk.columns:
                chunk["Label"] = "BENIGN"

            # Map labels to model-A space and drop unrelated attacks
            mapped = chunk["Label"].astype(str).map(map_label_to_model_a)
            chunk = chunk.loc[mapped != "__DROP__"].copy()
            chunk["Label"] = mapped.loc[mapped != "__DROP__"].values
            if chunk.empty:
                continue

            # Downsample BENIGN: keep all attacks, sample BENIGN up to the target.
            benign_mask = chunk["Label"].map(normalize_label_key).eq(BENIGN_LABEL_NORMALIZED)
            nb = int(benign_mask.sum())
            if nb > 0:
                k = int(min(benign_needed, nb))
                if k < nb:
                    benign_idx = np.flatnonzero(benign_mask.to_numpy())
                    take = rng.choice(benign_idx, size=k, replace=False) if k > 0 else np.array([], dtype=int)
                    keep_mask = np.zeros(len(chunk), dtype=bool)
                    keep_mask[take] = True
                    keep_mask |= (~benign_mask.to_numpy())
                    chunk = chunk.loc[keep_mask].copy()
                    nb = k

                benign_needed -= nb
                benign_remaining -= nb
                benign_remaining = max(0, benign_remaining)
            else:
                # still update remaining count if chunk had benign rows originally? nb=0
                pass

            # Impute + scale numeric features with training stats
            if chunk.empty:
                continue
            for c in feat_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(float(medians.get(c, 0.0)))
            chunk[feat_cols] = scaler.transform(chunk[feat_cols].astype(np.float32).values)

            chunk = chunk[keep_cols]

            # Write part parquet (compressed by engine defaults)
            part_path = parts_dir / f"part_{part_idx:05d}.parquet"
            chunk.to_parquet(part_path, index=False)
            part_idx += 1

            # Count kept labels
            vc = chunk["Label"].value_counts()
            for k, v in vc.items():
                kept_counts[str(k)] = kept_counts.get(str(k), 0) + int(v)

            if benign_needed <= 0 and fi >= 1:
                # We can continue to capture all attacks; benign already at target.
                pass

    # Combine parts
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if not part_files:
        raise RuntimeError("No output parts written; check label mapping.")
    print(f"[mc-2018] Combining {len(part_files)} parquet parts...", flush=True)
    df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)

    # Hard cap benign if we overshot slightly
    benign_mask = df["Label"].map(normalize_label_key).eq(BENIGN_LABEL_NORMALIZED)
    df_b = df.loc[benign_mask]
    df_a = df.loc[~benign_mask]
    if len(df_b) > target_benign:
        df_b = df_b.sample(n=target_benign, random_state=seed)
    df = pd.concat([df_b, df_a], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df.to_parquet(out_pq, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[mc-2018] wrote {out_pq.name} rows={len(df):,}", flush=True)
    print(df["Label"].value_counts().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

