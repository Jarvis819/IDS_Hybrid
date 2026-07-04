"""
Generate a custom synthetic dataset for zero-day evaluation.

Creates flows with feature distributions similar to CICIDS but introduces
controlled "zero-day" attack patterns not seen during training.
Use this to rigorously test model generalization.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
SYNTHETIC_DIR = ROOT / "synthetic_data"


def sample_from_empirical(
    df: pd.DataFrame,
    feat_cols: List[str],
    n: int,
    label: str,
    shift_mean: float = 0.0,
    scale_std: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample flows from empirical distribution with optional shift (for zero-day)."""
    rng = np.random.default_rng(seed)
    sub = df[df["Label"] == label] if "Label" in df.columns else df
    if len(sub) == 0:
        sub = df
    idx = rng.choice(len(sub), size=min(n, len(sub)), replace=True)
    sample = sub.iloc[idx][feat_cols].copy()
    sample = sample.astype(np.float64)
    if shift_mean != 0 or scale_std != 1:
        sample = sample * scale_std + shift_mean
    return sample


def inject_zero_day_pattern(
    base: np.ndarray,
    feat_indices: List[int],
    multipliers: List[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject anomalous pattern: scale selected features (simulates unknown attack)."""
    out = base.copy()
    for i, idx in enumerate(feat_indices):
        if 0 <= idx < out.shape[1]:
            m = multipliers[i] if i < len(multipliers) else multipliers[-1]
            out[:, idx] *= m
    return out


def generate_synthetic(
    processed_dir: Path,
    output_dir: Path,
    n_benign: int = 10_000,
    n_known_attack: int = 5_000,
    n_zero_day: int = 3_000,
    zero_day_label: str = "ZERO_DAY",
    seed: int = 42,
) -> None:
    """Generate synthetic dataset with benign, known-attack, and zero-day flows."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "traffic_all_processed.parquet")
    artifacts = joblib.load(processed_dir / "scaler.joblib")
    feat_cols = artifacts["numeric_feature_names"]

    labels = df["Label"].dropna().unique()
    known = [l for l in labels if str(l) != "BENIGN"]
    if len(known) == 0:
        known = ["BENIGN"]

    # 1) Benign: sample from real benign
    benign_sub = df[df["Label"] == "BENIGN"]
    if len(benign_sub) < n_benign:
        idx = rng.choice(len(benign_sub), size=n_benign, replace=True)
    else:
        idx = rng.choice(len(benign_sub), size=n_benign, replace=False)
    benign = benign_sub.iloc[idx].copy()
    benign["Label"] = "BENIGN"

    # 2) Known attack: sample from real attacks
    known_flows = []
    per_class = max(1, n_known_attack // len(known))
    for lbl in known[:5]:
        sub = df[df["Label"] == lbl]
        if len(sub) > 0:
            n_take = min(per_class, len(sub))
            idx = rng.choice(len(sub), size=n_take, replace=len(sub) < n_take)
            known_flows.append(sub.iloc[idx])
    if known_flows:
        known_df = pd.concat(known_flows, ignore_index=True)
    else:
        known_df = benign_sub.sample(n=min(n_known_attack, len(benign_sub)), random_state=seed).copy()
        known_df["Label"] = "DoS"

    # 3) Zero-day: sample from benign/attack then inject unseen pattern
    base_for_zd = df[df["Label"] != "BENIGN"]
    if len(base_for_zd) < 100:
        base_for_zd = df
    idx = rng.choice(len(base_for_zd), size=min(n_zero_day, len(base_for_zd)), replace=True)
    zd_base = base_for_zd.iloc[idx][feat_cols].copy().values

    # Inject pattern: boost Flow Bytes/s, Flow Packets/s (indices from CICIDS schema)
    # Flow Bytes/s ~20, Flow Packets/s ~21, Fwd Packets/s ~42
    zd_feat_idx = [20, 21, 42] if len(feat_cols) > 42 else [0, 1, 2]
    zd_mult = [2.5, 3.0, 1.8]
    zd_values = inject_zero_day_pattern(zd_base, zd_feat_idx, zd_mult, rng)

    zd_df = pd.DataFrame(zd_values, columns=feat_cols)
    zd_df["Label"] = zero_day_label
    zd_df["Source IP"] = "10.0.0." + pd.Series(rng.integers(1, 254, len(zd_df))).astype(str)
    zd_df["Destination IP"] = "192.168.1." + pd.Series(rng.integers(1, 254, len(zd_df))).astype(str)
    zd_df["Source Port"] = rng.integers(1024, 65535, len(zd_df))
    zd_df["Destination Port"] = rng.choice([80, 443, 22], len(zd_df))
    zd_df["Protocol"] = 6
    zd_df["Timestamp"] = pd.date_range("2024-01-15", periods=len(zd_df), freq="s").astype(str)
    zd_df["SourceFile"] = "synthetic_zero_day.csv"

    # Add metadata to benign and known
    for d in [benign, known_df]:
        if "Source IP" not in d.columns:
            d["Source IP"] = "192.168.10." + pd.Series(rng.integers(1, 50, len(d))).astype(str)
        if "Destination IP" not in d.columns:
            d["Destination IP"] = "10.0.0." + pd.Series(rng.integers(1, 254, len(d))).astype(str)
        if "SourceFile" not in d.columns:
            d["SourceFile"] = "synthetic.csv"

    out = pd.concat([benign, known_df, zd_df], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    if "Timestamp" in out.columns:
        out["Timestamp"] = out["Timestamp"].astype(str)

    out_path = output_dir / "synthetic_zeroday_eval.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(output_dir / "synthetic_zeroday_eval.csv", index=False)

    meta = {
        "n_benign": int(benign.shape[0]),
        "n_known_attack": int(known_df.shape[0]),
        "n_zero_day": int(zd_df.shape[0]),
        "zero_day_label": zero_day_label,
        "feat_cols": feat_cols,
    }
    joblib.dump(meta, output_dir / "synthetic_meta.joblib")

    print(f"Generated {out_path}")
    print(f"  Benign: {meta['n_benign']}, Known: {meta['n_known_attack']}, Zero-day: {meta['n_zero_day']}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for zero-day evaluation")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--n-benign", type=int, default=10_000)
    parser.add_argument("--n-known", type=int, default=5_000)
    parser.add_argument("--n-zero-day", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic(
        args.processed_dir,
        args.output_dir,
        n_benign=args.n_benign,
        n_known_attack=args.n_known,
        n_zero_day=args.n_zero_day,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
