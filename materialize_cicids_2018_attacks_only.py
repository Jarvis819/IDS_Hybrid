"""
One-time (or occasional) helper: stream each CICIDS 2018 TrafficForML CSV and write a
smaller copy that keeps all non-BENIGN flows and drops BENIGN rows.

Outputs go to:  CIC-IDS-2018-Dataset/_preprocessed/attacks_only/<same basename>.csv

preprocess_cicids_2018_binary.py automatically prefers those files over the originals
when present, so later binary preprocessing skips scanning multi-GB benign-majority days.

Re-run with --force after replacing/updating a raw CSV. Only the binary pipeline uses
this cache; multiclass / benign sampling still needs full raw files.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Set

import joblib
import pandas as pd
from tqdm import tqdm

from preprocess_cicids import (
    ROOT_DIR,
    BENIGN_LABEL_NORMALIZED,
    clean_column_names,
    clean_label_text,
    normalize_label_key,
)

RAW_2018_DIR = ROOT_DIR / "CIC-IDS-2018-Dataset"
PROCESSED_DIR = ROOT_DIR / "processed"
CACHE_DIR = RAW_2018_DIR / "_preprocessed" / "attacks_only"
CHUNK = 500_000


def _root_csvs() -> list[Path]:
    if not RAW_2018_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_2018_DIR}")
    return sorted(p for p in RAW_2018_DIR.glob("*.csv") if not p.name.startswith(".~lock"))


def _required_cols(feat_cols: list[str]) -> Set[str]:
    return set(feat_cols) | {
        "Label",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP",
    }


def _usecols_from_file(path: Path, required: Set[str]) -> list[str]:
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8", on_bad_lines="skip").columns)
    if "Label" not in header:
        try:
            header = list(pd.read_csv(path, nrows=0, encoding="latin-1", on_bad_lines="skip").columns)
        except Exception:
            pass
    avail = set(header)
    cols = [c for c in sorted(required) if c in avail]
    if "Label" not in cols and "Label" in avail:
        cols = ["Label"] + [c for c in cols if c != "Label"]
    if "Label" not in cols:
        raise ValueError(f"{path.name}: no Label column in CSV header")
    return cols


def materialize_one(
    src: Path,
    feat_cols: list[str],
    force: bool,
) -> tuple[int, int, Path]:
    required = _required_cols(feat_cols)
    out = CACHE_DIR / src.name
    if out.exists() and not force:
        return 0, 0, out

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    usecols = _usecols_from_file(src, required)
    encodings = ["utf-8", "latin-1"]

    wrote_header = False
    kept = 0
    scanned = 0

    def reader():
        for enc in encodings:
            try:
                return pd.read_csv(
                    src,
                    usecols=usecols,
                    chunksize=CHUNK,
                    low_memory=False,
                    encoding=enc,
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode {src}")

    # Rough progress: line count is expensive; tqdm tracks chunks only.
    r = reader()
    for chunk in tqdm(r, desc=src.name[:40], unit="chunk"):
        scanned += len(chunk)
        chunk = clean_column_names(chunk)
        chunk = clean_label_text(chunk)
        keys = chunk["Label"].astype(str).map(normalize_label_key)
        chunk = chunk.loc[~keys.eq(BENIGN_LABEL_NORMALIZED)].copy()
        if chunk.empty:
            continue
        kept += len(chunk)
        chunk.to_csv(out, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    return scanned, kept, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Write attack-only CICIDS 2018 CSV cache for binary preprocessing.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing attacks_only copies.")
    ap.add_argument(
        "--match",
        default="*",
        help="Glob matched against basenames (default: all), e.g. 'Thuesday*'",
    )
    args = ap.parse_args()

    art_path = PROCESSED_DIR / "scaler_binary.joblib"
    if not art_path.exists():
        raise FileNotFoundError(f"Need {art_path}; run preprocess_cicids_binary.py on 2017 data first.")
    feat_cols = list(joblib.load(art_path)["numeric_feature_names"])

    paths = [p for p in _root_csvs() if fnmatch.fnmatch(p.name, args.match)]
    if not paths:
        print(f"No CSVs matched --match {args.match!r}", flush=True)
        return 1

    print(f"[attacks-only] cache_dir={CACHE_DIR}", flush=True)
    print(f"[attacks-only] files={len(paths)} chunk_rows={CHUNK}", flush=True)

    for p in paths:
        out = CACHE_DIR / p.name
        if out.exists() and not args.force:
            print(f"[attacks-only] skip (exists): {p.name}; use --force to rebuild", flush=True)
            continue
        scanned, kept, outp = materialize_one(p, feat_cols, force=args.force)
        print(
            f"[attacks-only] {p.name}: scanned={scanned:,} kept_attacks={kept:,} -> {outp}",
            flush=True,
        )

    print("[attacks-only] done. Run: python preprocess_cicids_2018_binary.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
