"""
Print row counts for cleaned CICIDS 2018 artifacts (labels only — no plots).

Reads only label columns where possible; uses chunked CSV reads for large files.
Run from project root: python summarize_cicids_2018_labels.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from preprocess_cicids import (
    BENIGN_LABEL_NORMALIZED,
    TRAIN_ONLY_ATTACK_LABELS,
    normalize_label_key,
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"

CHUNK_ROWS = 500_000


def _add_counter(c: Counter, series: pd.Series) -> None:
    vc = series.value_counts(dropna=False)
    for k, v in vc.items():
        c[str(k)] += int(v)


def count_from_csv(path: Path, label_col: str) -> tuple[int, Counter]:
    c: Counter = Counter()
    total = 0
    reader = pd.read_csv(
        path,
        usecols=[label_col],
        chunksize=CHUNK_ROWS,
        low_memory=False,
        on_bad_lines="skip",
        engine="python",
    )
    for chunk in reader:
        total += len(chunk)
        _add_counter(c, chunk[label_col])
    return total, c


def count_from_parquet(path: Path, label_col: str) -> tuple[int, Counter]:
    last_err: Optional[Exception] = None
    for engine in (None, "fastparquet"):
        try:
            kw: dict = {"columns": [label_col]}
            if engine:
                kw["engine"] = engine
            df = pd.read_parquet(path, **kw)
            total = len(df)
            c: Counter = Counter()
            _add_counter(c, df[label_col])
            return total, c
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("read_parquet failed")


def load_label_counts(path: Path, label_col: str = "Label") -> tuple[int, Counter]:
    if not path.exists():
        return 0, Counter()
    if path.suffix.lower() == ".parquet":
        try:
            return count_from_parquet(path, label_col)
        except Exception as e:
            print(f"  (parquet read failed, trying CSV if present: {e})")
            csv_alt = path.with_suffix(".csv")
            if csv_alt.exists():
                try:
                    return count_from_csv(csv_alt, label_col)
                except Exception as e2:
                    print(f"  (CSV fallback failed: {e2})")
            return 0, Counter()
    if path.suffix.lower() == ".csv":
        try:
            return count_from_csv(path, label_col)
        except Exception as e:
            print(f"  (CSV read failed for {path.name}: {e})")
            return 0, Counter()
    return 0, Counter()


def binary_from_multiclass_labels(counter: Counter) -> tuple[int, int]:
    benign = sum(
        v for k, v in counter.items() if normalize_label_key(str(k)) == BENIGN_LABEL_NORMALIZED
    )
    total = sum(counter.values())
    return benign, total - benign


def print_section(title: str, path: Path, label_col: str = "Label") -> Counter:
    print(title)
    print(f"  file: {path}")
    if not path.exists():
        print("  (file not found)\n")
        return Counter()
    total, counter = load_label_counts(path, label_col=label_col)
    if total == 0:
        print(
            "  (no rows read - parquet may be unreadable with pyarrow; "
            "try: pip install fastparquet, or regenerate CSV from preprocess_cicids_2018.py)\n"
        )
        return Counter()
    print(f"  total rows: {total:,}")
    b, atk = binary_from_multiclass_labels(counter)
    print(f"  binary: BENIGN={b:,}  ATTACK(non-BENIGN)={atk:,}")
    print("  per-label:")
    for lbl in sorted(counter.keys(), key=lambda x: (-counter[x], str(x))):
        print(f"    {lbl!r}: {counter[lbl]:,}")
    print()
    return counter


def main() -> None:
    print("CICIDS 2018 cleaned label summary\n")

    # Version 1 & 2 from preprocess_cicids_2018.py (multi-class pipeline, Label = original class name)
    pq_v1 = PROCESSED / "traffic_2018_all_classes.parquet"
    csv_v1 = PROCESSED / "traffic_2018_all_classes.csv"
    pq_v2 = PROCESSED / "traffic_2018_train_classes.parquet"
    csv_v2 = PROCESSED / "traffic_2018_train_classes.csv"

    v1 = pq_v1 if pq_v1.exists() else csv_v1
    print_section("Version 1 (all attack classes, multi pipeline)", v1)

    v2 = pq_v2 if pq_v2.exists() else csv_v2
    c2 = print_section(
        "Version 2 (BENIGN + DDoS + DoS Hulk + PortScan only, multi pipeline)",
        v2,
    )
    if c2:
        print("  multiclass targets (same normalization as training):")
        nb = sum(
            v for k, v in c2.items() if normalize_label_key(str(k)) == BENIGN_LABEL_NORMALIZED
        )
        print(f"    BENIGN: {nb:,}")
        for lbl in sorted(TRAIN_ONLY_ATTACK_LABELS):
            key = normalize_label_key(lbl)
            n = sum(v for k, v in c2.items() if normalize_label_key(str(k)) == key)
            print(f"    {lbl}: {n:,}")
        print()

    # Binary pipeline output (Label is BENIGN / ATTACK)
    bin_csv = PROCESSED / "traffic_2018_binary_all_classes.csv"
    bin_pq = PROCESSED / "traffic_2018_binary_all_classes.parquet"
    bpath = bin_pq if bin_pq.exists() else bin_csv
    print_section("Binary pipeline (all classes rolled to BENIGN vs ATTACK)", bpath)

    # Optional: original attack names before collapse
    lo_path = bpath
    if lo_path.exists():
        if lo_path.suffix.lower() == ".parquet":
            try:
                df = pd.read_parquet(lo_path, columns=["LabelOriginal"])
                print("Binary pipeline - LabelOriginal (fine-grained) top counts:")
                vc = df["LabelOriginal"].value_counts()
                for lbl, cnt in vc.head(30).items():
                    print(f"  {lbl!r}: {int(cnt):,}")
                if len(vc) > 30:
                    print(f"  ... ({len(vc) - 30} more labels)")
                print()
            except Exception:
                pass
        else:
            try:
                c_orig: Counter = Counter()
                tot = 0
                for chunk in pd.read_csv(
                    lo_path,
                    usecols=["LabelOriginal"],
                    chunksize=CHUNK_ROWS,
                    low_memory=False,
                    on_bad_lines="skip",
                    engine="python",
                ):
                    tot += len(chunk)
                    _add_counter(c_orig, chunk["LabelOriginal"])
                print("Binary pipeline - LabelOriginal (fine-grained), all labels:")
                for lbl in sorted(c_orig.keys(), key=lambda x: (-c_orig[x], str(x))):
                    print(f"  {lbl!r}: {c_orig[lbl]:,}")
                print()
            except Exception as e:
                print(f"  (LabelOriginal column not available or read failed: {e})\n")


if __name__ == "__main__":
    main()
