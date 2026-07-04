"""
Build small RAW CICIDS CSVs for faculty demos (Predictions tab + preprocessing pipeline).

Writes:
  sample_data/demo_cicids_2017_raw.csv   — 1,000 BENIGN + 4,000 mixed attack rows (2017 TrafficLabelling)
  sample_data/demo_cicids_2018_raw.csv   — 1,000 BENIGN + 4,000 mixed attack rows (2018 TrafficForML)

Both years use the same sampling strategy: scan CSVs in order, take up to N benign and M non-benign
rows, then shuffle with a fixed seed for a reproducible demo file.

No scaling or training artifacts; uploads go through the same cleaning path as real traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_data"
RAW_2017 = ROOT / "TrafficLabelling"
RAW_2018 = ROOT / "CIC-IDS-2018-Dataset"


def _is_benign(label: pd.Series) -> pd.Series:
    return label.astype(str).str.strip().str.upper().eq("BENIGN")


def _read_chunks(path: Path, chunksize: int):
    encodings = ("utf-8", "latin-1")
    last_err = None
    for enc in encodings:
        try:
            for chunk in pd.read_csv(
                path,
                chunksize=chunksize,
                low_memory=False,
                encoding=enc,
                on_bad_lines="skip",
            ):
                yield chunk
            return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"Failed to read {path}")


def collect_from_dir(
    raw_dir: Path,
    n_benign: int,
    n_attack: int,
    chunksize: int,
    seed: int = 42,
) -> pd.DataFrame:
    benign_parts: list[pd.DataFrame] = []
    attack_parts: list[pd.DataFrame] = []
    bc, ac = 0, 0

    paths = sorted(p for p in raw_dir.glob("*.csv") if not p.name.startswith(".~"))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {raw_dir}")

    for path in paths:
        if bc >= n_benign and ac >= n_attack:
            break
        for chunk in _read_chunks(path, chunksize):
            chunk = chunk.copy()
            chunk.columns = chunk.columns.str.strip()
            if "Label" not in chunk.columns:
                continue
            chunk["SourceFile"] = path.name
            ben = _is_benign(chunk["Label"])
            b = chunk.loc[ben]
            a = chunk.loc[~ben]

            if bc < n_benign and not b.empty:
                need = n_benign - bc
                take = b.iloc[: min(need, len(b))]
                benign_parts.append(take)
                bc += len(take)
            if ac < n_attack and not a.empty:
                need = n_attack - ac
                take = a.iloc[: min(need, len(a))]
                attack_parts.append(take)
                ac += len(take)
            if bc >= n_benign and ac >= n_attack:
                break

    if bc < n_benign or ac < n_attack:
        raise RuntimeError(
            f"Could not collect enough rows from {raw_dir}. "
            f"Got BENIGN={bc}/{n_benign}, ATTACK={ac}/{n_attack}."
        )

    out = pd.concat(benign_parts + attack_parts, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", type=int, default=1000, help="2017 benign count")
    ap.add_argument("--attack", type=int, default=4000, help="2017 mixed-attack count")
    ap.add_argument("--18-benign", type=int, default=1000, dest="b18", help="2018 benign count")
    ap.add_argument("--18-attack", type=int, default=4000, dest="a18", help="2018 mixed-attack count")
    ap.add_argument("--chunksize", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    df17 = collect_from_dir(RAW_2017, args.benign, args.attack, args.chunksize, seed=args.seed)
    p17 = OUT / "demo_cicids_2017_raw.csv"
    df17.to_csv(p17, index=False)
    vc17 = _is_benign(df17["Label"]).map({True: "BENIGN", False: "ATTACK"})
    print(f"[raw-demo] {p17} rows={len(df17):,} cols={df17.shape[1]}\n{vc17.value_counts().to_string()}", flush=True)

    df18 = collect_from_dir(RAW_2018, args.b18, args.a18, args.chunksize, seed=args.seed)
    p18 = OUT / "demo_cicids_2018_raw.csv"
    df18.to_csv(p18, index=False)
    vc18 = _is_benign(df18["Label"]).map({True: "BENIGN", False: "ATTACK"})
    print(f"[raw-demo] {p18} rows={len(df18):,} cols={df18.shape[1]}\n{vc18.value_counts().to_string()}", flush=True)
    print("[raw-demo] 2018 Label (original strings, top 15):", flush=True)
    print(df18["Label"].astype(str).str.strip().value_counts().head(15).to_string(), flush=True)

    print("[raw-demo] Done. Upload these in Predictions (pipeline will preprocess).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
