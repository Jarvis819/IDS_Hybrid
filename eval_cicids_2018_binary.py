"""Evaluate binary BENIGN-vs-ATTACK model on CICIDS 2018 all-classes data.

Progress: tqdm over DataLoader batches. Forward is batched on GPU when available.
Graph construction per window uses HybridIDSDataset (dynamic path). On Windows, DataLoader
workers must stay 0 (pickle limit). On Linux/Mac, --num-workers may help.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_loader import HybridIDSDataset, collate_hybrid
from src.model import HybridIDSModel
from src.config import SEQ_LEN


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "outputs" / "models"


def effective_eval_num_workers(requested: int) -> int:
    if sys.platform == "win32" and int(requested) > 0:
        print(
            "[eval] num_workers>0 is not supported on Windows for this eval (dataset is not picklable). "
            "Using num_workers=0.",
            flush=True,
        )
        return 0
    return max(0, int(requested))


def read_binary_eval_table(path: Path) -> pd.DataFrame:
    """Prefer parquet if present; CSV may have malformed rows from unquoted commas in text fields."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    pq = path.with_suffix(".parquet")
    if pq.exists():
        print(f"[eval] Loading parquet {pq} (skips fragile CSV).", flush=True)
        return pd.read_parquet(pq)
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[eval] read_csv failed ({e}); retrying with on_bad_lines='skip'...", flush=True)
        try:
            return pd.read_csv(path, low_memory=False, on_bad_lines="skip", engine="python")
        except TypeError:
            return pd.read_csv(path, low_memory=False, error_bad_lines=False, warn_bad_lines=True)


def ensure_graph_columns(df: pd.DataFrame) -> pd.DataFrame:
    """HybridIDSDataset / build_graph_from_flows require Source IP and Destination IP."""
    df = df.copy()
    alias = {
        "Src IP": "Source IP",
        "Dst IP": "Destination IP",
        "Source IP": "Source IP",
        "Destination IP": "Destination IP",
    }
    for old, new in alias.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "Source IP" not in df.columns:
        df["Source IP"] = "0.0.0.0"
    if "Destination IP" not in df.columns:
        df["Destination IP"] = "0.0.0.0"
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.Timestamp.utcnow()
    return df


def run_eval(
    run_dir: Path,
    data_path: Path,
    batch_size: int = 256,
    num_workers: int = 0,
    max_windows: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> dict:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    feat_cols = ckpt["feat_cols"]
    label2idx = ckpt.get("label2idx", {"BENIGN": 0, "ATTACK": 1})
    idx2label = dict(ckpt.get("idx2label", {0: "BENIGN", 1: "ATTACK"}))
    # Normalize idx2label keys to int
    if idx2label:
        k0 = next(iter(idx2label.keys()))
        if isinstance(k0, str):
            idx2label = {int(k): v for k, v in idx2label.items()}

    if cache_dir is not None:
        from src.data_loader import CachedHybridIDSDataset
        from torch.utils.data import Subset

        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

        print(f"[eval] Using CachedHybridIDSDataset from: {cache_dir}", flush=True)
        ds_full = CachedHybridIDSDataset(cache_dir)
        if max_windows is not None and max_windows > 0 and max_windows < len(ds_full):
            ds = Subset(ds_full, list(range(int(max_windows))))
        else:
            ds = ds_full

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_hybrid,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        n_batches = len(dl)
        n_win = len(ds)
        print(
            f"[eval] Evaluating {n_win:,} cached windows in {n_batches:,} batches (batch_size={batch_size})...",
            flush=True,
        )
    else:
        df = read_binary_eval_table(data_path)
        if "Label" not in df.columns:
            raise ValueError("Label column is missing in 2018 binary test data.")

        df = ensure_graph_columns(df)

        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0

        # Ensure binary target labels
        df["Label"] = np.where(
            df["Label"].astype(str).str.upper() == "BENIGN", "BENIGN", "ATTACK"
        )
        print("Building HybridIDSDataset (feature matrix + temporal sort)...", flush=True)
        ds_full = HybridIDSDataset(df, feat_cols, label2idx, seq_len=SEQ_LEN, max_graph_flows=64)
        if max_windows is not None and max_windows > 0 and max_windows < len(ds_full):
            ds = Subset(ds_full, list(range(int(max_windows))))
        else:
            ds = ds_full
        nw = effective_eval_num_workers(num_workers)
        dl_kw: Dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "collate_fn": collate_hybrid,
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
        }
        if nw > 0:
            dl_kw["persistent_workers"] = True
            dl_kw["prefetch_factor"] = 2
        dl = DataLoader(ds, **dl_kw)
        n_batches = len(dl)
        n_win = len(ds)
        print(
            f"Evaluating {n_win:,} windows in {n_batches:,} batches (batch_size={batch_size}, workers={nw})...",
            flush=True,
        )
        if n_win > 2_000_000:
            print(
                "[eval] Very large window count — consider a smaller CSV slice or --max-windows if added.",
                flush=True,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridIDSModel(
        num_features=len(feat_cols),
        num_classes=len(label2idx),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for b in tqdm(
            dl,
            total=n_batches,
            desc="CICIDS 2018 eval (binary)",
            unit="batch",
            mininterval=0.5,
        ):
            seq = b["seq"].to(device)
            ei = b["edge_index"].to(device)
            ea = b["edge_attr"].to(device)
            bv = b["batch"].to(device)
            logits = model(seq, ei, ea, bv, b["batch_size"])
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(b["label"].cpu().numpy().tolist())

    classes = sorted(idx2label.keys()) if idx2label else [0, 1]
    labels_str = [idx2label[i] for i in classes] if idx2label else ["BENIGN", "ATTACK"]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(labels_str):
        per_class[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "labels": labels_str,
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=classes,
            target_names=labels_str,
            zero_division=0,
        ),
        "per_class": per_class,
        "num_windows": int(len(y_true)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--data", type=Path, default=PROCESSED_DIR / "traffic_2018_binary_eval_balanced.parquet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="If provided, evaluate using CachedHybridIDSDataset from this cache directory.",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap number of temporal windows (faster smoke test)",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: EVAL_NUM_WORKERS env or 0)",
    )
    args = ap.parse_args()

    if args.run:
        run_dir = MODELS_DIR / args.run
    else:
        cand = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.endswith("_binary")], reverse=True)
        if not cand:
            print("No binary run found. Run python run_train_binary.py first.")
            return 1
        run_dir = cand[0]

    nw = args.num_workers
    if nw is None:
        nw = int(os.environ.get("EVAL_NUM_WORKERS", "0"))
    metrics = run_eval(
        run_dir,
        args.data,
        batch_size=args.batch_size,
        num_workers=nw,
        max_windows=args.max_windows,
        cache_dir=args.cache_dir,
    )
    out = run_dir / "cicids2018_eval_binary.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved binary 2018 metrics to {out}")
    print(f"accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} windows={metrics['num_windows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

