"""
Evaluate trained model on CICIDS 2018 (version 2) test data.

This mirrors the detailed validation metrics used during training:
- Accuracy, macro precision/recall/F1
- Confusion matrix
- Per-class metrics

Metrics are saved per run as `cicids2018_eval.json` for use in the dashboard.

Performance notes:
- Forward pass is batched on GPU when available (efficient).
- Each window still builds a small graph inside HybridIDSDataset (same dynamic path as
  uncached training). For maximum speed you would precompute a 2018 cache like
  precompute_training_cache.py; we do not do that by default due to disk/time cost.
- On Linux/Mac, EVAL_NUM_WORKERS / --num-workers can parallelize __getitem__.
  On Windows, workers are forced to 0 (multiprocessing cannot pickle this Dataset).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "outputs" / "models"


def effective_eval_num_workers(requested: int) -> int:
    """
    Windows + multiprocessing DataLoader tries to pickle the whole Dataset (including a huge
    pandas DataFrame) into worker processes, which fails with OSError / truncated pickle.
    Force single-threaded loading on Windows for this eval path.
    """
    req = max(0, int(requested))
    if sys.platform == "win32" and req > 0:
        print(
            "[eval] num_workers>0 is not supported on Windows for this eval (dataset is not picklable). "
            "Using num_workers=0.",
            flush=True,
        )
        return 0
    return req


def read_eval_table(path: Path) -> pd.DataFrame:
    """Load CSV/parquet; CSV may contain commas inside unquoted text fields — skip bad lines if needed."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[eval] read_csv failed ({e}); retrying with on_bad_lines='skip'...", flush=True)
        try:
            return pd.read_csv(path, low_memory=False, on_bad_lines="skip", engine="python")
        except TypeError:
            return pd.read_csv(path, low_memory=False, error_bad_lines=False, warn_bad_lines=True)


def find_latest_multi_class_run() -> Optional[Path]:
    """Latest run folder with best_model.pt that is not the binary (2-class) model."""
    if not MODELS_DIR.exists():
        return None
    runs = sorted(MODELS_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    for r in runs:
        if not r.is_dir():
            continue
        ckpt_path = r / "best_model.pt"
        if not ckpt_path.exists():
            continue
        if r.name.endswith("_binary"):
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            label2idx = ckpt.get("label2idx") or {}
            # Binary pipeline uses exactly BENIGN + ATTACK (2 classes).
            if len(label2idx) == 2:
                continue
        except Exception:
            continue
        return r
    return None


def ensure_columns_for_hybrid_dataset(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Add missing feature / graph columns so HybridIDSDataset can build windows and graphs."""
    df = df.copy()
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    if "Source IP" not in df.columns:
        df["Source IP"] = "0.0.0.0"
    if "Destination IP" not in df.columns:
        df["Destination IP"] = "0.0.0.0"
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.Timestamp.utcnow()
    return df


def load_model_and_artifacts(run_dir: Path):
    """Load model checkpoint and metadata for evaluation."""
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt


def run_eval_2018(
    run_dir: Path,
    data_path: Path = None,
    batch_size: int = 256,
    max_windows: int = None,
    device: Optional[str] = None,
    num_workers: int = 0,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Evaluate model on CICIDS 2018 version-2 (train-classes-only) data."""
    from src.data_loader import HybridIDSDataset, collate_hybrid
    from src.model import HybridIDSModel
    from src.config import SEQ_LEN

    ckpt = load_model_and_artifacts(run_dir)
    label2idx = ckpt["label2idx"]
    idx2label = dict(ckpt["idx2label"])
    if idx2label:
        k0 = next(iter(idx2label.keys()))
        if isinstance(k0, str):
            idx2label = {int(k): v for k, v in idx2label.items()}
    feat_cols = ckpt["feat_cols"]
    num_classes = len(label2idx)

    # Build dataset and (optionally truncated) loader
    if cache_dir is not None:
        from src.data_loader import CachedHybridIDSDataset
        from torch.utils.data import Subset

        print(f"[eval] Using CachedHybridIDSDataset from: {cache_dir}", flush=True)
        dataset_full = CachedHybridIDSDataset(cache_dir)
        if max_windows is not None and max_windows > 0 and max_windows < len(dataset_full):
            max_windows = int(max_windows)
            dataset = Subset(dataset_full, list(range(max_windows)))
        else:
            dataset = dataset_full

        nw = 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_hybrid,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        n_batches = len(loader)
        n_win = len(dataset)
        print(
            f"[eval] Evaluating {n_win:,} cached windows in {n_batches:,} batches (batch_size={batch_size})...",
            flush=True,
        )
    else:
        # Prefer CSV (robust across environments); fall back to parquet if needed.
        if data_path is None:
            # Default to balanced CICIDS 2018 eval file for Model A
            data_path = PROCESSED_DIR / "traffic_2018_multiclass_eval_balanced.parquet"

        if not data_path.exists():
            # Try parquet as a secondary option
            parquet_path = PROCESSED_DIR / "traffic_2018_multiclass_eval_balanced.parquet"
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to read CICIDS 2018 data from {parquet_path}: {e}. "
                        "Re-run `python preprocess_cicids_2018.py` to regenerate CSV/Parquet."
                    )
            else:
                raise FileNotFoundError(
                    f"CICIDS 2018 processed file not found at {data_path} and parquet not found at {parquet_path}. "
                    "Run `python preprocess_cicids_2018.py` first."
                )
        else:
            df = read_eval_table(data_path)

        df = ensure_columns_for_hybrid_dataset(df, list(feat_cols))

        if max_windows is not None and max_windows > 0:
            max_windows = int(max_windows)
        else:
            max_windows = None

        print("Building HybridIDSDataset (feature matrix + temporal sort)...", flush=True)
        dataset_full = HybridIDSDataset(
            df, feat_cols, label2idx, seq_len=SEQ_LEN, max_graph_flows=64
        )
        if max_windows is not None and max_windows < len(dataset_full):
            from torch.utils.data import Subset

            indices = list(range(max_windows))
            dataset = Subset(dataset_full, indices)
        else:
            dataset = dataset_full

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

        loader = DataLoader(dataset, **dl_kw)
        n_batches = len(loader)
        n_win = len(dataset)
        print(
            f"Evaluating {n_win:,} windows in {n_batches:,} batches (batch_size={batch_size}, workers={nw})...",
            flush=True,
        )
        if n_win > 2_000_000:
            print(
                "[eval] Very large window count — full pass can take a long time. "
                "Use --max-windows 50000 for a faster smoke test.",
                flush=True,
            )

    model = HybridIDSModel(
        num_features=len(feat_cols),
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_preds = []
    all_true_idx = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=n_batches,
            desc="CICIDS 2018 eval (multi-class)",
            unit="batch",
            mininterval=0.5,
        ):
            seq = batch["seq"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_vec = batch["batch"].to(device)
            bs = batch["batch_size"]

            logits = model(seq, edge_index, edge_attr, batch_vec, bs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

            labels = batch["label"].cpu().numpy()
            all_true_idx.extend(labels.tolist())

    y_true = np.array(all_true_idx, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    classes = sorted(idx2label.keys())
    labels_str = [idx2label[i] for i in classes]

    metrics: dict = {}
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )
        prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(
            y_true,
            y_pred,
            labels=classes,
            target_names=labels_str,
            zero_division=0,
        )

        per_class = {}
        for i, lbl in enumerate(labels_str):
            per_class[lbl] = {
                "precision": float(prec_cls[i]),
                "recall": float(rec_cls[i]),
                "f1": float(f1_cls[i]),
                "support": int(support_cls[i]),
            }

        metrics.update(
            {
                "accuracy": float(acc),
                "precision_macro": float(prec_macro),
                "recall_macro": float(rec_macro),
                "f1_macro": float(f1_macro),
                "labels": labels_str,
                "confusion": cm.tolist(),
                "classification_report": report,
                "per_class": per_class,
                "num_windows": int(len(y_true)),
            }
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on CICIDS 2018 (version 2) data")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name (e.g. 20260301_170905). Default: latest.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DIR / "traffic_2018_multiclass_eval_balanced.parquet",
        help="Path to balanced CICIDS 2018 eval file for Model A (recommended)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="If provided, evaluate using CachedHybridIDSDataset from this directory (fast; no graph building).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for parallel __getitem__ (default: EVAL_NUM_WORKERS env or 0)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap number of temporal windows for quick evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save metrics JSON here (default: run_dir/cicids2018_eval.json)",
    )
    args = parser.parse_args()

    if args.run:
        run_dir = MODELS_DIR / args.run
    else:
        run_dir = find_latest_multi_class_run()
        if run_dir is None:
            print(
                "ERROR: No multi-class (Model A) run found. "
                "Train with `python run_train.py` or pass `--run <timestamp>` explicitly."
            )
            return 1

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    nw = args.num_workers
    if nw is None:
        nw = int(os.environ.get("EVAL_NUM_WORKERS", "0"))
    nw = max(0, nw)

    print(f"Evaluating model from {run_dir.name} on CICIDS 2018 (version 2)...")
    metrics = run_eval_2018(
        run_dir,
        data_path=args.data,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        num_workers=nw,
        cache_dir=args.cache_dir,
    )

    if not metrics:
        print("No metrics computed (empty dataset?).")
        return 1

    print("\n" + "=" * 50)
    print("CICIDS 2018 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Num temporal windows: {metrics['num_windows']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"].encode("ascii", errors="replace").decode("ascii"))
    print("=" * 50)

    out_path = args.output or run_dir / "cicids2018_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

