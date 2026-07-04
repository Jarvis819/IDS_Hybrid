"""
Offline cache generator for faster training.

Goal:
- Precompute temporal sequences (X_seq, y_seq)
- Precompute graph structure (edge_index, edge_attr) per cached window

After this script completes, training can use `CachedHybridIDSDataset` where
`__getitem__` is just indexing into NumPy memmaps (no pandas, no iterrows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm

import torch

from src.config import PROCESSED_DIR, SEQ_LEN, GRAPH_SIZE, WINDOW_STRIDE
from src.data_loader import build_label_map, load_processed_data


def _open_memmap(path: Path, mode: str, shape: tuple[int, ...], dtype: np.dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode=mode, dtype=dtype, shape=shape)


def _unique_first_occurrence_order(arr: np.ndarray) -> np.ndarray:
    """
    Unique values in first-occurrence order.

    numpy's `np.unique` sorts; we recover first-occurrence order using first indices.
    """
    if arr.size == 0:
        return arr
    uniq, first_idx = np.unique(arr, return_index=True)
    order = np.argsort(first_idx)
    return uniq[order]


def precompute_hybrid_window_cache(
    df: pd.DataFrame,
    feat_cols: List[str],
    label2idx: Dict[str, Any],
    cache_dir: Path,
    seq_len: int,
    max_graph_flows: int,
    window_stride: int,
    max_nodes: int = 64,
    *,
    idx2label: Optional[Dict[Union[int, str], str]] = None,
    max_windows_cap: Optional[int] = None,
    quiet: bool = False,
) -> int:
    """
    Write CachedHybridIDSDataset-compatible memmaps (X_seq, y_seq, edges) to cache_dir.

    Same graph/window logic as offline training cache generation. Optionally cap the number
    of windows (first N along the timeline) for faster inference on large uploads.

    Returns:
        num_items written
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Match HybridIDSDataset temporal ordering.
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

    num_features = len(feat_cols)
    n = len(df)

    feat_matrix = df[feat_cols].astype(np.float32).to_numpy(copy=True)
    labels_str = df["Label"].astype(str).fillna("BENIGN").to_numpy()
    y_flow = np.array(
        [label2idx.get(lbl, label2idx.get("BENIGN", 0)) for lbl in labels_str],
        dtype=np.int64,
    )

    if "Source IP" not in df.columns or "Destination IP" not in df.columns:
        raise KeyError(
            "DataFrame must contain 'Source IP' and 'Destination IP' for graph construction."
        )
    src_ip = df["Source IP"]
    dst_ip = df["Destination IP"]
    combined = pd.concat([src_ip, dst_ip], ignore_index=True)
    ip_codes_combined, _uniques = pd.factorize(combined, sort=False, use_na_sentinel=True)
    src_codes = ip_codes_combined[: len(df)]
    dst_codes = ip_codes_combined[len(df) :]

    base = max(1, n - seq_len - max_graph_flows)
    full_num_items = (base + window_stride - 1) // window_stride
    if max_windows_cap is not None:
        num_items = min(full_num_items, int(max_windows_cap))
    else:
        num_items = full_num_items

    edge_offsets = np.zeros(num_items + 1, dtype=np.int64)

    rng = range(num_items)
    if not quiet:
        print("Pass 1/2: computing edge counts...", flush=True)
        rng = tqdm(rng, desc="edge_count", mininterval=1.0)  # type: ignore[assignment]

    for i in rng:
        start_idx = i * window_stride
        g_start = max(0, start_idx)
        g_end = min(start_idx + max_graph_flows, n)

        src_w = src_codes[g_start:g_end]
        dst_w = dst_codes[g_start:g_end]

        valid_rows = (src_w != -1) & (dst_w != -1)
        if not np.any(valid_rows):
            edge_offsets[i + 1] = edge_offsets[i]
            continue

        src_non_missing = src_w[src_w != -1]
        dst_non_missing = dst_w[dst_w != -1]
        ips_concat = np.concatenate([src_non_missing, dst_non_missing], axis=0)
        ips_unique_ordered = _unique_first_occurrence_order(ips_concat)
        ips_selected = ips_unique_ordered[:max_nodes]

        if ips_selected.size == 0:
            edge_offsets[i + 1] = edge_offsets[i]
            continue

        keep = valid_rows & np.isin(src_w, ips_selected) & np.isin(dst_w, ips_selected)
        e_i = int(np.sum(keep))
        edge_offsets[i + 1] = edge_offsets[i] + e_i

    total_edges = int(edge_offsets[-1])
    if not quiet:
        print(f"Total cached windows: {num_items:,}", flush=True)
        print(f"Total edges across all cached windows: {total_edges:,}", flush=True)
        print("Pass 2/2: writing cached arrays...", flush=True)

    X_seq_path = cache_dir / "X_seq.npy"
    y_seq_path = cache_dir / "y_seq.npy"
    edge_offsets_path = cache_dir / "edge_offsets.npy"
    edge_index_path = cache_dir / "edge_index.npy"
    edge_attr_path = cache_dir / "edge_attr.npy"

    X_seq = _open_memmap(X_seq_path, "w+", (num_items, seq_len, num_features), np.float32)
    y_seq = _open_memmap(y_seq_path, "w+", (num_items,), np.int64)
    edge_index = _open_memmap(edge_index_path, "w+", (2, total_edges), np.int64)
    edge_attr = _open_memmap(edge_attr_path, "w+", (total_edges, num_features), np.float32)

    np.save(edge_offsets_path, edge_offsets)

    rng2 = range(num_items)
    if not quiet:
        rng2 = tqdm(rng2, desc="fill", mininterval=1.0)  # type: ignore[assignment]

    for i in rng2:
        start_idx = i * window_stride
        g_start = max(0, start_idx)
        g_end = min(start_idx + max_graph_flows, n)

        end = min(start_idx + seq_len, n)
        start = max(0, end - seq_len)
        length = end - start

        X_seq[i, :, :] = 0.0
        if length > 0:
            X_seq[i, seq_len - length :, :] = feat_matrix[start:end]

        center_idx = (start + end) // 2
        y_seq[i] = y_flow[center_idx]

        src_w = src_codes[g_start:g_end]
        dst_w = dst_codes[g_start:g_end]
        valid_rows = (src_w != -1) & (dst_w != -1)

        src_non_missing = src_w[src_w != -1]
        dst_non_missing = dst_w[dst_w != -1]
        ips_concat = np.concatenate([src_non_missing, dst_non_missing], axis=0)
        ips_unique_ordered = _unique_first_occurrence_order(ips_concat)
        ips_selected = ips_unique_ordered[:max_nodes]

        start_e = int(edge_offsets[i])
        end_e = int(edge_offsets[i + 1])
        e_i = end_e - start_e

        if e_i == 0 or ips_selected.size == 0:
            continue

        keep = valid_rows & np.isin(src_w, ips_selected) & np.isin(dst_w, ips_selected)
        e_sources = src_w[keep]
        e_dests = dst_w[keep]

        sort_idx = np.argsort(ips_selected)
        ips_sorted = ips_selected[sort_idx]
        inv = np.empty_like(sort_idx, dtype=np.int64)
        inv[sort_idx] = np.arange(ips_selected.shape[0], dtype=np.int64)

        src_sorted_pos = np.searchsorted(ips_sorted, e_sources)
        dst_sorted_pos = np.searchsorted(ips_sorted, e_dests)
        local_src = inv[src_sorted_pos]
        local_dst = inv[dst_sorted_pos]

        edge_index[:, start_e:end_e] = np.stack([local_src, local_dst], axis=0)

        flow_idx_local = np.arange(g_start, g_end, dtype=np.int64)[keep]
        edge_attr[start_e:end_e, :] = feat_matrix[flow_idx_local]

    idx_meta: Dict[str, str] = {}
    if idx2label:
        idx_meta = {str(k): v for k, v in idx2label.items()}

    meta: Dict[str, Any] = {
        "seq_len": seq_len,
        "max_graph_flows": max_graph_flows,
        "window_stride": window_stride,
        "max_nodes": max_nodes,
        "n_flows": n,
        "num_items": num_items,
        "num_features": num_features,
        "feat_cols": feat_cols,
        "label2idx": label2idx,
        "idx2label": idx_meta,
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not quiet:
        print(f"Cached hybrid window data written to: {cache_dir}", flush=True)
    return int(num_items)


def precompute_cache(
    processed_dir: Path,
    cache_dir: Path,
    seq_len: int,
    max_graph_flows: int,
    window_stride: int,
    data_filename: str = "traffic_all_processed.parquet",
    scaler_filename: str = "scaler.joblib",
    max_nodes: int = 64,
    force: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"
    if meta_path.exists() and not force:
        print(f"Cached training data already exists: {cache_dir}")
        return

    data_path = processed_dir / data_filename
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        # Some parquet files can be unreadable in this environment; fall back to CSV.
        csv_path = data_path.with_suffix(".csv")
        if not csv_path.exists():
            raise
        print(f"Parquet read failed ({e}); falling back to CSV: {csv_path}", flush=True)
        df = pd.read_csv(csv_path, low_memory=False)

    artifacts = joblib.load(processed_dir / scaler_filename)
    feat_cols = artifacts["numeric_feature_names"]

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        label2idx = ckpt["label2idx"]
        idx2label_ckpt = ckpt.get("idx2label", {})
        idx2label = {int(k): v for k, v in idx2label_ckpt.items()} if idx2label_ckpt else {}
    else:
        label2idx, idx2label = build_label_map(df["Label"])

    precompute_hybrid_window_cache(
        df,
        feat_cols,
        label2idx,
        cache_dir,
        seq_len,
        max_graph_flows,
        window_stride,
        max_nodes,
        idx2label=idx2label,
        max_windows_cap=None,
        quiet=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(PROCESSED_DIR / "cache" / "train_cache_v1"))
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--max_graph_flows", type=int, default=GRAPH_SIZE)
    parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--data_filename", type=str, default="traffic_all_processed.parquet")
    parser.add_argument("--scaler_filename", type=str, default="scaler.joblib")
    parser.add_argument("--checkpoint_path", type=Path, default=None, help="Optional checkpoint to force label2idx mapping.")
    parser.add_argument("--max_nodes", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    precompute_cache(
        processed_dir=Path(args.processed_dir),
        cache_dir=Path(args.cache_dir),
        seq_len=int(args.seq_len),
        max_graph_flows=int(args.max_graph_flows),
        window_stride=int(args.window_stride),
        data_filename=str(args.data_filename),
        scaler_filename=str(args.scaler_filename),
        checkpoint_path=args.checkpoint_path,
        max_nodes=int(args.max_nodes),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()

