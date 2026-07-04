"""Data loading, graph construction, and dataset for hybrid IDS."""
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_processed_data(processed_dir: Path) -> Tuple[pd.DataFrame, List[str], Any]:
    """Load processed parquet and preprocessing artifacts."""
    df = pd.read_parquet(processed_dir / "traffic_all_processed.parquet")
    artifacts = joblib.load(processed_dir / "scaler.joblib")
    feat_names = artifacts["numeric_feature_names"]
    return df, feat_names, artifacts


def build_label_map(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label to index and index to label mappings."""
    unique = labels.dropna().unique().tolist()
    label2idx = {str(l): i for i, l in enumerate(sorted(unique))}
    idx2label = {i: l for l, i in label2idx.items()}
    return label2idx, idx2label


def build_graph_from_flows(
    flow_indices: np.ndarray,
    df: pd.DataFrame,
    feat_cols: List[str],
    ip_cols: Tuple[str, str] = ("Source IP", "Destination IP"),
    max_nodes: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a graph from a subset of flows.
    Nodes = unique IPs, Edges = flows. Edge features = flow numeric features.

    Returns:
        edge_index: [2, num_edges]
        edge_attr: [num_edges, num_features]
        node_ids: list of IP strings (for debugging)
    """
    sub = df.iloc[flow_indices].copy()
    ips = pd.concat([sub[ip_cols[0]], sub[ip_cols[1]]]).dropna().astype(str).unique()
    ip2idx = {ip: i for i, ip in enumerate(ips[:max_nodes])}

    edges = []
    edge_feats = []
    for _, row in sub.iterrows():
        src = str(row[ip_cols[0]])
        dst = str(row[ip_cols[1]])
        if pd.isna(row[ip_cols[0]]) or pd.isna(row[ip_cols[1]]):
            continue
        if src not in ip2idx or dst not in ip2idx:
            continue
        edges.append((ip2idx[src], ip2idx[dst]))
        edge_feats.append(row[feat_cols].astype(np.float32).values)

    if len(edges) == 0:
        # Return minimal graph
        n = min(1, len(ip2idx))
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, len(feat_cols), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.tensor(np.stack(edge_feats), dtype=torch.float32)

    return edge_index, edge_attr, list(ip2idx.keys())


class HybridIDSDataset(Dataset):
    """Dataset yielding (temporal sequence, graph, label) for hybrid model."""

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        label2idx: Dict[str, int],
        seq_len: int = 32,
        max_graph_flows: int = 64,
        timestamp_col: str = "Timestamp",
        label_col: str = "Label",
        window_stride: int = 1,
        precomputed_feat_matrix: Optional[np.ndarray] = None,
        precomputed_labels: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.feat_cols = feat_cols
        self.label2idx = label2idx
        self.seq_len = seq_len
        self.max_graph_flows = max_graph_flows
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.window_stride = max(1, int(window_stride))

        # Sort by timestamp for temporal coherence
        if timestamp_col in self.df.columns:
            try:
                self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col], errors="coerce")
                self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
            except Exception:
                pass
        self.n = len(self.df)

        # Precompute numeric feature matrix once so that __getitem__
        # can slice windows via NumPy instead of going through pandas
        # and reallocating arrays every time.
        if precomputed_feat_matrix is not None:
            self.feat_matrix = precomputed_feat_matrix
        else:
            self.feat_matrix = (
                self.df[self.feat_cols].astype(np.float32).to_numpy(copy=True)
            )

        if precomputed_labels is not None:
            self.labels_arr = precomputed_labels
        else:
            self.labels_arr = self.df[self.label_col].astype(str).fillna("BENIGN").to_numpy()

    def __len__(self) -> int:
        base = max(1, self.n - self.seq_len - self.max_graph_flows)
        # Downsample windows by stride while still covering the capture.
        return (base + self.window_stride - 1) // self.window_stride

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map dataset index to starting flow index using stride
        start_idx = idx * self.window_stride

        # Temporal window: consecutive flows
        end = min(start_idx + self.seq_len, self.n)
        start = max(0, end - self.seq_len)
        length = end - start

        # Build fixed-length window backed by precomputed NumPy matrix.
        # We right-align the real flows and left-pad with zeros when near
        # the start of the capture.
        seq_np = np.zeros((self.seq_len, len(self.feat_cols)), dtype=np.float32)
        if length > 0:
            seq_np[self.seq_len - length :] = self.feat_matrix[start:end]
        seq_feats = torch.from_numpy(seq_np)

        # Graph: overlapping window for structure (aligned with start_idx)
        g_start = max(0, start_idx)
        g_end = min(start_idx + self.max_graph_flows, self.n)
        graph_indices = np.arange(g_start, g_end)

        edge_index, edge_attr, _ = build_graph_from_flows(
            graph_indices, self.df, self.feat_cols, max_nodes=64
        )

        # Label from center of window
        center_idx = (start + end) // 2
        label = self.labels_arr[center_idx]
        if pd.isna(label):
            label = "BENIGN"
        y = self.label2idx.get(str(label), self.label2idx.get("BENIGN", 0))
        y = torch.tensor(y, dtype=torch.long)

        return {
            "seq": seq_feats,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "label": y,
        }


def collate_hybrid(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch; pad sequences and batch graphs."""
    seqs = torch.stack([b["seq"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    # Batch graphs: disjoint union with offset node indices + batch vector
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    offset = 0
    for i, b in enumerate(batch):
        ei, ea = b["edge_index"], b["edge_attr"]
        if ei.numel() > 0:
            num_nodes = ei.max().item() + 1
            edge_indices.append(ei + offset)
            edge_attrs.append(ea)
            batch_vec.append(torch.full((num_nodes,), i, dtype=torch.long))
            offset += num_nodes
        else:
            batch_vec.append(torch.tensor([i], dtype=torch.long))
            offset += 1

    if any(e.numel() > 0 for e in edge_indices):
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        batch_vec = torch.cat(batch_vec, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, batch[0]["edge_attr"].shape[1], dtype=torch.float32)
        batch_vec = torch.cat(batch_vec, dim=0)

    return {
        "seq": seqs,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_vec,
        "label": labels,
        "batch_size": len(batch),
    }


def load_cached_training_meta(cache_dir: Path) -> Dict[str, Any]:
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Cached training meta not found at: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


class CachedHybridIDSDataset(Dataset):
    """
    Training dataset backed by precomputed NumPy arrays.

    This eliminates per-sample pandas slicing and edge construction during training.
    """

    def __init__(self, cache_dir: Path):
        meta = load_cached_training_meta(cache_dir)
        self.cache_dir = cache_dir
        self.meta = meta

        # Important for Windows: keep the Dataset object picklable.
        # We lazily memory-map arrays in each worker process when first needed.
        self.X_seq = None
        self.y_seq = None
        self.edge_offsets = None
        self.edge_index = None
        self.edge_attr = None

        self.num_items = int(self.meta["num_items"])

    def _ensure_loaded(self) -> None:
        if self.X_seq is not None:
            return

        # Memory-map numeric arrays for fast random access.
        self.X_seq = np.load(self.cache_dir / "X_seq.npy", mmap_mode="r")  # [N, S, F]
        self.y_seq = np.load(self.cache_dir / "y_seq.npy", mmap_mode="r")  # [N]
        self.edge_offsets = np.load(self.cache_dir / "edge_offsets.npy", mmap_mode="r")  # [N+1]
        self.edge_index = np.load(self.cache_dir / "edge_index.npy", mmap_mode="r")  # [2, E_total]
        self.edge_attr = np.load(self.cache_dir / "edge_attr.npy", mmap_mode="r")  # [E_total, F]

        # Sanity checks (cheap).
        n = int(self.X_seq.shape[0])
        if n != int(self.y_seq.shape[0]) or n != int(self.edge_offsets.shape[0] - 1):
            raise ValueError(
                "Cached dataset shape mismatch: "
                f"X_seq={self.X_seq.shape}, y_seq={self.y_seq.shape}, edge_offsets={self.edge_offsets.shape}"
            )

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()

        seq = self.X_seq[idx]  # [S, F]
        y = self.y_seq[idx]

        start = int(self.edge_offsets[idx])
        end = int(self.edge_offsets[idx + 1])

        # edge_index: [2, E_i], edge_attr: [E_i, F]
        if start == end:
            # Minimal graph with 0 edges; collate_hybrid will handle empty edge_index.
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros((0, seq.shape[1]), dtype=torch.float32)
        else:
            edge_index_np = self.edge_index[:, start:end]
            edge_attr_np = self.edge_attr[start:end]
            # Ensure the backing array is writable to avoid PyTorch warnings.
            edge_index = torch.from_numpy(edge_index_np.copy()).to(torch.long)
            edge_attr = torch.from_numpy(edge_attr_np.copy()).to(torch.float32)

        return {
            # Copy to ensure writable backing for PyTorch.
            "seq": torch.from_numpy(seq.copy()).to(torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "label": torch.tensor(y, dtype=torch.long),
        }
