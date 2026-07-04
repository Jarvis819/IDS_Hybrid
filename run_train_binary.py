"""Train binary IDS model (BENIGN vs ATTACK) with cached dataset path."""
import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from src.model import HybridIDSModel
from src.data_loader import CachedHybridIDSDataset, collate_hybrid, load_cached_training_meta
from src.config import (
    BATCH_SIZE,
    SEQ_LEN,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    HIDDEN_DIM,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    GNN_HIDDEN,
    GNN_LAYERS,
    DROPOUT,
)


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "outputs" / "models"


def main() -> int:
    cache_dir = PROCESSED_DIR / "cache" / "train_cache_binary_v1"
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        print(
            "Missing binary cached data. Run:\n"
            "python precompute_training_cache.py "
            "--cache_dir \"processed/cache/train_cache_binary_v1\" "
            "--data_filename \"traffic_all_binary_processed.parquet\" "
            "--scaler_filename \"scaler_binary.joblib\""
        )
        return 1
    meta = load_cached_training_meta(cache_dir)
    feat_cols = meta["feat_cols"]
    label2idx = meta["label2idx"]
    idx2label = {int(k): v for k, v in meta["idx2label"].items()}
    num_features = int(meta.get("num_features", len(feat_cols)))
    num_classes = int(meta.get("num_classes", len(label2idx)))
    dataset = CachedHybridIDSDataset(cache_dir)
    print(f"[binary] cache_dir={cache_dir}", flush=True)
    print(f"[binary] windows={dataset.num_items:,} features={num_features} classes={num_classes}", flush=True)
    n = dataset.num_items
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[binary] device={device}", flush=True)
    # Windows can become memory-heavy with many worker processes.
    # Keep default modest and configurable.
    num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
    num_workers = max(0, min(num_workers, os.cpu_count() or 1))
    print(f"[binary] dataloader_workers={num_workers}", flush=True)

    dl_kwargs = {
        "batch_size": BATCH_SIZE,
        "collate_fn": collate_hybrid,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dl_kwargs,
    )
    print(f"[binary] train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)

    model = HybridIDSModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        transformer_dim=TRANSFORMER_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS,
        gnn_hidden=GNN_HIDDEN,
        gnn_layers=GNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    run_dir = MODELS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_binary"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(EPOCHS):
        print(f"[binary] epoch {ep+1}/{EPOCHS} starting...", flush=True)
        model.train()
        tr_loss = 0.0
        tr_y, tr_p = [], []
        for b in tqdm(train_loader, desc=f"[binary] epoch {ep+1}/{EPOCHS} train", leave=False):
            seq = b["seq"].to(device)
            ei = b["edge_index"].to(device)
            ea = b["edge_attr"].to(device)
            bv = b["batch"].to(device)
            y = b["label"].to(device)
            logits = model(seq, ei, ea, bv, b["batch_size"])
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_p.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            tr_y.extend(y.detach().cpu().numpy().tolist())

        model.eval()
        va_loss = 0.0
        va_y, va_p, va_probs = [], [], []
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"[binary] epoch {ep+1}/{EPOCHS} val", leave=False):
                seq = b["seq"].to(device)
                ei = b["edge_index"].to(device)
                ea = b["edge_attr"].to(device)
                bv = b["batch"].to(device)
                y = b["label"].to(device)
                logits = model(seq, ei, ea, bv, b["batch_size"])
                loss = crit(logits, y)
                va_loss += loss.item()
                va_p.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                va_y.extend(y.cpu().numpy().tolist())
                va_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        tr_acc = float(accuracy_score(tr_y, tr_p)) if tr_y else 0.0
        va_acc = float(accuracy_score(va_y, va_p)) if va_y else 0.0
        history["train_loss"].append(tr_loss / max(1, len(train_loader)))
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss / max(1, len(val_loader)))
        history["val_acc"].append(va_acc)
        print(f"[binary] epoch {ep+1}/{EPOCHS} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}", flush=True)

        if va_acc > best:
            best = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "feat_cols": feat_cols,
                    "epoch": ep,
                },
                run_dir / "best_model.pt",
            )

    metrics = {}
    classes = sorted(idx2label.keys())
    labels_str = [idx2label[i] for i in classes]
    if va_y:
        y_true_val = np.array(va_y, dtype=int)
        y_pred_val = np.array(va_p, dtype=int)
        probs_val = np.concatenate(va_probs, axis=0) if va_probs else np.zeros((len(y_true_val), 2), dtype=np.float32)
        p, r, f, s = precision_recall_fscore_support(
            y_true_val, y_pred_val, labels=classes, average=None, zero_division=0
        )
        try:
            y_true_val_bin = label_binarize(y_true_val, classes=classes)
            roc_auc_macro = float(
                roc_auc_score(y_true_val_bin, probs_val, average="macro", multi_class="ovr")
            )
        except Exception:
            roc_auc_macro = float("nan")

        true_counts_val = {lbl: int((y_true_val == cls_idx).sum()) for cls_idx, lbl in zip(classes, labels_str)}
        pred_counts_val = {lbl: int((y_pred_val == cls_idx).sum()) for cls_idx, lbl in zip(classes, labels_str)}
        mis_counter_val = Counter()
        for t_idx, p_idx in zip(y_true_val, y_pred_val):
            if t_idx != p_idx:
                mis_counter_val[(labels_str[t_idx], labels_str[p_idx])] += 1
        top_mis_val = [{"true": t, "pred": p_, "count": int(c)} for (t, p_), c in mis_counter_val.most_common(5)]

        metrics = {
            "accuracy": float(accuracy_score(y_true_val, y_pred_val)),
            "precision_macro": float(precision_recall_fscore_support(y_true_val, y_pred_val, average="macro", zero_division=0)[0]),
            "recall_macro": float(precision_recall_fscore_support(y_true_val, y_pred_val, average="macro", zero_division=0)[1]),
            "f1_macro": float(precision_recall_fscore_support(y_true_val, y_pred_val, average="macro", zero_division=0)[2]),
            "roc_auc_macro": roc_auc_macro,
            "labels": labels_str,
            "confusion": confusion_matrix(y_true_val, y_pred_val, labels=classes).tolist(),
            "classification_report": classification_report(y_true_val, y_pred_val, labels=classes, target_names=labels_str, zero_division=0),
            "per_class": {
                lbl: {
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "f1": float(f[i]),
                    "support": int(s[i]),
                }
                for i, lbl in enumerate(labels_str)
            },
            "class_dist_true": true_counts_val,
            "class_dist_pred": pred_counts_val,
            "top_misclass_pairs": top_mis_val,
            "roc_curves": {
                lbl: {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": roc_auc_macro}
                for lbl in labels_str
            },
        }

    # Training split detailed metrics (same structure used by multi model)
    if tr_y:
        y_true_tr = np.array(tr_y, dtype=int)
        y_pred_tr = np.array(tr_p, dtype=int)
        p_tr, r_tr, f_tr, s_tr = precision_recall_fscore_support(
            y_true_tr, y_pred_tr, labels=classes, average=None, zero_division=0
        )
        true_counts_tr = {lbl: int((y_true_tr == cls_idx).sum()) for cls_idx, lbl in zip(classes, labels_str)}
        pred_counts_tr = {lbl: int((y_pred_tr == cls_idx).sum()) for cls_idx, lbl in zip(classes, labels_str)}
        mis_counter_tr = Counter()
        for t_idx, p_idx in zip(y_true_tr, y_pred_tr):
            if t_idx != p_idx:
                mis_counter_tr[(labels_str[t_idx], labels_str[p_idx])] += 1
        top_mis_tr = [{"true": t, "pred": p_, "count": int(c)} for (t, p_), c in mis_counter_tr.most_common(5)]

        metrics["train_split"] = {
            "accuracy": float(accuracy_score(y_true_tr, y_pred_tr)),
            "precision_macro": float(precision_recall_fscore_support(y_true_tr, y_pred_tr, labels=classes, average="macro", zero_division=0)[0]),
            "recall_macro": float(precision_recall_fscore_support(y_true_tr, y_pred_tr, labels=classes, average="macro", zero_division=0)[1]),
            "f1_macro": float(precision_recall_fscore_support(y_true_tr, y_pred_tr, labels=classes, average="macro", zero_division=0)[2]),
            "confusion": confusion_matrix(y_true_tr, y_pred_tr, labels=classes).tolist(),
            "per_class": {
                lbl: {
                    "precision": float(p_tr[i]),
                    "recall": float(r_tr[i]),
                    "f1": float(f_tr[i]),
                    "support": int(s_tr[i]),
                }
                for i, lbl in enumerate(labels_str)
            },
            "class_dist_true": true_counts_tr,
            "class_dist_pred": pred_counts_tr,
            "top_misclass_pairs": top_mis_tr,
        }

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "num_classes": num_classes,
                "num_features": num_features,
                "pipeline_type": "binary_all_attacks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "train_eval_metrics_binary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[binary] saved run to {run_dir}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

