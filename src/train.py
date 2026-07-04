"""Training script for hybrid IDS model."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    HIDDEN_DIM,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    GNN_HIDDEN,
    GNN_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    SEQ_LEN,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    WINDOW_STRIDE,
)
from .data_loader import (
    load_processed_data,
    build_label_map,
    HybridIDSDataset,
    collate_hybrid,
    CachedHybridIDSDataset,
    load_cached_training_meta,
)
from .model import HybridIDSModel


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    correct = 0
    total = 0
    for batch in loader:
        seq = batch["seq"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_vec = batch["batch"].to(device)
        labels = batch["label"].to(device)
        batch_size = batch["batch_size"]

        optimizer.zero_grad()
        logits = model(seq, edge_index, edge_attr, batch_vec, batch_size)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        batch_count += 1
        if batch_count % 25 == 0:
            print(f"  Batch {batch_count}/{len(loader)}", flush=True)

    return total_loss / len(loader), correct / total if total else 0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for batch in loader:
        seq = batch["seq"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_vec = batch["batch"].to(device)
        labels = batch["label"].to(device)
        batch_size = batch["batch_size"]

        logits = model(seq, edge_index, edge_attr, batch_vec, batch_size)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader) if len(loader) else 0, correct / total if total else 0, all_preds, all_labels


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Full training is configured to run on GPU only.", file=sys.stderr)
        print("Activate your GPU environment and run:", file=sys.stderr)
        print("  conda activate ids-gpu", file=sys.stderr)
        print("  python run_train.py", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = MODELS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training-time bottleneck:
    # - per-window graph construction via pandas slicing/iterrows is expensive.
    # - we replace it with an offline precompute cache that contains X_seq + graphs.
    cache_dir = PROCESSED_DIR / "cache" / "train_cache_v1"
    meta_path = cache_dir / "cache_meta.json"

    if not meta_path.exists():
        allow_slow = os.environ.get("ALLOW_SLOW_TRAINING", "").lower() in {"1", "true", "yes"}
        if not allow_slow:
            raise FileNotFoundError(
                f"Cached training data not found at {cache_dir}.\n"
                f"Run: python precompute_training_cache.py --cache_dir \"{cache_dir}\""
            )

    if meta_path.exists():
        meta = load_cached_training_meta(cache_dir)
        dataset = CachedHybridIDSDataset(cache_dir)

        label2idx = meta["label2idx"]
        idx2label = {int(k): v for k, v in meta["idx2label"].items()}
        feat_cols = meta["feat_cols"]
        num_classes = int(meta["num_classes"]) if "num_classes" in meta else len(label2idx)
        num_features = int(meta["num_features"])
        # The offline metrics stage needs `df` for correlation + slicing.
        df, _, _ = load_processed_data(PROCESSED_DIR)
        print(f"Using cached training dataset: {dataset.num_items:,} windows")
        print(f"Classes: {num_classes} -> {sorted(idx2label.values())}")
    else:
        # Fallback (slow): dynamic graph construction in HybridIDSDataset.
        df, feat_cols, _ = load_processed_data(PROCESSED_DIR)
        print(f"Using full processed dataset with {len(df):,} rows for training/validation")
        label2idx, idx2label = build_label_map(df["Label"])
        num_classes = len(label2idx)
        print(f"Classes: {num_classes} -> {list(label2idx.keys())}")
        num_features = len(feat_cols)

        dataset = HybridIDSDataset(
            df,
            feat_cols,
            label2idx,
            seq_len=SEQ_LEN,
            max_graph_flows=64,
            window_stride=WINDOW_STRIDE,
        )

    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # DataLoader optimization:
    # - with cached dataset, workers are safe because __getitem__ only slices NumPy memmaps.
    # - enable pinned memory and persistent workers for faster host->GPU transfer.
    num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
    num_workers = max(1, min(num_workers, os.cpu_count() or 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_hybrid,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_hybrid,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Starting training...", flush=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} starting...", flush=True)
        train_loss, train_acc = train_epoch(
            model,
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False),
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False),
            criterion,
            device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "feat_cols": feat_cols,
                "epoch": epoch,
            }, run_dir / "best_model.pt")

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Final detailed metrics for validation (and basic metrics for training) saved for dashboard
    print("Computing detailed validation and training metrics...", flush=True)
    model.eval()

    classes = sorted(idx2label.keys())
    labels_str = [idx2label[i] for i in classes]

    # ---- Validation split metrics ----
    val_true = []
    val_pred = []
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_vec = batch["batch"].to(device)
            labels = batch["label"].to(device)
            bs = batch["batch_size"]

            logits = model(seq, edge_index, edge_attr, batch_vec, bs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            val_true.extend(labels.cpu().numpy().tolist())
            val_pred.extend(preds.cpu().numpy().tolist())
            val_probs.append(probs.cpu().numpy())

    metrics: dict = {}
    if val_probs:
        val_probs_np = np.concatenate(val_probs, axis=0)
        y_true_val = np.array(val_true, dtype=int)
        y_pred_val = np.array(val_pred, dtype=int)

        acc = accuracy_score(y_true_val, y_pred_val)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_val, y_pred_val, labels=classes, average="macro", zero_division=0
        )
        prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
            y_true_val, y_pred_val, labels=classes, average=None, zero_division=0
        )

        y_true_val_bin = label_binarize(y_true_val, classes=classes)
        try:
            roc_auc_macro = roc_auc_score(
                y_true_val_bin, val_probs_np, average="macro", multi_class="ovr"
            )
        except Exception:
            roc_auc_macro = float("nan")

        cm_val = confusion_matrix(y_true_val, y_pred_val, labels=classes)
        report_val = classification_report(
            y_true_val,
            y_pred_val,
            labels=classes,
            target_names=labels_str,
            zero_division=0,
        )

        # Per-class ROC curves (validation)
        roc_curves_val = {}
        for i, lbl in enumerate(labels_str):
            try:
                from sklearn.metrics import roc_curve as _roc_curve

                fpr, tpr, _ = _roc_curve(y_true_val_bin[:, i], val_probs_np[:, i])
                auc_i = roc_auc_score(y_true_val_bin[:, i], val_probs_np[:, i])
                roc_curves_val[lbl] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc_i),
                }
            except Exception:
                continue

        # Per-class metrics table (validation)
        per_class_val = {}
        for i, lbl in enumerate(labels_str):
            per_class_val[lbl] = {
                "precision": float(prec_cls[i]),
                "recall": float(rec_cls[i]),
                "f1": float(f1_cls[i]),
                "support": int(support_cls[i]),
            }

        # Class distributions and misclassification pairs (validation)
        true_counts_val = {
            lbl: int((y_true_val == cls_idx).sum())
            for cls_idx, lbl in zip(classes, labels_str)
        }
        pred_counts_val = {
            lbl: int((y_pred_val == cls_idx).sum())
            for cls_idx, lbl in zip(classes, labels_str)
        }
        mis_counter_val = Counter()
        for t_idx, p_idx in zip(y_true_val, y_pred_val):
            if t_idx != p_idx:
                mis_counter_val[(idx2label[t_idx], idx2label[p_idx])] += 1
        top_mis_val = [
            {"true": t, "pred": p, "count": int(c)}
            for (t, p), c in mis_counter_val.most_common(5)
        ]

        # Correlation matrix for a subset of features (on validation subset)
        df_val = df.iloc[train_size:].reset_index(drop=True)
        num_cols = [c for c in df_val.columns if df_val[c].dtype in ["float64", "float32"]]
        if len(df_val) > 3000:
            df_corr = df_val.sample(3000, random_state=42)
        else:
            df_corr = df_val
        feat_subset = feat_cols[:10] if len(feat_cols) >= 10 else feat_cols
        corr = df_corr[feat_subset].corr()

        # Top-level metrics correspond to validation split
        metrics.update(
            {
                "accuracy": float(acc),
                "precision_macro": float(prec_macro),
                "recall_macro": float(rec_macro),
                "f1_macro": float(f1_macro),
                "roc_auc_macro": float(roc_auc_macro),
                "labels": labels_str,
                "confusion": cm_val.tolist(),
                "classification_report": report_val,
                "roc_curves": roc_curves_val,
                "corr_features": feat_subset,
                "corr_matrix": corr.values.tolist(),
                "per_class": per_class_val,
                "class_dist_true": true_counts_val,
                "class_dist_pred": pred_counts_val,
                "top_misclass_pairs": top_mis_val,
            }
        )

    # ---- Training split basic metrics ----
    train_true = []
    train_pred = []
    with torch.no_grad():
        for batch in train_loader:
            seq = batch["seq"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_vec = batch["batch"].to(device)
            labels = batch["label"].to(device)
            bs = batch["batch_size"]

            logits = model(seq, edge_index, edge_attr, batch_vec, bs)
            preds = logits.argmax(dim=1)

            train_true.extend(labels.cpu().numpy().tolist())
            train_pred.extend(preds.cpu().numpy().tolist())

    if train_true:
        y_true_tr = np.array(train_true, dtype=int)
        y_pred_tr = np.array(train_pred, dtype=int)

        acc_tr = accuracy_score(y_true_tr, y_pred_tr)
        prec_macro_tr, rec_macro_tr, f1_macro_tr, _ = precision_recall_fscore_support(
            y_true_tr, y_pred_tr, labels=classes, average="macro", zero_division=0
        )
        prec_cls_tr, rec_cls_tr, f1_cls_tr, support_cls_tr = precision_recall_fscore_support(
            y_true_tr, y_pred_tr, labels=classes, average=None, zero_division=0
        )

        cm_tr = confusion_matrix(y_true_tr, y_pred_tr, labels=classes)
        per_class_tr = {}
        for i, lbl in enumerate(labels_str):
            per_class_tr[lbl] = {
                "precision": float(prec_cls_tr[i]),
                "recall": float(rec_cls_tr[i]),
                "f1": float(f1_cls_tr[i]),
                "support": int(support_cls_tr[i]),
            }

        true_counts_tr = {
            lbl: int((y_true_tr == cls_idx).sum())
            for cls_idx, lbl in zip(classes, labels_str)
        }
        pred_counts_tr = {
            lbl: int((y_pred_tr == cls_idx).sum())
            for cls_idx, lbl in zip(classes, labels_str)
        }
        mis_counter_tr = Counter()
        for t_idx, p_idx in zip(y_true_tr, y_pred_tr):
            if t_idx != p_idx:
                mis_counter_tr[(idx2label[t_idx], idx2label[p_idx])] += 1
        top_mis_tr = [
            {"true": t, "pred": p, "count": int(c)}
            for (t, p), c in mis_counter_tr.most_common(5)
        ]

        metrics["train_split"] = {
            "accuracy": float(acc_tr),
            "precision_macro": float(prec_macro_tr),
            "recall_macro": float(rec_macro_tr),
            "f1_macro": float(f1_macro_tr),
            "confusion": cm_tr.tolist(),
            "per_class": per_class_tr,
            "class_dist_true": true_counts_tr,
            "class_dist_pred": pred_counts_tr,
            "top_misclass_pairs": top_mis_tr,
        }

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "num_classes": num_classes,
            "num_features": num_features,
            "epochs": EPOCHS,
        }, f, indent=2)
    if metrics:
        with open(run_dir / "train_eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
