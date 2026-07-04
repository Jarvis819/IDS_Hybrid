"""
Evaluate trained model on synthetic zero-day dataset.

Reports accuracy on known classes and attack detection rate for ZERO_DAY (unseen) flows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
SYNTHETIC_DIR = ROOT / "synthetic_data"
MODELS_DIR = ROOT / "outputs" / "models"


def load_model_and_artifacts(run_dir: Path):
    """Load model checkpoint and metadata."""
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt


def run_eval(
    run_dir: Path,
    synthetic_path: Path = None,
    batch_size: int = 128,
    max_samples: int = None,
    device: str = None,
) -> dict:
    """Evaluate model on synthetic data. Returns metrics dict."""
    from src.data_loader import HybridIDSDataset, collate_hybrid
    from src.model import HybridIDSModel
    from src.config import SEQ_LEN

    if synthetic_path is None:
        synthetic_path = SYNTHETIC_DIR / "synthetic_zeroday_eval.parquet"
    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found at {synthetic_path}. "
            "Run: python generate_synthetic_dataset.py"
        )

    ckpt = load_model_and_artifacts(run_dir)
    label2idx = ckpt["label2idx"]
    idx2label = ckpt["idx2label"]
    feat_cols = ckpt["feat_cols"]
    num_classes = len(label2idx)

    # Build extended label map for dataset (ZERO_DAY maps to BENIGN for placeholder)
    eval_label2idx = dict(label2idx)
    if "ZERO_DAY" not in eval_label2idx:
        eval_label2idx["ZERO_DAY"] = eval_label2idx.get("BENIGN", 0)

    df = pd.read_parquet(synthetic_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    dataset = HybridIDSDataset(
        df, feat_cols, eval_label2idx, seq_len=SEQ_LEN, max_graph_flows=64
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=0
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
    all_true_labels = []
    benign_idx = label2idx.get("BENIGN", 0)

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_vec = batch["batch"].to(device)
            bs = batch["batch_size"]

            logits = model(seq, edge_index, edge_attr, batch_vec, bs)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())

            # Recover true labels from dataset (center of each window)
            for i in range(bs):
                # Approximate: each batch item corresponds to a window; we need true label
                # The dataset __getitem__ uses center_idx. We don't have direct access here.
                # Instead we'll collect true labels in a second pass.
                pass

    # Second pass: get true labels for each dataset index
    n = len(dataset)
    for idx in range(n):
        end = min(idx + SEQ_LEN, dataset.n)
        start = max(0, end - SEQ_LEN)
        center_idx = (start + end) // 2
        true_label = dataset.df.iloc[center_idx]["Label"]
        true_label = str(true_label) if not pd.isna(true_label) else "BENIGN"
        all_true_labels.append(true_label)

    all_preds = np.array(all_preds)
    pred_labels = [idx2label.get(int(p), "UNK") for p in all_preds]

    # Metrics
    known_classes = [c for c in set(all_true_labels) if c in label2idx]
    zero_day_mask = np.array([l == "ZERO_DAY" for l in all_true_labels])
    known_mask = ~zero_day_mask

    metrics = {}

    # Known classes: accuracy, classification report
    if known_mask.any():
        y_true_known = all_true_labels
        y_pred_known = pred_labels
        # For sklearn we need to filter to known-only
        known_idx = np.where(known_mask)[0]
        y_true_k = [all_true_labels[i] for i in known_idx]
        y_pred_k = [pred_labels[i] for i in known_idx]
        metrics["known_accuracy"] = np.mean([a == b for a, b in zip(y_true_k, y_pred_k)])
        metrics["known_classification_report"] = classification_report(
            y_true_k, y_pred_k, zero_division=0
        )
        metrics["known_confusion_matrix"] = confusion_matrix(y_true_k, y_pred_k).tolist()
        metrics["known_labels"] = list(set(y_true_k))

    # Zero-day: attack detection rate (% predicted as non-BENIGN)
    if zero_day_mask.any():
        zd_preds = all_preds[zero_day_mask]
        detected = np.sum(zd_preds != benign_idx)
        total_zd = len(zd_preds)
        metrics["zero_day_detection_rate"] = float(detected / total_zd) if total_zd else 0.0
        metrics["zero_day_total"] = int(total_zd)
        metrics["zero_day_detected"] = int(detected)

    # Overall
    metrics["total_samples"] = len(all_true_labels)
    metrics["n_benign"] = int(np.sum([l == "BENIGN" for l in all_true_labels]))
    metrics["n_known_attack"] = int(np.sum(known_mask) - metrics.get("n_benign", 0))
    metrics["n_zero_day"] = int(zero_day_mask.sum())

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on synthetic zero-day data")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name (e.g. 20260301_170905). Default: latest.",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=SYNTHETIC_DIR / "synthetic_zeroday_eval.parquet",
        help="Path to synthetic parquet",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples for quick eval")
    parser.add_argument("--output", type=Path, default=None, help="Save metrics JSON here")
    args = parser.parse_args()

    if args.run:
        run_dir = MODELS_DIR / args.run
    else:
        runs = sorted(MODELS_DIR.iterdir(), key=lambda p: p.name, reverse=True)
        run_dir = None
        for r in runs:
            if (r / "best_model.pt").exists():
                run_dir = r
                break
        if run_dir is None:
            print("ERROR: No trained model found. Run python run_train.py first.")
            return 1

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    print(f"Evaluating model from {run_dir.name} on synthetic data...")
    metrics = run_eval(
        run_dir,
        synthetic_path=args.synthetic,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 50)
    print("SYNTHETIC EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"  Benign: {metrics['n_benign']}, Known attack: {metrics['n_known_attack']}, Zero-day: {metrics['n_zero_day']}")
    if "known_accuracy" in metrics:
        print(f"\nKnown classes accuracy: {metrics['known_accuracy']:.4f}")
        print("\nClassification report (known classes):")
        report = metrics["known_classification_report"]
        print(report.encode("ascii", errors="replace").decode("ascii"))
    if "zero_day_detection_rate" in metrics:
        print(f"\nZero-day attack detection rate: {metrics['zero_day_detection_rate']:.4f}")
        print(f"  ({metrics['zero_day_detected']}/{metrics['zero_day_total']} ZERO_DAY flows flagged as attack)")
    print("=" * 50)

    # Save metrics (default: run_dir/synthetic_eval.json)
    out_path = args.output or run_dir / "synthetic_eval.json"
    out = {k: v for k, v in metrics.items() if k != "known_classification_report"}
    out["known_classification_report"] = metrics.get("known_classification_report", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
