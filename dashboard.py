"""Streamlit dashboard for hybrid IDS: data exploration, model metrics, and predictions."""
import gc
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader

from preprocess_cicids import (
    clean_column_names,
    drop_identifier_and_duplicates,
    coerce_numerics_and_handle_infinite,
    apply_cicids_trafficforml_column_aliases,
    assign_synthetic_flow_endpoints_if_missing,
    ensure_numeric_feature_columns_present,
)
from precompute_training_cache import precompute_hybrid_window_cache
from src.data_loader import CachedHybridIDSDataset, collate_hybrid
from src.model import HybridIDSModel
from src.config import GRAPH_SIZE, SEQ_LEN, WINDOW_STRIDE

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
SYNTHETIC_DIR = ROOT / "synthetic_data"
PRIMITIVE_DIR = ROOT / "analysis_outputs" / "primitive"

st.set_page_config(page_title="Hybrid IDS Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1rem; border-radius: 0.5rem; color: white; margin: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_processed_data_for_view(mode_key: str):
    """Load post-preprocessed CICIDS 2017 table for dashboard analysis."""
    if mode_key == "binary":
        pq = PROCESSED_DIR / "traffic_all_binary_processed.parquet"
        csv = PROCESSED_DIR / "traffic_all_binary_processed.csv"
    else:
        pq = PROCESSED_DIR / "traffic_all_processed.parquet"
        csv = PROCESSED_DIR / "traffic_all_processed.csv"

    if pq.exists():
        return pd.read_parquet(pq), pq
    if csv.exists():
        return pd.read_csv(csv, low_memory=False), csv
    return None, None


@st.cache_data
def load_training_history():
    if not MODELS_DIR.exists():
        return None, None
    runs = sorted(MODELS_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    for run_dir in runs[:5]:
        hist_path = run_dir / "history.json"
        cfg_path = run_dir / "config.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            cfg = {}
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
            return hist, cfg
    return None, None


@st.cache_resource
def load_preprocess_artifacts():
    """Load scaler and preprocessing artifacts for inference-time preprocessing."""
    artifacts_path = PROCESSED_DIR / "scaler.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found at {artifacts_path}. "
            "Run `python preprocess_cicids.py` first."
        )
    return joblib.load(artifacts_path)


@st.cache_resource
def load_preprocess_artifacts_binary():
    artifacts_path = PROCESSED_DIR / "scaler_binary.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Binary preprocessing artifacts not found at {artifacts_path}. "
            "Run `python preprocess_cicids_binary.py` first."
        )
    return joblib.load(artifacts_path)


def get_runs_by_mode(mode: str):
    """
    mode:
      - 'multi' => num_classes > 2 (or non-binary naming fallback)
      - 'binary' => num_classes == 2 or run name endswith _binary
    """
    if not MODELS_DIR.exists():
        return []
    runs = []
    for p in sorted(MODELS_DIR.iterdir(), key=lambda x: x.name, reverse=True):
        if not (p / "best_model.pt").exists():
            continue
        cfg_path = p / "config.json"
        num_classes = None
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                    num_classes = int(cfg.get("num_classes", -1))
            except Exception:
                num_classes = None
        is_binary = (num_classes == 2) or p.name.endswith("_binary")
        if mode == "binary" and is_binary:
            runs.append(p.name)
        if mode == "multi" and not is_binary:
            runs.append(p.name)
    return runs


def benign_first_class_order(labels: list[str]) -> list[str]:
    """Order classes for bar charts: BENIGN first when present, then others in original order."""
    ls = list(labels)
    if "BENIGN" not in ls:
        return ls
    rest = [x for x in ls if x != "BENIGN"]
    return ["BENIGN"] + rest


def _inference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_inference_model(run_name: str):
    """Load trained HybridIDSModel and metadata; weights on CPU until moved to device for inference."""
    run_dir = MODELS_DIR / run_name
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt found in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    label2idx = ckpt["label2idx"]
    idx2label = ckpt["idx2label"]
    feat_cols = ckpt["feat_cols"]

    model = HybridIDSModel(
        num_features=len(feat_cols),
        num_classes=len(label2idx),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, label2idx, idx2label, feat_cols


def run_predictions_on_df(
    df_pred: pd.DataFrame,
    run_name: str,
    max_windows: int = 1000,
    mode: str = "multi",
    progress_bar: Optional[Any] = None,
    status_callback: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess uploaded data and run window-level predictions.

    Returns:
        pred_counts, preview, meta (timing, device, row/window counts)
    """
    t0 = time.perf_counter()
    model, label2idx, idx2label, feat_cols = load_inference_model(run_name)
    device = _inference_device()
    model = model.to(device)

    # 1) Apply the same cleaning steps as training (+ CICIDS 2018 TrafficForML name alignment)
    if status_callback:
        status_callback("Cleaning columns and aligning CICIDS 2018 names (if needed)…")
    df_raw = df_pred.copy()
    df_clean = clean_column_names(df_raw)
    df_clean = apply_cicids_trafficforml_column_aliases(df_clean)
    df_clean, _ = drop_identifier_and_duplicates(df_clean)
    df_clean = coerce_numerics_and_handle_infinite(df_clean)

    # 2) Load training-time scaler and imputation stats
    if status_callback:
        status_callback("Applying training scaler and building flow windows…")
    artifacts = load_preprocess_artifacts() if mode == "multi" else load_preprocess_artifacts_binary()
    numeric_feature_names = artifacts["numeric_feature_names"]
    imputation_medians = artifacts.get("imputation_medians", {})
    scaler = artifacts["scaler"]

    # Scale using the full column set the StandardScaler was fit on (order + count must match).
    # The checkpoint's feat_cols can be a subset (e.g. older run); using only that subset breaks
    # after re-running preprocess with a wider numeric set.
    scale_cols = list(numeric_feature_names)
    missing_model = [c for c in feat_cols if c not in numeric_feature_names]
    if missing_model:
        raise ValueError(
            "This model's saved feature list does not match the current preprocessing artifacts. "
            "Either select a model trained after the latest preprocess, or re-run training. "
            f"Examples of checkpoint features not in scaler: {', '.join(missing_model[:10])}"
        )

    df_clean = ensure_numeric_feature_columns_present(df_clean, scale_cols)
    missing = [c for c in scale_cols if c not in df_clean.columns]
    if missing:
        raise ValueError(
            "Uploaded dataset is missing required feature columns after preprocessing. "
            f"Examples of missing columns: {', '.join(missing[:10])}"
        )

    for col in scale_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        median_val = imputation_medians.get(col, 0.0)
        df_clean[col] = df_clean[col].fillna(median_val)

    scaled_values = scaler.transform(df_clean[scale_cols].astype(np.float32).values)
    df_clean[scale_cols] = scaled_values

    df_use = df_clean
    if "Label" not in df_use.columns:
        df_use["Label"] = "BENIGN"
    # CICIDS 2018 TrafficForML exports often omit endpoint IPs; graphs require Source/Destination IP.
    if "SourceFile" not in df_use.columns:
        df_use["SourceFile"] = "uploaded_sample.csv"
    df_use = assign_synthetic_flow_endpoints_if_missing(df_use)

    n_rows = len(df_use)
    base = max(1, n_rows - SEQ_LEN - GRAPH_SIZE)
    n_win_full = (base + WINDOW_STRIDE - 1) // WINDOW_STRIDE
    max_windows_eff = min(max_windows, n_win_full)

    idx2label_for_cache = None
    if idx2label:
        idx2label_for_cache = {int(k): v for k, v in idx2label.items()}

    cache_dir = Path(tempfile.mkdtemp(prefix="pred_hybrid_cache_"))
    dataset = None
    all_preds: list[int] = []
    n_win_scored = 0
    n_batches = 0
    batch_size = 8
    t_preprocess = 0.0
    t_infer = 0.0
    try:
        if status_callback:
            status_callback(
                "Precomputing window + graph cache (same format as training cache; one-time cost per upload)…"
            )
        precompute_hybrid_window_cache(
            df_use,
            feat_cols,
            label2idx,
            cache_dir,
            SEQ_LEN,
            GRAPH_SIZE,
            WINDOW_STRIDE,
            64,
            idx2label=idx2label_for_cache,
            max_windows_cap=max_windows_eff,
            quiet=True,
        )
        dataset = CachedHybridIDSDataset(cache_dir)
        n_win_scored = len(dataset)

        # Larger batches on GPU; cap so small jobs still batch efficiently
        if device.type == "cuda":
            batch_size = min(512, max(32, n_win_scored))
        else:
            batch_size = min(256, max(32, n_win_scored))
        batch_size = max(8, min(batch_size, n_win_scored))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_hybrid,
            num_workers=0,
        )
        n_batches = len(loader)
        t_preprocess = time.perf_counter() - t0

        if status_callback:
            status_callback(
                f"Running model on **{device}** — {n_win_scored:,} cached windows in {n_batches:,} batches…"
            )

        t_infer0 = time.perf_counter()
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                seq = batch["seq"].to(device, non_blocking=True)
                edge_index = batch["edge_index"].to(device, non_blocking=True)
                edge_attr = batch["edge_attr"].to(device, non_blocking=True)
                batch_vec = batch["batch"].to(device, non_blocking=True)
                bs = batch["batch_size"]
                logits = model(seq, edge_index, edge_attr, batch_vec, bs)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(pred)
                if progress_bar is not None and n_batches:
                    progress_bar.progress((bi + 1) / n_batches)

        t_infer = time.perf_counter() - t_infer0
    finally:
        if dataset is not None:
            del dataset
        gc.collect()
        shutil.rmtree(cache_dir, ignore_errors=True)

    model.cpu()

    pred_labels = [idx2label.get(int(p), "UNK") for p in all_preds]
    pred_counts = (
        pd.Series(pred_labels)
        .value_counts()
        .rename_axis("Predicted label")
        .reset_index(name="Count")
    )
    preview = pd.DataFrame(
        {"window_index": list(range(len(pred_labels))), "predicted_label": pred_labels}
    )
    meta = {
        "n_rows": int(n_rows),
        "n_windows_available": int(n_win_full),
        "n_windows_scored": int(n_win_scored),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "device": str(device),
        "seconds_preprocess": float(t_preprocess),
        "seconds_inference": float(t_infer),
        "seconds_total": float(time.perf_counter() - t0),
    }
    return pred_counts, preview, meta


def main():
    st.markdown('<p class="main-header">🛡️ Hybrid IDS – Zero-Day Attack Detection</p>', unsafe_allow_html=True)
    st.markdown("Temporal (Transformer) + Structural (GNN) + Few-shot Learning")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧾 Primitive Data Analysis",
        "📋 Preprocessing",
        "📈 Training",
        "📊 Predictions on CICIDS 2018",
        "🔍 Predictions",
        "🧪 Zero Day Eval",
    ])

    with tab1:
        st.subheader("Primitive Data Analysis (Precomputed)")
        st.caption(
            "This pane only loads precomputed analysis outputs to keep the frontend lightweight."
        )

        summary_path = PRIMITIVE_DIR / "summary.json"
        if not summary_path.exists():
            st.warning(
                "Primitive analysis not found. Run `python primitive_data_analysis.py` first."
            )
        else:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows", f"{summary['shape']['rows']:,}")
            with c2:
                st.metric("Columns", f"{summary['shape']['cols']:,}")
            with c3:
                st.metric("Duplicate rows", f"{summary['duplicate_rows']:,}")
            with c4:
                st.metric("Missing-value columns", summary["num_missing_columns"])

            st.subheader("Basic Dataset Understanding")
            st.markdown("**df.shape**")
            st.code(f"({summary['shape']['rows']}, {summary['shape']['cols']})")
            st.markdown("**df.columns**")
            st.dataframe(pd.DataFrame({"column": summary["columns"]}))
            st.markdown("**df.info()**")
            info_path = PRIMITIVE_DIR / "df_info.txt"
            if info_path.exists():
                st.code(info_path.read_text(encoding="utf-8"), language="text")
            st.markdown("**df.describe()**")
            desc_path = PRIMITIVE_DIR / "df_describe.csv"
            if desc_path.exists():
                st.dataframe(pd.read_csv(desc_path))

            st.subheader("Missing Values Analysis")
            missing_path = PRIMITIVE_DIR / "missing_values.csv"
            if missing_path.exists():
                st.dataframe(pd.read_csv(missing_path).head(100))

            st.subheader("Duplicate Rows")
            st.code(f"df.duplicated().sum() = {summary['duplicate_rows']}")

            st.subheader("Label Distribution (VERY IMPORTANT)")
            label_path = PRIMITIVE_DIR / "label_counts.csv"
            if label_path.exists():
                label_df = pd.read_csv(label_path)
                st.dataframe(label_df)
                fig = px.bar(label_df, x="Label", y="count", title="Label count")
                st.plotly_chart(fig, use_container_width=True)
            label_plot = PRIMITIVE_DIR / "label_countplot.png"
            if label_plot.exists():
                st.image(str(label_plot), caption="sns.countplot(x='Label', data=df)")

            st.subheader("Correlation Analysis")
            corr_path = PRIMITIVE_DIR / "correlation_matrix.csv"
            if corr_path.exists():
                st.caption("Showing top-left section of full correlation matrix.")
                corr_df = pd.read_csv(corr_path, index_col=0)
                st.dataframe(corr_df.iloc[:30, :30])
            corr_plot = PRIMITIVE_DIR / "correlation_heatmap.png"
            if corr_plot.exists():
                st.image(str(corr_plot), caption="sns.heatmap(corr, cmap='coolwarm')")

            st.subheader("Feature Importance (RandomForest)")
            fi_path = PRIMITIVE_DIR / "feature_importances.csv"
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path)
                st.dataframe(fi_df.head(50))
                fig = px.bar(
                    fi_df.head(20).iloc[::-1],
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 20 feature importances",
                )
                st.plotly_chart(fig, use_container_width=True)
            fi_plot = PRIMITIVE_DIR / "feature_importances_top20.png"
            if fi_plot.exists():
                st.image(str(fi_plot), caption="RandomForest feature importance (top 20)")

    with tab3:
        train_mode = st.radio(
            "Training view",
            ["Multi-attack model (A)", "Binary model (B)"],
            index=0,
            horizontal=True,
            key="train_mode",
        )
        is_binary_train = "Binary model (B)" in train_mode
        hist_metric_file = "train_eval_metrics_binary.json" if is_binary_train else "train_eval_metrics.json"
        train_runs = get_runs_by_mode("binary" if is_binary_train else "multi")

        st.subheader("Training Curves")
        hist = None
        cfg = None
        if train_runs:
            default_run = train_runs[0]
            run_for_curves = st.selectbox(
                "Select run for training curves",
                train_runs,
                key="train_curves_run_binary" if is_binary_train else "train_curves_run_multi",
                index=0,
            )
            hist_path = MODELS_DIR / run_for_curves / "history.json"
            cfg_path = MODELS_DIR / run_for_curves / "config.json"
            if hist_path.exists():
                try:
                    with open(hist_path) as f:
                        hist = json.load(f)
                except Exception:
                    hist = None
            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = None
        else:
            default_run = None

        if hist is None:
            st.info("No training history found for this model type yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["train_loss"], name="Train Loss", mode="lines"))
                fig.add_trace(go.Scatter(y=hist["val_loss"], name="Val Loss", mode="lines"))
                fig.update_layout(title="Loss", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["train_acc"], name="Train Acc", mode="lines"))
                fig.add_trace(go.Scatter(y=hist["val_acc"], name="Val Acc", mode="lines"))
                fig.update_layout(title="Accuracy", height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Validation Metrics & Visualizations")
        metric_runs = []
        if MODELS_DIR.exists():
            metric_runs = [
                p.name
                for p in sorted(MODELS_DIR.iterdir(), key=lambda x: x.name, reverse=True)
                if p.name in train_runs and (p / hist_metric_file).exists()
            ]
        if not metric_runs:
            st.info(
                f"No detailed metrics found for this model type. "
                f"Expected `{hist_metric_file}` in a run folder."
            )
        else:
            run_name = st.selectbox(
                "Select run for detailed metrics",
                metric_runs,
                key="train_metrics_run_binary" if is_binary_train else "train_metrics_run_multi",
            )
            if run_name:
                metrics_path = MODELS_DIR / run_name / hist_metric_file
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                except Exception as e:
                    st.error(f"Failed to load metrics for {run_name}: {e}")
                    metrics = None

        if metric_runs and run_name and metrics:
            # Metric cards (validation)
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.metric("Val Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with mcol2:
                st.metric("Val Macro Precision", f"{metrics.get('precision_macro', 0):.2%}")
            with mcol3:
                st.metric("Val Macro Recall", f"{metrics.get('recall_macro', 0):.2%}")
            with mcol4:
                st.metric("Val Macro F1", f"{metrics.get('f1_macro', 0):.2%}")
            with mcol5:
                roc_auc_val = metrics.get("roc_auc_macro")
                st.metric("Val Macro ROC AUC", "N/A" if roc_auc_val is None else f"{roc_auc_val:.2f}")

            # Per-class table (validation + training in the same shape)
            st.subheader("Per-class Precision / Recall / F1")
            per_val = pd.DataFrame.from_dict(metrics.get("per_class", {}), orient="index")
            per_val.index.name = "Class"
            per_val.reset_index(inplace=True)

            train_split = metrics.get("train_split")
            if train_split and "per_class" in train_split:
                per_train = pd.DataFrame.from_dict(train_split["per_class"], orient="index")
                per_train.index.name = "Class"
                per_train.reset_index(inplace=True)
                per_val = per_val.merge(per_train, on="Class", suffixes=("_val", "_train"))
            else:
                per_val["precision_train"] = np.nan
                per_val["recall_train"] = np.nan
                per_val["f1_train"] = np.nan
                per_val["support_train"] = np.nan

            st.dataframe(per_val)

            # Normalized confusion matrix (row-wise)
            st.subheader("Normalized Confusion Matrix (row-wise)")
            cm = np.array(metrics["confusion"], dtype=float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_norm = cm / row_sums
            labels_str = metrics["labels"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_norm,
                    x=labels_str,
                    y=labels_str,
                    colorscale="Blues",
                    colorbar=dict(title="Row-normalized"),
                )
            )
            fig_cm.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Class distribution train vs validation (true and predicted)
            st.subheader("Class Distribution (Train vs Validation)")
            dist_val_true = metrics.get("class_dist_true", {})
            dist_val_pred = metrics.get("class_dist_pred", {})
            # Fallback from validation confusion when explicit dicts are absent.
            if (not dist_val_true or not dist_val_pred) and cm.size > 0:
                dist_val_true = {lbl: int(cm[idx, :].sum()) for idx, lbl in enumerate(labels_str)}
                dist_val_pred = {lbl: int(cm[:, idx].sum()) for idx, lbl in enumerate(labels_str)}
            dist_train_true = {}
            dist_train_pred = {}
            if train_split:
                dist_train_true = train_split.get("class_dist_true", {})
                dist_train_pred = train_split.get("class_dist_pred", {})

            chart_labels = benign_first_class_order(labels_str)
            df_dist = pd.DataFrame(
                {
                    "Class": chart_labels,
                    "Train True": [dist_train_true.get(lbl, 0) for lbl in chart_labels],
                    "Train Pred": [dist_train_pred.get(lbl, 0) for lbl in chart_labels],
                    "Val True": [dist_val_true.get(lbl, 0) for lbl in chart_labels],
                    "Val Pred": [dist_val_pred.get(lbl, 0) for lbl in chart_labels],
                }
            )
            fig_dist = px.bar(
                df_dist.melt(id_vars="Class", var_name="Split/Type", value_name="Count"),
                x="Class",
                y="Count",
                color="Split/Type",
                barmode="group",
                title="Class distribution (true vs predicted)",
                category_orders={"Class": chart_labels},
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            

            # Correlation matrix heatmap
            if "corr_features" in metrics and "corr_matrix" in metrics:
                st.subheader("Feature Correlation (subset)")
                corr_features = metrics["corr_features"]
                corr_matrix = np.array(metrics["corr_matrix"])
                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_features,
                        y=corr_features,
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Feature correlation subset was not saved for this run.")


            # Top misclassification pairs (validation and training)
            st.subheader("Top Misclassification Pairs (true → predicted)")
            top_mis_val = metrics.get("top_misclass_pairs", [])
            if not top_mis_val and cm.size > 0:
                pairs = []
                for i, t_lbl in enumerate(labels_str):
                    for j, p_lbl in enumerate(labels_str):
                        if i != j and cm[i, j] > 0:
                            pairs.append({"true": t_lbl, "pred": p_lbl, "count": int(cm[i, j])})
                pairs = sorted(pairs, key=lambda x: x["count"], reverse=True)[:5]
                top_mis_val = pairs
            st.markdown("**Validation:**")
            st.dataframe(pd.DataFrame(top_mis_val))
            if train_split and "top_misclass_pairs" in train_split:
                st.markdown("**Training:**")
                st.dataframe(pd.DataFrame(train_split["top_misclass_pairs"]))
            else:
                st.info("Training misclassification pairs not available for this run.")

    with tab5:
        st.subheader("Model Predictions")
        pred_mode = st.radio(
            "Prediction mode",
            ["Multi-attack (3-class model)", "All attacks combined (binary)"],
            index=0,
            horizontal=True,
        )
        mode_key = "multi" if "Multi-attack" in pred_mode else "binary"
        st.caption(
            f"Weights are loaded from **`{MODELS_DIR.resolve()}`** (each run folder has `best_model.pt`). "
            "**Multi-attack** uses preprocessing artifacts `processed/scaler.joblib` (from "
            "`preprocess_cicids.py`). **Binary** uses `processed/scaler_binary.joblib` (from "
            "`preprocess_cicids_binary.py`). Switch the mode below, choose a run of that type, "
            "then upload once — the same CICIDS-style file works for both."
        )
        st.markdown(
            "**Dataset requirements for correct predictions:**\n"
            "- Ideally CICIDS-style flow data (like the `TrafficLabelling` CSVs).\n"
            "- Our pipeline will clean columns, drop `Flow ID`, handle `Infinity`, impute missing values, "
            "and apply the same scaler used during training.\n"
            "- The uploaded data must contain the numeric feature columns used during training "
            "(we will check and show an error if key columns are missing).\n"
            "- Optional but recommended: `Source IP`, `Destination IP`, `Timestamp` for richer temporal/graph context "
            "(TrafficForML exports often omit IPs; the pipeline can synthesize placeholders).\n"
            "- Accepted formats: `.csv` (raw CICIDS flows) or `.parquet` with similar columns."
        )

        runs = get_runs_by_mode(mode_key)
        _pred_options = ["(none)"] + runs
        run_sel = st.selectbox(
            "Trained model run",
            _pred_options,
            index=1 if runs else 0,
            key=f"pred_run_{mode_key}",
            help="Runs are newest-first; the latest compatible run is selected by default. "
            "Multi-class and binary selections are remembered separately when you switch mode.",
        )

        uploaded = st.file_uploader(
            "Upload dataset file (.parquet or .csv) for prediction",
            type=["parquet", "csv"],
        )
        max_windows = st.slider(
            "Max windows to score",
            min_value=100,
            max_value=5000,
            value=4500,
            step=100,
            help="Each window is a short sequence of flows; larger values take longer.",
        )
        st.caption(
            "**Windows are not rows.** The model scores **sliding windows** over time-sorted flows "
            f"(see seq_len={SEQ_LEN} and graph span). The full upload is loaded, but only the **first** "
            f"{max_windows} windows are run through the model—so with max_windows=5000 you get at most "
            "5000 predictions, not “5000 rows only,” and rows after the early part of the timeline are "
            "not used unless you raise this cap. "
            "**Row order matters:** flows should follow capture time (as in the original CSVs). "
            "If rows are randomly shuffled, every window mixes traffic types and predictions collapse toward BENIGN."
        )

        dev_hint = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(
            f"Predictions start only after you click **Run predictions**. "
            f"Runtime scales with **max windows** (not file size alone). "
            f"This machine will use **{dev_hint}** for inference. "
            "Rough guide: hundreds of windows per second on a typical GPU vs slower on CPU."
        )

        can_run = run_sel != "(none)" and uploaded is not None
        upload_sig = ""
        if uploaded is not None:
            upload_sig = f"{uploaded.name}|{getattr(uploaded, 'size', 0)}"
        pred_ctx = f"{upload_sig}|{run_sel}|{max_windows}|{mode_key}"

        run_clicked = st.button(
            "Run predictions",
            type="primary",
            disabled=not can_run,
            key=f"pred_run_btn_{mode_key}",
        )

        if run_clicked and can_run:
            try:
                if uploaded.name.endswith(".parquet"):
                    df_pred = pd.read_parquet(uploaded)
                else:
                    df_pred = pd.read_csv(uploaded, low_memory=False)
            except Exception as e:
                st.error(f"Failed to read file: {e}")
            else:
                status_ph = st.empty()
                progress = st.progress(0.0)
                est_batches = max(1, math.ceil(min(max_windows, max(1, len(df_pred) - SEQ_LEN - 64)) / (512 if torch.cuda.is_available() else 256)))
                status_ph.info(
                    f"Estimated **~{est_batches}+** inference batches for up to **{max_windows:,}** windows "
                    f"({dev_hint}). First run includes model load and preprocessing."
                )

                def _cb(msg: str) -> None:
                    status_ph.info(msg)

                try:
                    pred_counts, preview, meta = run_predictions_on_df(
                        df_pred,
                        run_sel,
                        max_windows=max_windows,
                        mode=mode_key,
                        progress_bar=progress,
                        status_callback=_cb,
                    )
                except ValueError as e:
                    progress.progress(1.0)
                    st.session_state.pop("pred_display", None)
                    st.error(str(e))
                except Exception as e:
                    progress.progress(1.0)
                    st.session_state.pop("pred_display", None)
                    st.error(f"Prediction failed: {e}")
                else:
                    progress.progress(1.0)
                    st.session_state["pred_display"] = {
                        "ctx": pred_ctx,
                        "counts": pred_counts,
                        "preview": preview,
                        "meta": meta,
                    }

        disp = st.session_state.get("pred_display")
        
        # if disp and disp.get("ctx") == pred_ctx:
        #     meta = disp["meta"]
        #     pred_counts = disp["counts"]
        #     preview = disp["preview"]
        #     st.success(
        #         f"Scored **{meta['n_windows_scored']:,}** windows "
        #         f"(of **{meta['n_windows_available']:,}** available for this file’s length). "
        #         f"**{meta['n_rows']:,}** rows loaded. "
        #         f"Device: **{meta['device']}**, batch size **{meta['batch_size']}**."
        #     )
        #     t1 = meta["seconds_preprocess"]
        #     t2 = meta["seconds_inference"]
        #     tt = meta["seconds_total"]
        #     st.caption(
        #         f"Timing: preprocessing **{t1:.2f}s** · inference **{t2:.2f}s** · total **{tt:.2f}s**"
        #     )
        #     fig = px.bar(
        #         pred_counts,
        #         x="Predicted label",
        #         y="Count",
        #         title="Predicted label distribution",
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        #     st.subheader("Sample predictions (window-level)")
        #     st.dataframe(preview.head(50))
        if disp and disp.get("ctx") == pred_ctx:
            meta = disp["meta"]
            pred_counts = disp["counts"]
            preview = disp["preview"]

            st.success(
                f"Scored **{meta['n_windows_scored']:,}** windows "
                f"(of **{meta['n_windows_available']:,}** available for this file’s length). "
                f"**{meta['n_rows']:,}** rows loaded. "
                f"Device: **{meta['device']}**, batch size **{meta['batch_size']}**."
            )

            t1 = meta["seconds_preprocess"]
            t2 = meta["seconds_inference"]
            tt = meta["seconds_total"]

            st.caption(
                f"Timing: preprocessing **{t1:.2f}s** · inference **{t2:.2f}s** · total **{tt:.2f}s**"
            )

            # =====================================================
            # ORIGINAL DISTRIBUTION CHART
            # =====================================================

            fig = px.bar(
                pred_counts,
                x="Predicted label",
                y="Count",
                title="Predicted Label Distribution",
                color="Predicted label",
            )
            st.plotly_chart(fig, use_container_width=True)

            # =====================================================
            # SECURITY RISK ASSESSMENT
            # =====================================================

            total_windows = int(pred_counts["Count"].sum())

            attack_windows = int(
                pred_counts[
                    pred_counts["Predicted label"]
                    .astype(str)
                    .str.upper()
                    .ne("BENIGN")
                ]["Count"].sum()
            )

            risk_score = (
                attack_windows / total_windows * 100
                if total_windows > 0
                else 0
            )

            st.markdown("---")
            st.subheader("🛡 Security Risk Assessment")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Windows", f"{total_windows:,}")

            with col2:
                st.metric("Attack Windows", f"{attack_windows:,}")

            with col3:
                st.metric("Risk Score", f"{risk_score:.1f}%")

            with col4:
                non_benign = pred_counts[
                    pred_counts["Predicted label"]
                    .astype(str)
                    .str.upper()
                    .ne("BENIGN")
                ]

                if len(non_benign):
                    top_attack = non_benign.iloc[0]["Predicted label"]
                    st.metric("Top Attack", top_attack)
                else:
                    st.metric("Top Attack", "None")

            if risk_score < 10:
                st.success(
                    "🟢 Network appears safe. Very little malicious activity detected."
                )
            elif risk_score < 40:
                st.warning(
                    "🟡 Suspicious traffic detected. Further investigation recommended."
                )
            else:
                st.error(
                    "🔴 High attack activity detected. Immediate investigation recommended."
                )

            # =====================================================
            # THREAT LEVEL GAUGE
            # =====================================================

            st.subheader("🚨 Threat Level")

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={"text": "Threat Score (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "lightgreen",
                            },
                            {
                                "range": [30, 70],
                                "color": "gold",
                            },
                            {
                                "range": [70, 100],
                                "color": "salmon",
                            },
                        ],
                    },
                )
            )

            gauge_fig.update_layout(height=350)

            st.plotly_chart(gauge_fig, use_container_width=True)

            # =====================================================
            # TIMELINE + PIE CHART
            # =====================================================

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📈 Attack Timeline")

                timeline_df = preview.copy()

                timeline_df["attack_flag"] = (
                    timeline_df["predicted_label"]
                    .astype(str)
                    .str.upper()
                    .ne("BENIGN")
                ).astype(int)

                timeline_fig = px.line(
                    timeline_df,
                    x="window_index",
                    y="attack_flag",
                    title="Attack Presence Across Windows",
                )

                timeline_fig.update_yaxes(
                    tickvals=[0, 1],
                    ticktext=["Benign", "Attack"],
                )

                st.plotly_chart(
                    timeline_fig,
                    use_container_width=True,
                )

            with col_right:
                st.subheader("🥧 Traffic Composition")

                pie_fig = px.pie(
                    pred_counts,
                    values="Count",
                    names="Predicted label",
                    title="Traffic Composition",
                )

                st.plotly_chart(
                    pie_fig,
                    use_container_width=True,
                )

            # =====================================================
            # AUTOMATED SECURITY REPORT
            # =====================================================

            st.subheader("📄 Automated Security Report")

            if len(pred_counts):
                most_common_class = pred_counts.iloc[0]["Predicted label"]
            else:
                most_common_class = "Unknown"

            if risk_score < 10:
                recommendation = (
                    "Traffic appears normal. Continue routine monitoring."
                )
            elif risk_score < 40:
                recommendation = (
                    "Potential attack indicators detected. Review suspicious traffic."
                )
            else:
                recommendation = (
                    "High attack activity detected. Immediate analyst investigation recommended."
                )

            st.markdown(
                f"""
        ### Executive Summary

        **Traffic Analysed:** {total_windows:,} windows

        **Attack Windows Detected:** {attack_windows:,}

        **Most Common Prediction:** {most_common_class}

        **Threat Score:** {risk_score:.2f}%

        ### Recommendation

        {recommendation}
        """
            )

            # =====================================================
            # SAMPLE PREDICTIONS
            # =====================================================

            st.subheader("Sample Predictions (Window-Level)")
            st.dataframe(preview.head(50), use_container_width=True)

        elif run_sel == "(none)":
            st.info("Select a trained model run, upload a file, then click **Run predictions**.")
        elif uploaded is None:
            st.info("Upload a `.csv` or `.parquet` file, then click **Run predictions**.")

    with tab4:
        st.subheader("CICIDS 2018 Evaluation (Precomputed)")
        eval_mode = st.radio(
            "Evaluation type",
            ["Multi-attack (2018 version 2)", "Binary all-attacks (2018 version 1)"],
            index=1,
            horizontal=True,
            key="eval2018_mode",
        )
        if "Multi-attack" in eval_mode:
            eval_runs = [
                p.name for p in sorted(MODELS_DIR.iterdir(), key=lambda x: x.name, reverse=True)
                if (p / "cicids2018_eval.json").exists()
            ] if MODELS_DIR.exists() else []
            eval_file = "cicids2018_eval.json"
        else:
            eval_runs = [
                p.name for p in sorted(MODELS_DIR.iterdir(), key=lambda x: x.name, reverse=True)
                if (p / "cicids2018_eval_binary.json").exists()
            ] if MODELS_DIR.exists() else []
            eval_file = "cicids2018_eval_binary.json"

        if not eval_runs:
            st.info(
                f"No 2018 evaluation metrics found for selected mode. "
                f"Expected file: `{eval_file}` inside a model run directory."
            )
        else:
            run_sel_2018 = st.selectbox("Model run", eval_runs, key="eval2018_run")
            path = MODELS_DIR / run_sel_2018 / eval_file
            with open(path) as f:
                m = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
            c2.metric("Macro Precision", f"{m.get('precision_macro', 0):.2%}")
            c3.metric("Macro Recall", f"{m.get('recall_macro', 0):.2%}")
            c4.metric("Macro F1", f"{m.get('f1_macro', 0):.2%}")
            st.caption(f"Windows evaluated: {m.get('num_windows', 0):,}")

            st.subheader("Per-class metrics")
            df_pc = pd.DataFrame.from_dict(m.get("per_class", {}), orient="index")
            df_pc.index.name = "Class"
            st.dataframe(df_pc.reset_index())

            st.subheader("Normalized Confusion Matrix")
            cm = np.array(m.get("confusion", []), dtype=float)
            labels = m.get("labels", [])
            if cm.size > 0:
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                cm_norm = cm / row_sums
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm_norm,
                        x=labels,
                        y=labels,
                        colorscale="Blues",
                        colorbar=dict(title="Row-normalized"),
                    )
                )
                fig_cm.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_cm, use_container_width=True)

            if m.get("classification_report"):
                st.subheader("Classification report")
                st.code(m["classification_report"], language="text")

    with tab6:
        st.subheader("Zero-Day Evaluation")
        st.caption("Evaluate the trained model on synthetic data (benign + known attacks + ZERO_DAY).")
        if not (SYNTHETIC_DIR / "synthetic_zeroday_eval.parquet").exists():
            st.warning(
                "Synthetic data not found. Run: `python generate_synthetic_dataset.py`"
            )
        else:
            runs = []
            if MODELS_DIR.exists():
                runs = [
                    p.name for p in sorted(MODELS_DIR.iterdir(), key=lambda x: x.name, reverse=True)
                    if (p / "best_model.pt").exists()
                ]
            _synth_opts = ["(none)"] + runs
            run_sel = st.selectbox(
                "Model run",
                _synth_opts,
                index=1 if runs else 0,
                key="synth_run",
            )
            if run_sel != "(none)":
                eval_path = MODELS_DIR / run_sel / "synthetic_eval.json"
                if eval_path.exists():
                    with open(eval_path) as f:
                        ev = json.load(f)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total samples", ev.get("total_samples", 0))
                    with col2:
                        st.metric("Known accuracy", f"{ev.get('known_accuracy', 0):.2%}")
                    with col3:
                        st.metric(
                            "Zero-day detection",
                            f"{ev.get('zero_day_detection_rate', 0):.2%}",
                            help="% of ZERO_DAY flows flagged as attack",
                        )
                    if "known_classification_report" in ev and ev["known_classification_report"]:
                        st.subheader("Classification Report (known classes)")
                        st.code(ev["known_classification_report"], language="text")
                else:
                    st.info(
                        "No synthetic eval for this run. Run: "
                        "`python eval_synthetic.py --run " + run_sel + "`"
                    )

    with tab2:
        st.subheader("Preprocessing Summary")
        summary_path = PROCESSED_DIR / "preprocessing_summary.txt"
        if summary_path.exists():
            st.code(summary_path.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("Preprocessing summary not found.")

        st.subheader("CICIDS 2017 After Preprocessing")
        st.caption(
            "Column counts differ by design: multi-attack preprocessing drops near-zero-variance and "
            "near-duplicate (highly correlated) numeric features before scaling; binary preprocessing "
            "keeps all remaining numeric flow features and adds **LabelOriginal**, so the table is wider."
        )
        view_mode = st.radio(
            "Processed dataset view",
            ["Multi-attack preprocessing output", "Binary preprocessing output"],
            index=0,
            horizontal=True,
            key="processed_view_mode",
        )
        mode_key = "binary" if "Binary" in view_mode else "multi"
        df_proc, proc_path = load_processed_data_for_view(mode_key)
        if df_proc is None:
            st.info("Processed dataset not found for selected mode.")
        else:
            st.caption(f"Loaded from: `{proc_path}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df_proc):,}")
            c2.metric("Columns", f"{len(df_proc.columns):,}")
            c3.metric("Duplicate rows", f"{int(df_proc.duplicated().sum()):,}")
            c4.metric("Missing-value columns", int(df_proc.isna().sum().gt(0).sum()))

            st.markdown("**Columns after preprocessing**")
            st.dataframe(pd.DataFrame({"column": df_proc.columns.tolist()}), use_container_width=True)

            st.markdown("**Missing values (top 30 columns)**")
            missing_df = (
                df_proc.isna().sum()
                .sort_values(ascending=False)
                .rename_axis("column")
                .reset_index(name="missing_count")
            )
            st.dataframe(missing_df.head(30), use_container_width=True)

            if "Label" in df_proc.columns:
                st.markdown("**Label distribution after preprocessing**")
                lbl = (
                    df_proc["Label"].astype(str).value_counts()
                    .rename_axis("Label")
                    .reset_index(name="count")
                )
                st.dataframe(lbl, use_container_width=True)
                st.plotly_chart(px.bar(lbl, x="Label", y="count", title="Post-preprocessing label count"), use_container_width=True)



if __name__ == "__main__":
    main()
