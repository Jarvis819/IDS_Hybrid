# Hybrid IDS (CICIDS) — pipelines and commands

Work from the project root (`GeneratedLabelledFlows`). Use the same Python environment where `torch`, `sklearn`, and `streamlit` are installed.

---

## 0. `processed/` — caches only vs full wipe

### 0.1 Free disk: **caches only** (what “delete the caches” should mean)

To reclaim space **without** breaking the dashboard or forcing a long rebuild, delete **only** large memmap trees under `processed/cache/`, for example:

| Path (under `processed/cache/`) | Role |
|----------------------------------|------|
| `train_cache_v1/` | Multiclass training memmaps (`run_train.py`) |
| `train_cache_binary_v1/` | Binary training memmaps (`run_train_binary.py`) |
| `eval_cache_2018_multi_bal_v1/` | Optional — faster multiclass 2018 eval |
| `eval_cache_2018_binary_bal_v1/` | Optional — faster binary 2018 eval |

**Keep** unless you truly want a full rebuild: `scaler.joblib`, `scaler_binary.joblib`, `traffic_all_processed.parquet`, `traffic_all_binary_processed.parquet`, `traffic_2018_*`, summaries, and any small JSON/txt next to them. Those are what Streamlit and the preprocess/eval scripts expect after preprocessing.

### 0.2 Empty **entire** `processed/` — full rebuild (heavy, intentional only)

**Warning:** Deleting **everything** under `processed/` (parquets, scalers, all of `cache/`, 2018-aligned outputs, summaries) is **not** the same as “delete caches.” It removes artifacts the dashboard uses for predictions and forces the full pipeline below. Only do this when you mean a clean slate.

The repo still needs **raw inputs** outside `processed/`:

| Raw input | Location |
|-----------|----------|
| CICIDS 2017 flow CSVs | `TrafficLabelling/*.csv` |
| CICIDS 2018 flow CSVs | `CIC-IDS-2018-Dataset/*.csv` |

Ensure `processed/` exists (empty is fine). Then run **in this order**:

1. **`python preprocess_cicids.py`** — multiclass 2017 parquet + `scaler.joblib` (+ summaries). Do **not** set `PREPROCESS_WRITE_CSV` unless you want the extra multi‑GB `traffic_all_processed.csv`.
2. **`python preprocess_cicids_binary.py`** — binary parquet + `scaler_binary.joblib` (+ JSON summary).
3. **`python precompute_training_cache.py --cache_dir "processed/cache/train_cache_v1"`** — multiclass training memmaps (add `--force` to replace).
4. **`python precompute_training_cache.py --cache_dir "processed/cache/train_cache_binary_v1" --data_filename "traffic_all_binary_processed.parquet" --scaler_filename "scaler_binary.joblib"`** — binary training memmaps (`--force` as needed).
5. **`python preprocess_cicids_2018.py`** — needs step 1 (`scaler.joblib`); writes 2018 aligned parquets/CSVs.
6. **`python reprocess_cicids_2018_eval_balanced.py`** — balanced 2018 eval parquets/CSVs for default eval paths.
7. **`python run_train.py`** and **`python run_train_binary.py`** — need steps 3–4 caches; write `outputs/models/<run>/`.
8. **`python eval_cicids_2018.py`** / **`python eval_cicids_2018_binary.py`** — need trained runs from step 7 and data from step 6.
9. *(Optional)* **§3.5** — precompute `eval_cache_2018_*` for faster 2018 eval.

**After rebuild — save disk (optional)**

- `preprocess_cicids_2018.py` and `reprocess_cicids_2018_eval_balanced.py` always write **both** `.parquet` and `.csv`. If you only use parquets in scripts/pandas, you may **delete** the matching large `processed/traffic_2018_*.csv` files to reclaim tens of GB.
- Same idea for `traffic_all_processed.csv` / `traffic_all_binary_processed.csv` if you ever generated them (`PREPROCESS_WRITE_CSV=1` for multiclass).

---

## Large artifacts — what to keep

| Path (under `processed/` unless noted) | Typical size | Required for |
|----------------------------------------|--------------|--------------|
| `scaler.joblib` | small | **Multiclass**: dashboard predictions, `preprocess_cicids.py` output, 2018 alignment, training scaler |
| `traffic_all_processed.parquet` | ~100–250 MB | **Multiclass** training & `precompute_training_cache.py` default |
| `traffic_all_processed.csv` | ~1–2 GB | **Optional** duplicate of parquet (only if you rely on CSV readers); safe to delete if you only use `.parquet` |
| `scaler_binary.joblib` | small | **Binary** pipeline |
| `traffic_all_binary_processed.parquet` | ~100–250 MB | **Binary** training & binary cache precompute |
| `traffic_all_binary_processed.csv` | ~1–2 GB | **Optional** duplicate; delete if unused |
| `cache/train_cache_v1/` (`*.npy`, `cache_meta.json`) | many GB | **Multiclass training** (`run_train.py`). Not used by Streamlit predictions (those use a temp cache) |
| `cache/train_cache_binary_v1/` | many GB | **Binary training** (`run_train_binary.py`) |
| `outputs/models/<run>/best_model.pt` | tens–hundreds MB | **Inference** (dashboard, eval scripts) |
| `traffic_2018_*.parquet` / CSV | varies | **CLI eval on CICIDS 2018** (`eval_cicids_2018*.py`), dashboard “Predictions on CICIDS 2018” **metrics** (reads JSON only) |
| `cache/eval_cache_2018_multi_bal_v1/` | can be huge | **Optional** — faster multiclass 2018 eval; recreate with §3.5; safe to delete |
| `cache/eval_cache_2018_binary_bal_v1/` | can be huge | **Optional** — faster binary 2018 eval; recreate with §3.5; safe to delete |
| ~~`tmp_mc2018_parts/`~~ | — | **Removed** — temporary build shards only |

**Streamlit predictions** need: a `best_model.pt` run + the matching scaler (`scaler.joblib` or `scaler_binary.joblib`). They **do not** read `processed/cache/*` (graphs are precomputed to a **temporary** folder per run).

All rows above are **regenerable**; see **§0.2** for the exact command order after a full wipe.

---

## 1. Multiclass model (3-class / Model A)

### 1.1 Preprocess CICIDS 2017 (raw → `processed/`)

Requires raw CSVs under `TrafficLabelling/*.csv`.

```bash
python preprocess_cicids.py
```

Produces at least: `processed/traffic_all_processed.parquet`, `processed/scaler.joblib`.

Optional CSV export (large):

```bash
set PREPROCESS_WRITE_CSV=1
python preprocess_cicids.py
```

### 1.2 Precompute training windows + graphs (fast training)

```bash
python precompute_training_cache.py --cache_dir "processed/cache/train_cache_v1"
```

Overwrite an existing cache:

```bash
python precompute_training_cache.py --cache_dir "processed/cache/train_cache_v1" --force
```

### 1.3 Train

```bash
python run_train.py
```

Writes a new folder under `outputs/models/<timestamp>/` with `best_model.pt`, `config.json`, `history.json`, etc.

---

## 2. Binary model (BENIGN vs ATTACK)

### 2.1 Preprocess (binary parquet + scaler)

```bash
python preprocess_cicids_binary.py
```

Produces: `processed/traffic_all_binary_processed.parquet`, `processed/scaler_binary.joblib`.

### 2.2 Precompute binary training cache

```bash
python precompute_training_cache.py --cache_dir "processed/cache/train_cache_binary_v1" --data_filename "traffic_all_binary_processed.parquet" --scaler_filename "scaler_binary.joblib"
```

With `--force` to rebuild.

### 2.3 Train binary

```bash
python run_train_binary.py
```

---

## 3. CICIDS 2018 — prepare aligned data and eval sets

Requires **multiclass** `scaler.joblib` from §1.1. Raw 2018 CSVs under `CIC-IDS-2018-Dataset/*.csv`.

### 3.1 Preprocess full 2018 → train-feature space

```bash
python preprocess_cicids_2018.py
```

Writes e.g. `traffic_2018_all_classes.parquet`, `traffic_2018_train_classes.parquet`, and CSV copies.

### 3.2 Balanced eval tables (for `eval_cicids_2018*.py` defaults)

Adjust targets via env if needed (defaults are very large benign caps — see script).

```bash
python reprocess_cicids_2018_eval_balanced.py
```

Produces e.g. `traffic_2018_multiclass_eval_balanced.parquet`, `traffic_2018_binary_eval_balanced.parquet`.

### 3.3 Multiclass evaluation on 2018 (Model A)

Latest multiclass run, default balanced parquet:

```bash
python eval_cicids_2018.py
```

Explicit run and data:

```bash
python eval_cicids_2018.py --run 20260301_120000 --data processed/traffic_2018_multiclass_eval_balanced.parquet
```

With a precomputed eval cache (see §3.5):

```bash
python eval_cicids_2018.py --cache-dir processed/cache/eval_cache_2018_multi_bal_v1
```

Optional: `--max-windows 50000` for a quicker pass (works with or without `--cache-dir`).

Metrics JSON default: `outputs/models/<run>/cicids2018_eval.json`.

### 3.4 Binary evaluation on 2018

```bash
python eval_cicids_2018_binary.py
```

Or:

```bash
python eval_cicids_2018_binary.py --run my_run_binary --data processed/traffic_2018_binary_eval_balanced.parquet
```

With a precomputed eval cache (see §3.5):

```bash
python eval_cicids_2018_binary.py --cache-dir processed/cache/eval_cache_2018_binary_bal_v1
```

Writes `cicids2018_eval_binary.json` next to that run.

### 3.5 Optional: precompute CICIDS 2018 eval caches (faster `eval_cicids_2018*.py`)

These are **not** used by the Streamlit Predictions tab. They precompute the same window+graph memmaps as training cache, but built from the **balanced 2018 eval parquets** (§3.2). Expect large disk use and a long first pass.

Prerequisites: §3.1–3.2 done; a trained **`best_model.pt`** whose `label2idx` matches how you will run eval (pass that checkpoint below).

Replace `YOUR_MULTICLASS_RUN` / `YOUR_BINARY_RUN` with a folder name under `outputs/models/` (the run you evaluate).

**Multiclass (Model A) eval cache** → `processed/cache/eval_cache_2018_multi_bal_v1/`:

```bash
python precompute_training_cache.py --processed_dir processed --cache_dir processed/cache/eval_cache_2018_multi_bal_v1 --data_filename traffic_2018_multiclass_eval_balanced.parquet --scaler_filename scaler.joblib --checkpoint_path outputs/models/YOUR_MULTICLASS_RUN/best_model.pt --force
```

Then:

```bash
python eval_cicids_2018.py --cache-dir processed/cache/eval_cache_2018_multi_bal_v1
```

**Binary eval cache** → `processed/cache/eval_cache_2018_binary_bal_v1/`:

```bash
python precompute_training_cache.py --processed_dir processed --cache_dir processed/cache/eval_cache_2018_binary_bal_v1 --data_filename traffic_2018_binary_eval_balanced.parquet --scaler_filename scaler_binary.joblib --checkpoint_path outputs/models/YOUR_BINARY_RUN/best_model.pt --force
```

Then:

```bash
python eval_cicids_2018_binary.py --cache-dir processed/cache/eval_cache_2018_binary_bal_v1
```

**Notes**

- `--checkpoint_path` keeps `label2idx` / window labels aligned with that checkpoint (same as training cache precompute).
- Omit `--force` if the cache directory is empty and you want the script to skip when `cache_meta.json` already exists.
- On Windows PowerShell you can use the same one-line commands; adjust paths if your project is not the current directory.

---

## 4. Dashboard (preprocess + predict on uploads)

```bash
streamlit run dashboard.py
```

- **Predictions** tab: pick multi vs binary; the **newest** compatible run is pre-selected when models exist.
- Inference precomputes window+graph tensors to a **temp directory** (same format as training cache), then runs the model — no dependency on `processed/cache/train_cache_*` for uploads.

---

## 5. Demo raw CSV samples (optional)

```bash
python make_raw_demo_samples.py
```

Writes `sample_data/demo_cicids_2017_raw.csv` and `sample_data/demo_cicids_2018_raw.csv`.

---

## 6. Quick reference

Use **§0.2** for the authoritative ordered checklist after a **full** empty `processed/` (through eval). Short copy-paste:

```text
python preprocess_cicids.py
python preprocess_cicids_binary.py
python precompute_training_cache.py --cache_dir "processed/cache/train_cache_v1" --force
python precompute_training_cache.py --cache_dir "processed/cache/train_cache_binary_v1" --data_filename "traffic_all_binary_processed.parquet" --scaler_filename "scaler_binary.joblib" --force
python preprocess_cicids_2018.py
python reprocess_cicids_2018_eval_balanced.py
python run_train.py
python run_train_binary.py
python eval_cicids_2018.py
python eval_cicids_2018_binary.py
```

Then optionally **§3.5** for 2018 eval caches.
