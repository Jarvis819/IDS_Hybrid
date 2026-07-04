# A Deep Learning Framework Combining Temporal and Structural Network Behavior Modeling with Rapid Adaptation for Zero-Day Attack Detection

**M.Tech Thesis Presentation**

---

## 1. Problem Statement

- **Zero-day attacks**: Novel threats with no prior signatures
- **Limitations of traditional IDS**: Rule-based and signature-based systems fail on unseen attacks
- **Need**: Models that generalize to new attack types with minimal labeled examples

---

## 2. Proposed Approach

**Hybrid architecture** combining:

| Component | Role |
|-----------|------|
| **Transformer** | Captures temporal patterns in flow sequences |
| **GNN** | Models structural (graph) behavior of host communication |
| **Few-shot learning** | Rapid adaptation to new attack types with few examples |

---

## 3. Architecture Overview

```
Flow Features (80 dims)
        │
        ├──► Flow Encoder (MLP)
        │         │
        │         ▼
        │    Temporal Encoder (Transformer)
        │         │
        │         ▼
        │    [Temporal Embedding]
        │
        └──► Graph (IPs → nodes, flows → edges)
                  │
                  ▼
             Edge GNN
                  │
                  ▼
             [Structural Embedding]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Fusion Layer        Prototypical Head
        │                   (few-shot)
        ▼
   Classifier → Attack Type
```

---

## 4. Dataset & Preprocessing

- **Source**: CICIDS-style TrafficLabelling (8 CSV files)
- **Preprocessing**:
  - Dropped `Flow ID` (identifier; harms generalization)
  - Handled `Infinity` / missing values
  - StandardScaler for numeric features
- **Features**: 80 numeric flow statistics (bytes/s, IAT, packet lengths, flags, etc.)
- **Labels**: BENIGN + multiple attack types (DoS, DDoS, PortScan, WebAttacks, Infiltration, etc.)

---

## 5. Experimental Setup

- **Train/Val split**: 80/20
- **Temporal window**: 32 consecutive flows
- **Graph**: IPs as nodes, flows as edges
- **Evaluation**: Accuracy, F1; zero-day evaluation on synthetic dataset

---

## 6. Results (Placeholder)

- Training curves: loss and accuracy over epochs
- Confusion matrix on validation set
- Zero-day detection: performance on synthetic `ZERO_DAY` class (unseen during training)

---

## 7. Custom Synthetic Dataset

- **Purpose**: Prove generalization to unseen attack patterns
- **Generation**: Sample from empirical distributions + inject controlled anomalous pattern
- **Zero-day class**: `ZERO_DAY` with boosted Flow Bytes/s, Flow Packets/s
- **Usage**: `python generate_synthetic_dataset.py`

---

## 8. Dashboard

- **Streamlit app**: `streamlit run dashboard.py`
- **Tabs**: Data Overview, Training Curves, Predictions, Preprocessing Summary

---

## 9. Conclusion

- Hybrid temporal + structural modeling improves representation
- Few-shot capability enables rapid adaptation
- Synthetic dataset validates zero-day detection claims

---

## 10. Future Work

- Larger-scale experiments on full CICIDS
- Prototypical few-shot episodic training
- Real-time deployment and latency analysis
