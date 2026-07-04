"""Minimal training run for debugging - 5k samples, 2 epochs."""
import sys
print("Starting quick train...", flush=True)

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data, build_label_map, HybridIDSDataset, collate_hybrid
from src.model import HybridIDSModel
from src.config import PROCESSED_DIR, MODELS_DIR, BATCH_SIZE, SEQ_LEN

print("Loading data...", flush=True)
df, feat_cols, _ = load_processed_data(PROCESSED_DIR)
df = df.sample(5000, random_state=42).reset_index(drop=True)
print(f"Data shape: {df.shape}", flush=True)

label2idx, _ = build_label_map(df["Label"])
num_classes = len(label2idx)
print(f"Classes: {num_classes}", flush=True)

print("Creating dataset...", flush=True)
dataset = HybridIDSDataset(df, feat_cols, label2idx, seq_len=SEQ_LEN, max_graph_flows=64)
n = len(dataset)
print(f"Dataset len: {n}", flush=True)
train_ds, val_ds = random_split(dataset, [int(0.8 * n), n - int(0.8 * n)])

loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_hybrid, num_workers=0)
print(f"Batches per epoch: {len(loader)}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

model = HybridIDSModel(num_features=len(feat_cols), num_classes=num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

print("Training 2 epochs...", flush=True)
for epoch in range(2):
    model.train()
    for i, batch in enumerate(loader):
        seq = batch["seq"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_vec = batch["batch"].to(device)
        labels = batch["label"].to(device)
        logits = model(seq, edge_index, edge_attr, batch_vec, batch["batch_size"])
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} batch {i+1}/{len(loader)} loss={loss.item():.4f}", flush=True)
    print(f"Epoch {epoch+1} done", flush=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODELS_DIR / "quick_test.pt")
print("Saved quick_test.pt", flush=True)
