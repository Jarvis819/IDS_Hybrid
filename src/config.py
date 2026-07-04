"""Configuration for hybrid IDS model and training."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# Model
NUM_FEATURES = 80
HIDDEN_DIM = 128
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
GNN_LAYERS = 2
GNN_HIDDEN = 64
DROPOUT = 0.2

# Training
# Batch size tuned for stability and accuracy
BATCH_SIZE = 256
SEQ_LEN = 32  # flows per temporal window
GRAPH_SIZE = 64  # max nodes per graph
# Stride between consecutive temporal windows (1 = fully overlapping)
WINDOW_STRIDE = 1
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Few-shot
N_WAY = 5
K_SHOT = 5
N_QUERY = 15
