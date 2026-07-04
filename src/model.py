"""Hybrid IDS model: Transformer (temporal) + GNN (structural) + few-shot head."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowEncoder(nn.Module):
    """MLP encoder for per-flow features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):
    """Transformer encoder for flow sequences."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, S, D]
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(x.size(0), -1)
        pos_emb = self._positional_encoding(pos)
        x = x + pos_emb
        out = self.transformer(x, src_key_padding_mask=mask)
        return out.mean(dim=1)  # [B, D] global pooling


    def _positional_encoding(self, pos: torch.Tensor) -> torch.Tensor:
        pe = torch.zeros(pos.size(0), pos.size(1), self.d_model, device=pos.device)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=pos.device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(pos.unsqueeze(-1) * div)
        pe[:, :, 1::2] = torch.cos(pos.unsqueeze(-1) * div)
        return pe


class EdgeGNN(nn.Module):
    """Simple GNN over edges: aggregate edge features to nodes, then to graph."""

    def __init__(self, edge_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=edge_attr.device)

        row, col = edge_index[0], edge_index[1]
        x = self.edge_proj(edge_attr)  # [E, H]

        num_nodes = batch.size(0)
        out = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        cnt = torch.zeros(num_nodes, 1, device=x.device)
        out.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(-1)), x)
        cnt.scatter_add_(0, col.unsqueeze(-1), torch.ones_like(col, dtype=torch.float32).unsqueeze(-1))
        cnt = cnt.clamp(min=1)
        out = out / cnt

        for lin in self.layers:
            out_src = out[row]
            out_dst = out[col]
            msg = F.gelu(lin(torch.cat([out_src, out_dst], dim=-1)))
            out_new = torch.zeros_like(out)
            out_new.scatter_add_(0, col.unsqueeze(-1).expand(-1, msg.size(-1)), msg)
            cnt_msg = torch.zeros(num_nodes, 1, device=x.device)
            cnt_msg.scatter_add_(0, col.unsqueeze(-1), torch.ones_like(col, dtype=torch.float32).unsqueeze(-1))
            cnt_msg = cnt_msg.clamp(min=1)
            out = out + out_new / cnt_msg
            out = F.dropout(out, p=self.dropout, training=self.training)

        # Graph-level: scatter mean by batch
        graph = torch.zeros(batch_size, self.hidden_dim, device=out.device)
        cnt_g = torch.zeros(batch_size, 1, device=out.device)
        graph.scatter_add_(0, batch.unsqueeze(-1).expand(-1, out.size(-1)), out)
        cnt_g.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch, dtype=torch.float32).unsqueeze(-1))
        graph = graph / cnt_g.clamp(min=1)
        return graph


class HybridIDSModel(nn.Module):
    """Combines temporal (Transformer) and structural (GNN) encodings for classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.flow_enc = FlowEncoder(num_features, hidden_dim, transformer_dim, dropout)
        self.temporal_enc = TemporalEncoder(
            transformer_dim, transformer_heads, transformer_layers, dropout=dropout
        )
        self.gnn = EdgeGNN(num_features, gnn_hidden, gnn_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(transformer_dim + gnn_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # Temporal branch
        B, S, F = seq.shape
        seq_emb = self.flow_enc(seq)  # [B, S, D]
        temporal_out = self.temporal_enc(seq_emb)  # [B, D]

        # Structural branch
        graph_out = self.gnn(edge_index, edge_attr, batch, batch_size)  # [B, G]

        # Fusion
        fused = torch.cat([temporal_out, graph_out], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits


class PrototypicalHead(nn.Module):
    """Few-shot prototypical classifier: compute class prototypes from support, classify query."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

    def forward(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
        query_emb: torch.Tensor,
    ) -> torch.Tensor:
        # support_emb: [N_s, D], support_labels: [N_s]
        # query_emb: [N_q, D]
        prototypes = []
        for c in range(self.num_classes):
            mask = support_labels == c
            if mask.any():
                prototypes.append(support_emb[mask].mean(dim=0))
            else:
                prototypes.append(support_emb.mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)  # [C, D]
        dists = torch.cdist(query_emb, prototypes)
        logits = -dists
        return logits
