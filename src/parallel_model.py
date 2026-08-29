"""
Parallel three-task transformer jet classifier (GN2-inspired).

Architecture
------------
Single shared TransformerEncoder → three parallel task heads:

  1. Jet-flavour classification  — attention-pooled jet summary → 3 classes
  2. Track-origin classification  — per-track embeddings → 8 classes
  3. Track-pair vertexing          — pairwise Bilinear → same-vertex score

All dimensions are powers of 2 for efficient GPU computation.

Forward flow:
    x (B,K,F) → init_net (2-layer MLP) → encoder → per-track embeddings
      │
      ├── pool_attn → weighted sum → pooled (B,D) → jet_head → jet_logits
      ├── origin_head → origin_logits (B,K,8)
      └── pair_head: Bilinear(emb_i, emb_j) → pair_logits (B,K,K)

Parameter budget: approximately 55 k for the default configuration.
"""
import torch
import torch.nn as nn


class ParallelOriginVertexJetTransformer(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_origin_classes, n_jet_classes,
                 gate_temp=0.1):
        super().__init__()
        self.n_origin_classes = n_origin_classes
        self.gate_temp        = gate_temp

        # -- per-track initialisation network (GN2-style MLP) --
        self.init_net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # -- single shared transformer encoder --
        _enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(_enc_layer, num_layers=n_layers)

        # -- global attention pooling --
        self.pool_attn = nn.Linear(d_model, 1)

        # -- three parallel task heads --
        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

        self.origin_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_origin_classes))

        # Bilinear computes  x1^T W x2 + b  for each track pair
        self.pair_head = nn.Bilinear(d_model, d_model, 1)

    def forward(self, x, mask):
        """Single shared-encoder forward with three parallel heads.

        Args:
            x:    (B, K, F)      — track features
            mask: (B, K)  bool   — track validity

        Returns:
            dict: jet_logits (B,3), origin_logits (B,K,8),
                  pair_logits (B,K,K)
        """
        B, K, _ = x.shape
        track_padding_mask = ~mask

        # -- shared encoder --
        h = self.init_net(x)                                    # (B, K, D)
        h = self.encoder(h, src_key_padding_mask=track_padding_mask)  # (B, K, D)

        # -- global attention pooling --
        scores = self.pool_attn(h)                              # (B, K, 1)
        scores = scores.masked_fill(track_padding_mask.unsqueeze(-1), float("-inf"))
        attn_w = torch.softmax(scores, dim=1)                   # (B, K, 1)
        pooled = (h * attn_w).sum(dim=1)                        # (B, D)

        # -- parallel task heads --
        jet_logits    = self.jet_head(pooled)                   # (B, 3)
        origin_logits = self.origin_head(h)                     # (B, K, 8)

        # pairwise compatibility: for each pair (i,j), Bilinear(emb_i, emb_j)
        D = h.shape[-1]
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)             # (B, K, K, D)
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)             # (B, K, K, D)
        pair_logits = self.pair_head(
            h_i.reshape(B * K * K, D),
            h_j.reshape(B * K * K, D),
        ).reshape(B, K, K)                                      # (B, K, K)

        return {
            "jet_logits":    jet_logits,
            "origin_logits": origin_logits,
            "pair_logits":   pair_logits,
        }


def build_parallel_origin_vertex_jet(config):
    """Factory: Config → ParallelOriginVertexJetTransformer."""
    return ParallelOriginVertexJetTransformer(
        in_dim=config.n_feats,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        n_origin_classes=config.n_origin_classes,
        n_jet_classes=config.n_jet_classes,
        gate_temp=config.gate_temp,
    )
