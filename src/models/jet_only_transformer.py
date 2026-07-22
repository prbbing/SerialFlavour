"""Unified jet-only Transformer used for backbone-controlled ablations."""

import torch
import torch.nn as nn


class JetOnlyTransformer(nn.Module):
    """Jet classifier with configurable track initialisation and pooling.

    The ``mlp + attention`` variant deliberately uses the same module names,
    operations and ordering as the jet path in
    :class:`ParallelOriginVertexJetTransformer`. This allows direct state-dict
    transfer and an exact-logit equivalence test against Parallel-jet-only.
    """

    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_jet_classes, track_init="mlp", pooling="attention"):
        super().__init__()
        self.track_init = track_init
        self.pooling = pooling

        if track_init == "mlp":
            self.init_net = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        elif track_init == "linear":
            self.init_net = nn.Linear(in_dim, d_model)
        else:
            raise ValueError("track_init must be 'mlp' or 'linear'")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        if pooling == "attention":
            self.pool_attn = nn.Linear(d_model, 1)
        elif pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            raise ValueError("pooling must be 'attention' or 'cls'")

        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

    def forward(self, x, mask):
        track_padding_mask = ~mask
        tracks = self.init_net(x)

        if self.pooling == "attention":
            hidden = self.encoder(
                tracks, src_key_padding_mask=track_padding_mask)
            scores = self.pool_attn(hidden)
            scores = scores.masked_fill(
                track_padding_mask.unsqueeze(-1), float("-inf"))
            attention = torch.softmax(scores, dim=1)
            pooled = (hidden * attention).sum(dim=1)
        else:
            batch_size = x.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls_token, tracks], dim=1)
            cls_valid = torch.ones(
                batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = ~torch.cat([cls_valid, mask], dim=1)
            hidden = self.encoder(
                sequence, src_key_padding_mask=padding_mask)
            pooled = hidden[:, 0]

        return {"jet_logits": self.jet_head(pooled)}


def build_jet_only_transformer(config):
    return JetOnlyTransformer(
        in_dim=config.n_feats,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        n_jet_classes=config.n_jet_classes,
        track_init=config.jet_only_track_init,
        pooling=config.jet_only_pooling,
    )
