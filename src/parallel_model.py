"""GN2-inspired Parallel transformer with three configurable task heads.

Jet features are copied to every track and concatenated with the track inputs
before a shared Transformer encoder. The resulting track representation may
optionally be projected before attention pooling and the three task-specific
MLPs. Track-origin and track-pair tasks are conditioned on the pooled jet
representation.
"""

import torch
import torch.nn as nn


def _make_task_head(in_dim, hidden_dims, out_dim):
    layers = []
    current = in_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), nn.ReLU()))
        current = width
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class ParallelOriginVertexJetTransformer(nn.Module):
    def __init__(
            self, track_in_dim, jet_in_dim, d_model, n_heads, n_layers, d_ffn,
            dropout, n_origin_classes, n_jet_classes,
            track_projection_dim=None, task_head_hidden_dims=(16,)):
        super().__init__()
        self.n_origin_classes = n_origin_classes

        self.init_net = nn.Sequential(
            nn.Linear(track_in_dim + jet_in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        task_dim = track_projection_dim or d_model
        self.track_projection = (
            nn.Identity() if track_projection_dim is None
            else nn.Linear(d_model, track_projection_dim))
        self.task_dim = task_dim
        self.pool_attn = nn.Linear(task_dim, 1)

        hidden_dims = tuple(task_head_hidden_dims)
        self.jet_head = _make_task_head(
            task_dim, hidden_dims, n_jet_classes)
        self.origin_head = _make_task_head(
            2 * task_dim, hidden_dims, n_origin_classes)
        self.pair_head = _make_task_head(
            3 * task_dim, hidden_dims, 1)

    def representations(self, track_features, jet_features, mask):
        """Return track embeddings, pooled jet embedding, and pool weights."""
        padding = ~mask
        repeated_jet = jet_features.unsqueeze(1).expand(
            -1, track_features.shape[1], -1)
        combined = torch.cat((track_features, repeated_jet), dim=-1)
        encoded = self.encoder(
            self.init_net(combined), src_key_padding_mask=padding)
        track_embedding = self.track_projection(encoded)
        scores = self.pool_attn(track_embedding).masked_fill(
            padding.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(scores, dim=1)
        pooled = (track_embedding * attention).sum(dim=1)
        return track_embedding, pooled, attention

    def task_logits(self, track_embedding, pooled):
        """Apply structurally identical MLP heads to task-specific inputs."""
        batch, tracks, dimension = track_embedding.shape
        pooled_tracks = pooled.unsqueeze(1).expand(-1, tracks, -1)
        origin_logits = self.origin_head(torch.cat(
            (track_embedding, pooled_tracks), dim=-1))

        embedding_i = track_embedding.unsqueeze(2).expand(
            -1, -1, tracks, -1)
        embedding_j = track_embedding.unsqueeze(1).expand(
            -1, tracks, -1, -1)
        pooled_pairs = pooled[:, None, None, :].expand(
            -1, tracks, tracks, -1)
        pair_inputs = torch.cat(
            (embedding_i, embedding_j, pooled_pairs), dim=-1)
        pair_logits = self.pair_head(
            pair_inputs.reshape(batch * tracks * tracks, 3 * dimension)
        ).reshape(batch, tracks, tracks)

        return {
            "jet_logits": self.jet_head(pooled),
            "origin_logits": origin_logits,
            "pair_logits": pair_logits,
        }

    def forward(self, track_features, jet_features, mask):
        """Run the shared encoder and the three Parallel task heads."""
        track_embedding, pooled, _ = self.representations(
            track_features, jet_features, mask)
        return self.task_logits(track_embedding, pooled)


def build_parallel_origin_vertex_jet(config):
    """Build a Parallel model from a resolved runtime configuration."""
    return ParallelOriginVertexJetTransformer(
        track_in_dim=config.n_feats,
        jet_in_dim=config.n_jet_feats,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        n_origin_classes=config.n_origin_classes,
        n_jet_classes=config.n_jet_classes,
        track_projection_dim=config.track_projection_dim,
        task_head_hidden_dims=config.task_head_hidden_dims,
    )
