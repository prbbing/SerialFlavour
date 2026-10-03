"""A reusable capacity-matched tabular readout and a set/graph readout."""

import torch
from torch import nn


def _mlp(inputs, hidden, outputs, dropout=0.0, activation=nn.SiLU):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(inputs, width), activation()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class TabularReadout(nn.Module):
    def __init__(self, inputs, hidden, outputs=1, dropout=0.0):
        super().__init__()
        self.network = _mlp(inputs, hidden, outputs, dropout=dropout)

    def forward(self, batch):
        return self.network(batch["features"])


class GraphSetReadout(nn.Module):
    """Permutation-invariant readout over per-atom features and an optional predicted graph."""

    def __init__(self, node_dim, hidden, decoder_hidden, layers=2, use_edges=False, use_atom_type=False,
                 atom_types=10, dropout=0.0):
        super().__init__()
        self.use_edges = use_edges
        self.use_atom_type = use_atom_type
        in_dim = node_dim
        if use_atom_type:
            self.atom_embedding = nn.Embedding(atom_types, hidden, padding_idx=0)
            in_dim += hidden
        self.node = _mlp(in_dim, [hidden], hidden, dropout=dropout)
        if use_edges:
            self.messages = nn.ModuleList([_mlp(2 * hidden, [hidden], hidden, dropout=dropout) for _ in range(layers)])
        self.decoder = _mlp(hidden, decoder_hidden, 1, dropout=dropout)

    def forward(self, batch):
        features = batch["features"]
        if self.use_atom_type:
            features = torch.cat([features, self.atom_embedding(batch["z"])], dim=-1)
        h = self.node(features)
        if self.use_edges:
            source, destination = batch["pair_index"]
            weight = 1.0 - batch["bond_probs"][:, 0]
            reverse_source, reverse_destination = destination, source
            source = torch.cat([source, reverse_source])
            destination = torch.cat([destination, reverse_destination])
            weight = torch.cat([weight, weight])
            for message in self.messages:
                numerator = h.new_zeros(h.shape).index_add_(0, destination, h[source] * weight[:, None])
                denominator = h.new_zeros((h.shape[0], 1)).index_add_(0, destination, weight[:, None])
                aggregate = numerator / (denominator + 1e-6)
                h = h + message(torch.cat([h, aggregate], dim=-1))
        count = batch["target"].shape[0]
        pooled = h.new_zeros((count, h.shape[1])).index_add_(0, batch["batch"], h)
        return self.decoder(pooled)


def fit_standardization(values):
    # Accumulate in float64; zero/constant feature slots keep scale one.
    values = values.astype("float64")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype("float32"), std.astype("float32")
