"""Frozen native head plus a zero-initialized permutation-invariant graph delta."""

from copy import deepcopy

import torch
from torch import nn


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class NativeResidualGraphReadout(nn.Module):
    """All recipes use the same trainable modules and full H/g/geometry access.

    Native receives standardized g through a compensated copy of its original
    head. Output is B-normalized gap, not eV. Only delta is optimized.
    Geometry enters exclusively through distances, making translation/rotation
    irrelevant. Symmetric pair features and bidirectional sum aggregation make
    the readout invariant to a consistent atom/pair relabelling.
    """

    def __init__(self, source_head, metadata, settings):
        super().__init__()
        self.native = deepcopy(source_head).cpu()
        for name in ("g_mean", "g_std", "h_mean", "h_std", "target_mean", "target_std"):
            self.register_buffer(name, torch.as_tensor(metadata[name], dtype=torch.float32).reshape(1, -1))
        with torch.no_grad():
            linear = [layer for layer in self.native if isinstance(layer, nn.Linear)]
            first, last = linear[0], linear[-1]
            weight = first.weight.clone()
            first.weight.mul_(self.g_std)
            first.bias.add_((weight @ self.g_mean[0]))
            scale = float(metadata["source_target_std"][0]) / float(metadata["target_std"][0])
            shift = (float(metadata["source_target_mean"][0]) - float(metadata["target_mean"][0])) / float(metadata["target_std"][0])
            last.weight.mul_(scale)
            last.bias.mul_(scale).add_(shift)
        self.native.requires_grad_(False).eval()
        width, atom_dim = int(settings["width"]), len(metadata["h_mean"])
        self.atom_type = nn.Embedding(10, int(settings["atom_type_width"]), padding_idx=0)
        self.register_buffer("rbf_centers", torch.linspace(0, settings["distance_max"], settings["rbf_count"]))
        self.rbf_width = settings["distance_max"] / (settings["rbf_count"] - 1)
        qhidden = int(metadata["charge_hidden_width"])
        bhidden = int(metadata["bond_hidden_width"])
        self.node = mlp(atom_dim + settings["atom_type_width"] + 1 + qhidden, width, width)
        edge_width = settings["rbf_count"] + 5 + bhidden
        self.messages = nn.ModuleList([mlp(2 * width + edge_width, width, width)
                                       for _ in range(settings["layers"])])
        self.updates = nn.ModuleList([mlp(2 * width, width, width) for _ in range(settings["layers"])])
        self.output = mlp(width + len(metadata["g_mean"]), width, 1)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()
        return self

    def forward(self, batch):
        g = (batch["embedding"] - self.g_mean) / self.g_std
        h = (batch["atom_embedding"] - self.h_mean) / self.h_std
        nodes = self.node(torch.cat([h, self.atom_type(batch["atom_z"]), batch["charge"], batch["charge_hidden"]], dim=-1))
        left, right = batch["pair_index"]
        distances = (batch["pos"][left] - batch["pos"][right]).square().sum(-1).sqrt()
        radial = torch.exp(-0.5 * ((distances[:, None] - self.rbf_centers) / self.rbf_width).square())
        edge = torch.cat([radial, batch["bond_probs"], batch["bond_hidden"]], dim=-1)
        for message, update in zip(self.messages, self.updates):
            pair = torch.cat([nodes[left] + nodes[right], (nodes[left] - nodes[right]).abs(), edge], dim=-1)
            values = message(pair)
            aggregate = torch.zeros_like(nodes).index_add_(0, left, values).index_add_(0, right, values)
            nodes = nodes + update(torch.cat([nodes, aggregate], dim=-1))
        pooled = nodes.new_zeros((len(g), nodes.shape[-1])).index_add_(0, batch["batch"], nodes)
        return self.native(g) + self.output(torch.cat([g, pooled], dim=-1))

    def physical(self, batch):
        return self(batch) * self.target_std + self.target_mean
