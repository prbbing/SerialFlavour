"""Tiny SchNet encoder with property and local (per-atom/per-pair) heads.

Architecture reference: Schuett et al., NeurIPS 2017, arXiv:1706.08566.
The property regression heads and the local auxiliary heads are our adaptation,
not a reproduction of the paper's property-specific physical readouts. No
pretrained weights are used.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ShiftedSoftplus(nn.Module):
    def forward(self, value):
        return F.softplus(value) - math.log(2)


class Interaction(nn.Module):
    def __init__(self, hidden, gaussians):
        super().__init__()
        self.filters = nn.Sequential(nn.Linear(gaussians, hidden), ShiftedSoftplus(), nn.Linear(hidden, hidden))
        self.lin1 = nn.Linear(hidden, hidden, bias=False)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin = nn.Linear(hidden, hidden)
        self.activation = ShiftedSoftplus()

    def forward(self, h, edge_index, radial, envelope):
        source, destination = edge_index
        messages = self.lin1(h)[source] * self.filters(radial) * envelope[:, None]
        aggregate = torch.zeros_like(h).index_add_(0, destination, messages)
        return h + self.lin(self.activation(self.lin2(aggregate)))


def make_head(inputs, hidden, outputs):
    widths = [inputs, *hidden, outputs]
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(fan_in, fan_out))
        if index < len(widths) - 2:
            layers.append(ShiftedSoftplus())
    return nn.Sequential(*layers)


class TinySchNet(nn.Module):
    def __init__(self, settings, property_tasks, local_tasks=()):
        super().__init__()
        self.property_tasks = list(property_tasks)
        self.local_tasks = list(local_tasks)
        self.bond_classes = int(settings.get("bond_classes", 5))
        hidden = settings["hidden_channels"]
        self.cutoff = settings["cutoff"]
        self.embedding = nn.Embedding(10, hidden, padding_idx=0)
        self.register_buffer("centers", torch.linspace(0, self.cutoff, settings["num_gaussians"]))
        self.width = self.cutoff / (settings["num_gaussians"] - 1)
        self.interactions = nn.ModuleList([Interaction(hidden, settings["num_gaussians"]) for _ in range(settings["num_interactions"])])
        head_hidden = settings["head_hidden"]
        self.heads = nn.ModuleDict()
        for task in self.property_tasks:
            self.heads[task] = make_head(hidden, head_hidden, 1)
        self.charge_head = make_head(2 * hidden, head_hidden, 1) if "charge" in self.local_tasks else None
        self.bond_head = make_head(3 * hidden, head_hidden, self.bond_classes) if "bond" in self.local_tasks else None

    def forward(self, batch):
        distance = batch["distance"]
        radial = torch.exp(-0.5 * ((distance[:, None] - self.centers[None]) / self.width).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1)
        h = self.embedding(batch["z"])
        for interaction in self.interactions:
            h = interaction(h, batch["edge_index"], radial, envelope)
        batch_index = batch["batch"]
        count = int(batch_index.max()) + 1
        g = h.new_zeros((count, h.shape[1])).index_add_(0, batch_index, h)
        if self.property_tasks:
            prediction = torch.cat([self.heads[task](g) for task in self.property_tasks], dim=1)
        else:
            prediction = h.new_zeros((count, 0))
        output = {"embedding": g, "atomic_embedding": h, "prediction": prediction,
                  "main_prediction": prediction[:, :1], "aux_predictions": prediction[:, 1:]}
        if self.charge_head is not None:
            conditioned = torch.cat([h, g[batch_index]], dim=-1)
            output["charge_prediction"] = self.charge_head(conditioned)[:, 0]
        if self.bond_head is not None:
            pair_index = batch["pair_index"]
            left, right = h[pair_index[0]], h[pair_index[1]]
            # Unordered chemical bonds must not depend on endpoint/atom ordering.
            pair_inputs = torch.cat([left + right, (left - right).abs(), g[batch["pair_batch"]]], dim=-1)
            output["bond_logits"] = self.bond_head(pair_inputs)
        return output
