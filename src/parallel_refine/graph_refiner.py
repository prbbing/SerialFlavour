"""Modular weighted-pair GNN and data adapters for FG0--FG2.

The graph encoder consumes only frozen Parallel predictions.  It has no access
to truth origin or truth-pair labels, keeping the downstream A/B/Y contract
identical to the tabular refiners.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.parallel_refine.cache import FrozenFeatureCache
from src.parallel_refine.graph_cache import GraphFeatureCache


def assert_cache_alignment(table: FrozenFeatureCache, graph: GraphFeatureCache):
    """Reject a graph/table pairing unless jet identity and labels agree."""
    if not np.array_equal(table.source_index, graph.source_index):
        raise ValueError("structured and graph cache source_index mismatch")
    if not np.array_equal(table.labels, graph.labels):
        raise ValueError("structured and graph cache labels mismatch")
    if not np.array_equal(table.event_number, graph.event_number):
        raise ValueError("structured and graph cache event_number mismatch")


def graph_node_values(graph: GraphFeatureCache, recipe: str, index: int):
    if recipe == "FG0":
        return graph.track_mask[index].astype(np.float32, copy=True)[..., None]
    if recipe == "FG1":
        return graph.origin_probs[index].astype(np.float32, copy=True)
    if recipe == "FG2":
        return graph.track_embedding[index].astype(np.float32, copy=True)
    raise ValueError(f"not a graph recipe: {recipe}")


def resolve_graph_config(graph_config, graph: GraphFeatureCache):
    """Resolve the symbolic graph output width against one frozen checkpoint."""
    resolved = copy.deepcopy(graph_config)
    if resolved["output_dim"] == "track_embedding_dim":
        resolved["output_dim"] = int(graph.track_embedding.shape[-1])
    return resolved


class CachedGraphDataset(Dataset):
    def __init__(self, table, graph, context_columns, recipe):
        assert_cache_alignment(table, graph)
        self.table = table
        self.graph = graph
        self.context_columns = np.asarray(context_columns, dtype=np.int64)
        self.recipe = recipe

    def __len__(self):
        return len(self.table.labels)

    def __getitem__(self, index):
        return {
            "context": torch.from_numpy(np.array(
                self.table.features[index, self.context_columns],
                dtype=np.float32, copy=True)),
            "pair_probs": torch.from_numpy(np.array(
                self.graph.pair_probs[index], dtype=np.float32, copy=True)),
            "track_mask": torch.from_numpy(np.array(
                self.graph.track_mask[index], dtype=np.bool_, copy=True)),
            "node_values": torch.from_numpy(graph_node_values(
                self.graph, self.recipe, index)),
            "y": torch.tensor(int(self.table.labels[index]), dtype=torch.long),
        }


def create_graph_loader(table, graph, context_columns, recipe, *, batch_size,
                        shuffle, num_workers, seed):
    from src.config import dataloader_generator, seed_dataloader_worker

    return DataLoader(
        CachedGraphDataset(table, graph, context_columns, recipe),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator(seed), worker_init_fn=seed_dataloader_worker)


class WeightedPairMessageLayer(nn.Module):
    """Directed, scalar-edge message passing without a B×K×K×H tensor."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values, adjacency, mask):
        # A[i,j] is the frozen probability that i and j are a matching pair.
        out_degree = adjacency.sum(-1, keepdim=True).clamp_min(1e-6)
        in_degree = adjacency.sum(-2, keepdim=True).clamp_min(1e-6)
        normalized = adjacency / torch.sqrt(out_degree * in_degree)
        outgoing = torch.bmm(normalized, values)
        incoming = torch.bmm(normalized.transpose(1, 2), values)
        update = self.update(torch.cat([values, outgoing, incoming], dim=-1))
        result = self.norm(values + update)
        return result * mask.unsqueeze(-1).to(result.dtype)


class PairGraphEncoder(nn.Module):
    """Encode FG0/FG1/FG2 node inputs and weighted directed pair topology."""

    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            WeightedPairMessageLayer(hidden_dim, dropout) for _ in range(num_layers))
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim), nn.ReLU(), nn.LayerNorm(output_dim))

    def forward(self, node_values, pair_probs, track_mask):
        mask = track_mask.bool()
        tracks = mask.shape[1]
        eye = torch.eye(tracks, dtype=torch.bool, device=mask.device).unsqueeze(0)
        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
        adjacency = pair_probs.float() * valid_pair.to(pair_probs.dtype)
        values = self.node_projection(node_values.float())
        values = values * mask.unsqueeze(-1).to(values.dtype)
        for layer in self.layers:
            values = layer(values, adjacency, mask)
        count = mask.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        mean = values.sum(1) / count
        maximum = values.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.readout(torch.cat([mean, maximum], dim=-1))


class GraphDNNRefiner(nn.Module):
    """Static pooled context plus one modular pair-graph encoder."""

    def __init__(self, context_dim, node_dim, graph_config, mean, std,
                 n_classes: int = 3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.encoder = PairGraphEncoder(
            node_dim, graph_config["hidden_dim"], graph_config["num_layers"],
            graph_config["output_dim"], graph_config["dropout"])
        layers = []
        current = context_dim + graph_config["output_dim"]
        for width in graph_config["hidden_dims"]:
            layers.extend([nn.Linear(current, int(width)), nn.ReLU(),
                           nn.Dropout(graph_config["dropout"])])
            current = int(width)
        layers.append(nn.Linear(current, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, context, node_values, pair_probs, track_mask):
        graph = self.encoder(node_values, pair_probs, track_mask)
        context = (context - self.mean) / self.std
        return self.classifier(torch.cat([context, graph], dim=-1))


def save_graph_description(path, *, recipe, context_columns, context_names,
                           node_dim, graph_config):
    payload = {
        "model_type": "graph_dnn_refiner",
        "recipe": recipe,
        "context_columns": np.asarray(context_columns, dtype=np.int64).tolist(),
        "context_feature_names": list(context_names),
        "node_dim": int(node_dim),
        "graph_config": graph_config,
        "n_classes": 3,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def load_graph_refiner(directory, device):
    directory = Path(directory)
    description = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    normalization = np.load(directory / "normalization.npz")
    model = GraphDNNRefiner(
        len(description["context_columns"]), description["node_dim"],
        description["graph_config"], normalization["mean"], normalization["std"],
        description["n_classes"]).to(device)
    model.load_state_dict(torch.load(
        directory / "best_graph_refiner.pt", map_location=device, weights_only=True))
    model.eval()
    return model, description
