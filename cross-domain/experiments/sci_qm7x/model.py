"""Project SchNet-style distance model; not an author QM7-X MTL reproduction.

Independent package-local implementation of continuous-filter interactions
inspired by Schuett et al., NeurIPS 2017, https://arxiv.org/abs/1706.08566.
No third-party source or pretrained weights are imported. Scalar distance
features give translation/rotation invariance and permutation equivariance.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ShiftedSoftplus(nn.Module):
    def forward(self, value):
        return F.softplus(value) - math.log(2)


class Interaction(nn.Module):
    def __init__(self, width, radial_width):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(radial_width, width), ShiftedSoftplus(), nn.Linear(width, width))
        self.source = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(nn.Linear(width, width), ShiftedSoftplus(), nn.Linear(width, width))

    def forward(self, h, radial, envelope, mask):
        # Dense small-molecule graph (N <= 23), no scatter/CUDA atomic sums.
        messages = self.source(h)[:, None, :, :] * self.filter(radial) * envelope[..., None]
        return (h + self.update(messages.sum(2))) * mask[..., None]


class MolecularModel(nn.Module):
    def __init__(self, settings, variant='multi_task'):
        super().__init__()
        if variant not in ('single_task', 'multi_task'):
            raise ValueError('unknown variant')
        self.variant = variant
        width, hidden = settings['width'], settings['head_hidden']
        self.cutoff = float(settings['cutoff'])
        self.embedding = nn.Embedding(18, width, padding_idx=0)  # includes S=16 and Cl=17
        self.register_buffer('centers', torch.linspace(0, self.cutoff, settings['num_gaussians']))
        self.radial_width = self.cutoff / (settings['num_gaussians'] - 1)
        self.interactions = nn.ModuleList([Interaction(width, settings['num_gaussians']) for _ in range(settings['num_interactions'])])
        self.main_head = nn.Sequential(nn.Linear(width, hidden), ShiftedSoftplus(), nn.Linear(hidden, 1))
        # Both variants have exactly the same architecture and initialization order.
        # ST does not optimize these dormant local heads or use auxiliary labels.
        self.aux_hidden = nn.Sequential(nn.Linear(2 * width, hidden), ShiftedSoftplus())
        self.aux_output = nn.Linear(hidden, 2)

    def forward(self, batch):
        mask = batch['mask']
        if (~mask.any(1)).any():
            raise ValueError('empty molecule')
        distances = (batch['coords'][:, :, None] - batch['coords'][:, None, :]).square().sum(-1).sqrt()
        edges = mask[:, :, None] & mask[:, None, :]
        edges = edges & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        envelope = 0.5 * (torch.cos(math.pi * (distances / self.cutoff).clamp(max=1)) + 1)
        envelope = envelope * edges * (distances < self.cutoff)
        radial = torch.exp(-0.5 * ((distances[..., None] - self.centers) / self.radial_width).square())
        h = self.embedding(batch['z']) * mask[..., None]
        for interaction in self.interactions:
            h = interaction(h, radial, envelope, mask)
        g = h.sum(1) / mask.sum(1, keepdim=True)
        conditioned = torch.cat([h, g[:, None].expand_as(h)], -1)
        hidden = self.aux_hidden(conditioned) * mask[..., None]
        return {'H': h, 'g': g, 'main': self.main_head(g)[:, 0],
                'auxiliary': self.aux_output(hidden) * mask[..., None], 'aux_hidden': hidden}


class Readout(nn.Module):
    """All recipes: full-H learned attention/mean/max pooling and same geometry.

    Identical input widths/parameters; only the declared extra values differ.
    A zero output layer reproduces cached native HOMO exactly at epoch zero.
    No upstream module is attached, and there is no BatchNorm/dropout state.
    """
    def __init__(self, settings, refinement, target_std):
        super().__init__()
        width, extra = settings['width'], settings['head_hidden']
        hidden = refinement['hidden']
        self.cutoff = float(settings['cutoff'])
        self.register_buffer('geom_centers', torch.linspace(0, self.cutoff, refinement['geometry_bins']))
        self.geom_width = self.cutoff / max(1, refinement['geometry_bins'] - 1)
        self.node = nn.Sequential(nn.Linear(width + extra + refinement['geometry_bins'], hidden[0]), nn.SiLU(),
                                  nn.Linear(hidden[0], hidden[0]), nn.SiLU())
        self.attention = nn.Linear(hidden[0], 1)
        layers, previous = [], 3 * hidden[0] + width + 2
        for output in hidden:
            layers.extend([nn.Linear(previous, output), nn.SiLU()])
            previous = output
        self.regression = nn.Sequential(*layers, nn.Linear(previous, 1))
        nn.init.zeros_(self.regression[-1].weight)
        nn.init.zeros_(self.regression[-1].bias)
        self.register_buffer('target_std', torch.tensor(float(target_std)))

    def forward(self, batch):
        mask = batch['mask']
        distance = (batch['coords'][:, :, None] - batch['coords'][:, None, :]).square().sum(-1).sqrt()
        edges = mask[:, :, None] & mask[:, None, :]
        edges = edges & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        radial = torch.exp(-0.5 * ((distance[..., None] - self.geom_centers) / self.geom_width).square())
        geometry = (radial * edges[..., None]).sum(2) / edges.sum(2, keepdim=True).clamp_min(1)
        h = self.node(torch.cat([batch['H'], batch['extra'], geometry], -1))
        weights = self.attention(h).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
        attended = (h * weights[..., None]).sum(1)
        average = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = h.masked_fill(~mask[..., None], -torch.inf).amax(1)
        inputs = torch.cat([attended, average, maximum, batch['g'], batch['native_scaled'][:, None],
                            mask.sum(1, keepdim=True).to(h.dtype) / 23], -1)
        return batch['native'] + self.regression(inputs)[:, 0] * self.target_std
