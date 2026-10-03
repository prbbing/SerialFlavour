"""Width/layer-reduced author CrystalTransformer, exposing frozen H/g."""
import torch
from torch import nn
import torch.nn.functional as F

REFERENCE_COMMIT = '0141ff9e09277d2229c9d7a24c1bcc5eac9de78e'


class CrystalTransformer(nn.Module):
    def __init__(self, settings, variant='multi_task'):
        super().__init__()
        width = settings['feature_size']
        if variant not in ('single_task', 'mt_main_only', 'multi_task'):
            raise ValueError('unsupported upstream variant')
        self.variant = variant
        self.coords_embed = nn.Linear(3, width // 2)
        self.atom_embed = nn.Linear(100, width // 2)
        layer = nn.TransformerEncoderLayer(width, settings['num_heads'], settings['dim_feedforward'],
                                           settings['dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, settings['num_layers'], enable_nested_tensor=False)
        if variant == 'single_task':
            # Published ST implementation has two linear layers with no activation.
            self.output_linear1 = nn.Linear(width, settings['st_head_hidden'])
            self.output_linear2 = nn.Linear(settings['st_head_hidden'], 1)
        else:
            self.output_linear1 = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            self.output_linear2 = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, z, coords, mask):
        # Z=0 is padding; match author's 100-dimensional Z-1 one-hot for real sites.
        atom = F.one_hot((z - 1).clamp_min(0), 100).to(coords.dtype)
        atom = atom.masked_fill(mask.unsqueeze(-1), 0)
        src = torch.cat([self.atom_embed(atom), self.coords_embed(coords)], -1)
        h = self.transformer_encoder(src, src_key_padding_mask=mask)
        g = h[:, 0]
        if self.variant == 'single_task':
            return {'H': h, 'g': g, 'main': self.output_linear2(self.output_linear1(g)).squeeze(-1)}
        # Author MT head1=formation energy; head2=bandgap.
        aux_hidden = self.output_linear1[:2](g)
        return {'H': h, 'g': g, 'main': self.output_linear2(g).squeeze(-1),
                'auxiliary': self.output_linear1(g).squeeze(-1), 'aux_hidden': aux_hidden}


class Readout(nn.Module):
    """Same full H access and parameter budget for all predicted-aux comparisons."""
    def __init__(self, width, hidden, target_mean, target_std):
        super().__init__()
        if not hidden or any(type(value) is not int or value <= 0 for value in hidden):
            raise ValueError('readout hidden widths must be positive integers')
        self.atom_projection = nn.Sequential(nn.Linear(width, hidden[0]), nn.ReLU())
        layers = []
        input_width = width + hidden[0] + 2
        for output_width in hidden:
            layers.extend([nn.Linear(input_width, output_width), nn.ReLU()])
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.head = nn.Sequential(*layers)
        # Residual starts exactly at native; shared fitter selects epoch 0 on B_val too.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('target_mean', torch.tensor(float(target_mean)))
        self.register_buffer('target_std', torch.tensor(float(target_std)))

    def forward(self, batch):
        valid = (~batch['mask']).unsqueeze(-1)
        local = self.atom_projection(batch['H']).masked_fill(~valid, 0).sum(1) / valid.sum(1)
        native = (batch['native'] - self.target_mean) / self.target_std
        x = torch.cat([batch['g'], local, native[:, None], batch['extra'][:, None]], 1)
        return batch['native'] + self.head(x).squeeze(-1) * self.target_std