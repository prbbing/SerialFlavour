"""Use the unmodified author class; expose H from its same encoder forward."""
import torch
from torch import nn
from transformers import XLMRobertaConfig

from experiments.nlp_massive_xlm.official_xlmr import XLMRIntentClassSlotFill
from experiments.nlp_massive_xlm.data import processed_dir
from pipeline.io import read_json

class Upstream(nn.Module):
    def __init__(self, model_config, labels, multi=True):
        super().__init__()
        config = XLMRobertaConfig(vocab_size=labels['vocab_size'], pad_token_id=1, bos_token_id=0, eos_token_id=2,
                                 max_position_embeddings=514, type_vocab_size=1, **model_config)
        config.head_intent_pooling = 'mean'
        config.head_num_layers = 1
        config.head_layer_dim = config.hidden_size
        config.slot_loss_coef = 1.0
        config._attn_implementation = 'eager'
        self.network = XLMRIntentClassSlotFill(config, dict(enumerate(labels['intents'])), dict(enumerate(labels['slots'])))
        self.network.xlmr.pooler.requires_grad_(False)  # author heads use sequence H, not pooler
        if not multi:
            self.network.slot_classifier.requires_grad_(False)

    def forward(self, batch):
        captured = {}
        def capture(module, args, output):
            captured['embedding'] = output[0]
        handle = self.network.xlmr.register_forward_hook(capture)
        try:
            _, (intent, slot) = self.network(batch['input_ids'], batch['attention_mask'], None, None)
        finally:
            handle.remove()
        return {'embedding': captured['embedding'], 'logits': intent, 'slot_logits': slot}

def build(context, variant):
    labels = read_json(processed_dir(context) / 'labels.json')
    return Upstream(context.config['model'], labels, multi=variant == 'multi_task')

class Readout(nn.Module):
    def __init__(self, channels, classes, width=32):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(channels, width), nn.GELU())
        self.output = nn.Linear(2 * width, classes)
        # The frozen native logits are available to every recipe; epoch zero
        # reproduces native exactly and is eligible for B_val selection.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, batch):
        h = self.projection(batch['features'])
        mask = batch['attention_mask'].bool().unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        maximum = h.masked_fill(~mask, -torch.inf).max(1).values
        return batch['native_logits'] + self.output(torch.cat((mean, maximum), -1))
