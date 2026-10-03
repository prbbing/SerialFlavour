"""Use the unmodified author class; expose H from its same encoder forward."""
import torch
from torch import nn
from transformers import XLMRobertaConfig

from experiments.nlp_massive_xlm.official_xlmr import XLMRIntentClassSlotFill
from experiments.nlp_massive_xlm.data import processed_dir, pretrained_assets, pretrained_directory, pretrained_enabled
from pipeline.io import read_json

class Upstream(nn.Module):
    def __init__(self, model_config, labels, multi=True):
        super().__init__()
        model_config = dict(model_config)
        initialization = model_config.pop('initialization', 'random')
        gradient_checkpointing = model_config.pop('gradient_checkpointing', False)
        if initialization not in ('random', 'pretrained_base'):
            raise ValueError('unknown model initialization')
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
        if gradient_checkpointing:
            self.network.xlmr.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

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

def build(context, variant, initialize=False):
    labels = read_json(processed_dir(context) / 'labels.json')
    model = Upstream(context.config['model'], labels, multi=variant == 'multi_task')
    if initialize and pretrained_enabled(context):
        from safetensors.torch import load_file
        from experiments.nlp_massive_xlm.data import BASE_WEIGHTS_SHA256
        pretrained_assets(context, allow_download=False)
        path = pretrained_directory(context) / 'model.safetensors'
        base_config = read_json(pretrained_directory(context) / 'config.json')
        for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size',
                    'layer_norm_eps', 'max_position_embeddings', 'type_vocab_size', 'hidden_act',
                    'pad_token_id', 'bos_token_id', 'eos_token_id', 'position_embedding_type'):
            if getattr(model.network.config, key) != base_config[key]:
                raise ValueError(f'base architecture mismatch for {key}; no partial weight loading')
        weights = load_file(str(path), device='cpu')
        encoder = {k.removeprefix('roberta.'): v for k, v in weights.items() if k.startswith('roberta.')}
        if not encoder:
            raise ValueError('base MLM weights contain no roberta encoder')
        reconstructed_buffers = []
        # Older Transformers releases persisted these derived buffers; 4.44.2
        # reconstructs them. Validate their values before excluding them, and
        # never ignore a learned encoder parameter.
        for key in ('embeddings.position_ids', 'embeddings.token_type_ids'):
            if key in encoder:
                expected = getattr(model.network.xlmr.embeddings, key.split('.')[-1])
                if not torch.equal(encoder[key], expected):
                    raise ValueError(f'base derived buffer mismatch: {key}')
                del encoder[key]
                reconstructed_buffers.append(key)
        missing, unexpected = model.network.xlmr.load_state_dict(encoder, strict=False)
        if set(missing) != {'pooler.dense.weight', 'pooler.dense.bias'} or unexpected:
            raise ValueError(f'base encoder weight mismatch: {missing}, {unexpected}')
        # Base MLM contains no trained sentence pooler; it is unused and frozen.
        model.pretrained_metadata = {'model_id': 'FacebookAI/xlm-roberta-base', 'sha256': BASE_WEIGHTS_SHA256,
                                     'loaded_encoder_parameters': len(encoder), 'missing_unused_pooler': missing,
                                     'validated_reconstructed_buffers': reconstructed_buffers}
        del weights, encoder
    return model

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
