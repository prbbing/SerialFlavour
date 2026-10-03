"""NLP-only upstream options missing from shared fit; orchestration stays common."""
import csv
import math
import time
import torch

from experiments.nlp_massive_xlm.evaluate import metrics
from pipeline.fit import save_checkpoint
from pipeline.io import write_json
from pipeline.runtime import to_device

def fit(model, train_loader, val_loader, loss, directory, settings, metadata, device):
    directory.mkdir(parents=True, exist_ok=True)
    model.to(device)
    accumulation = int(settings['gradient_accumulation_steps'])
    if accumulation < 1 or settings['epochs'] < 1 or not len(train_loader):
        raise ValueError('invalid epoch/accumulation/loader budget')
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * settings['epochs']
    warmup = math.ceil(total_updates * settings['warmup_ratio'])
    if not 0 <= settings['warmup_ratio'] < 1:
        raise ValueError('warmup_ratio must lie in [0, 1)')
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith('.bias') or 'LayerNorm.weight' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': settings['weight_decay']},
                                  {'params': no_decay, 'weight_decay': 0.0}], lr=settings['learning_rate'])
    best_accuracy, best_nll, best_epoch = -math.inf, math.inf, 0
    completed_updates, history = 0, []
    started = time.perf_counter()
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective_sum, samples_seen = 0.0, 0
        for j, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            objective, samples = loss(model, batch)
            if not torch.isfinite(objective):
                raise FloatingPointError('nonfinite NLP objective')
            window_start = (j // accumulation) * accumulation
            window_samples = min(accumulation * train_loader.batch_size,
                                 len(train_loader.dataset) - window_start * train_loader.batch_size)
            # Correctly scale short final accumulation windows and microbatches.
            (objective * (samples / window_samples)).backward()
            objective_sum += objective.detach().item() * samples
            samples_seen += samples
            if (j + 1) % accumulation == 0 or j + 1 == len(train_loader):
                scale = ((completed_updates + 1) / warmup if completed_updates < warmup else
                         (total_updates - completed_updates) / max(total_updates - warmup, 1))
                for group in optimizer.param_groups:
                    group['lr'] = settings['learning_rate'] * scale
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1
        validation = metrics(model, val_loader, device)
        accuracy, nll = validation['accuracy'], validation['main_nll']
        if not math.isfinite(accuracy) or not math.isfinite(nll):
            raise FloatingPointError('nonfinite validation metric')
        if accuracy > best_accuracy or (accuracy == best_accuracy and nll < best_nll):
            best_accuracy, best_nll, best_epoch = accuracy, nll, epoch
            save_checkpoint(directory / 'best.pt', {'state_dict': model.state_dict(), 'metadata': metadata,
                'epoch': epoch, 'validation_metric': accuracy, 'validation_nll': nll, 'selection_mode': 'max'})
        record = {'epoch': epoch, 'train_loss': objective_sum / samples_seen, 'validation_metric': accuracy,
                  'validation_nll': nll, 'optimizer_updates': completed_updates,
                  'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter() - started}
        history.append(record)
        print(f'{directory}: epoch {epoch}, loss={record["train_loss"]:.5f}, A_val acc={accuracy:.5f}, nll={nll:.5f}', flush=True)
        if epoch - best_epoch >= settings['early_stopping_patience']:
            break
    write_json(directory / 'history.json', history)
    with (directory / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(directory / 'training_manifest.json', {
        **metadata, 'training': settings, 'best_validation_metric': best_accuracy, 'best_validation_nll': best_nll,
        'best_epoch': best_epoch, 'epochs_run': len(history), 'early_stopped': len(history) < settings['epochs'],
        'selection_tie_break': 'lowest intent NLL on A_val; earliest if both equal',
        'optimizer_updates': completed_updates, 'planned_optimizer_updates': total_updates, 'warmup_updates': warmup,
        'effective_batch_size': train_loader.batch_size * accumulation, 'precision': 'float32; no AMP',
        'slot_reduction': 'sample-weighted microbatch mean of first-subword slot CE within accumulation window',
        'bias_LayerNorm_weight_decay': 0.0,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'seconds': time.perf_counter() - started})
    return [directory / n for n in ('best.pt', 'history.json', 'history.csv', 'training_manifest.json')]
