"""Width-reduced author SegNet MTAN/STAN; no pretrained weights.

Adapted from lorenmt/mtan im2im_pred/model_segnet_{mtan,stan}.py (MIT).
Five scales, shared attention transitions, pooling indices and prediction heads
follow the author implementation. Width and task count are configurable.
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())


def attention(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 1), nn.BatchNorm2d(b), nn.ReLU(),
                         nn.Conv2d(b, b, 1), nn.BatchNorm2d(b), nn.Sigmoid())


def head(channels, outputs):
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                         nn.Conv2d(channels, outputs, 1))


class MTAN(nn.Module):
    def __init__(self, widths, multi_task=True):
        super().__init__()
        if len(widths) != 5 or widths[-1] != widths[-2]:
            raise ValueError('Author SegNet MTAN requires five scales with equal last two widths')
        self.multi_task = multi_task
        self.tasks = 3 if multi_task else 1
        self.encoder_block = nn.ModuleList([conv(3 if i == 0 else widths[i-1], widths[i]) for i in range(5)])
        self.decoder_block = nn.ModuleList([conv(widths[i], widths[max(i-1, 0)]) for i in range(5)])
        self.conv_block_enc = nn.ModuleList([
            nn.Sequential(*[conv(widths[i], widths[i]) for _ in range(1 if i < 2 else 2)]) for i in range(5)])
        self.conv_block_dec = nn.ModuleList([
            nn.Sequential(*[conv(widths[max(i-1, 0)], widths[max(i-1, 0)]) for _ in range(1 if i < 2 else 2)]) for i in range(5)])
        self.encoder_att = nn.ModuleList([nn.ModuleList([
            attention(widths[i] * (1 if i == 0 else 2), widths[i]) for i in range(5)]) for _ in range(self.tasks)])
        self.decoder_att = nn.ModuleList([nn.ModuleList([
            attention(widths[i] + widths[max(i-1, 0)], widths[max(i-1, 0)]) for i in range(5)]) for _ in range(self.tasks)])
        self.encoder_block_att = nn.ModuleList([conv(widths[i], widths[min(i+1, 4)]) for i in range(5)])
        self.decoder_block_att = nn.ModuleList([conv(widths[i], widths[max(i-1, 0)] if i < 4 else widths[4]) for i in range(5)])
        self.pred_task1 = head(widths[0], 13)
        if multi_task:
            self.pred_task2 = head(widths[0], 1)
            self.pred_task3 = head(widths[0], 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        enc, pooled, indices, dec, unpooled = [], [], [], [], []
        for i in range(5):
            first = self.encoder_block[i](x if i == 0 else pooled[-1])
            second = self.conv_block_enc[i](first)
            pool, index = F.max_pool2d(second, 2, return_indices=True)
            enc.append((first, second))
            pooled.append(pool)
            indices.append(index)
        for j in range(5):
            i = 4-j
            up = F.max_unpool2d(pooled[-1] if j == 0 else dec[-1], indices[i], 2, output_size=enc[i][1].shape)
            unpooled.append(up)
            dec.append(self.conv_block_dec[i](self.decoder_block[i](up)))
        hidden = []
        for task in range(self.tasks):
            previous = None
            for i in range(5):
                inputs = enc[i][0] if i == 0 else torch.cat((enc[i][0], previous), 1)
                masked = self.encoder_att[task][i](inputs) * enc[i][1]
                previous = F.max_pool2d(self.encoder_block_att[i](masked), 2)
            for j in range(5):
                i = 4-j
                up = F.interpolate(previous, size=unpooled[j].shape[-2:], mode='bilinear', align_corners=True)
                up = self.decoder_block_att[i](up)
                previous = self.decoder_att[task][i](torch.cat((unpooled[j], up), 1)) * dec[j]
            hidden.append(previous)
        output = {'shared': dec[-1], 'semantic_hidden': hidden[0], 'logits': self.pred_task1(hidden[0])}
        if self.multi_task:
            output.update(depth=self.pred_task2(hidden[1]), normal=F.normalize(self.pred_task3(hidden[2]), dim=1, eps=1e-8),
                          aux_hidden=torch.cat(hidden[1:], 1))
        return output
