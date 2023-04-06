# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.models.model_utils import SqueezeAndExcitation, SqueezeAndExcitationWeight


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


class SqueezeAndExciteReweigh(nn.Module):
    def __init__(self, temp, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteReweigh, self).__init__()
        self.temp = temp
        self.se = SqueezeAndExcitationWeight(channels_in*2, activation=activation)
        self.act = nn.Sigmoid()

    def forward(self, rgb, depth, hard=False, prev_weight=None, random=False, test=False):
        if random:
            bs = rgb.shape[0]
            b0 = torch.randint(0, 2, (bs, ))
            b1 = 1 - b0
            w_norm = torch.stack([b0, b1], dim=1).to(rgb.device)

        else:
            x = torch.concat([rgb, depth], dim=1)
            w = self.act(self.se(x))
            # print(w)
            if test:
                # w = (w > 0.5).float()
                w = torch.stack([w, 1 - w], dim=1)
                w_norm = F.gumbel_softmax(w / self.temp, hard=True)
            else:
                # self.cnt += 1
                # self.mean = self.mean + (w.mean().detach() - self.mean) / self.cnt
                w = torch.stack([w, 1 - w], dim=1)
                # print(w_norm)
                w_norm = F.gumbel_softmax(w / self.temp, hard=hard)

            # b0 = (w < 1.0 * self.mean).float().detach() - w.detach() + w
            # w_norm = DiffSoftmax(w, tau=self.temp, hard=hard)

        if prev_weight is not None:
            b1 = w_norm[:, 1] * prev_weight
            b0 = 1 - b1
            w_norm = torch.stack([b0, b1], dim=1)
        return w_norm.view(-1, 2, 1, 1)