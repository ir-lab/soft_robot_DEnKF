from __future__ import absolute_import
from __future__ import print_function

import torch

AVAI_OPTIMS = ['adamw']


def build_optimizer(model, name, optim='adamw', lr=0.0003, weight_decay=5e-04, adam_eps=1e-3):
    if optim not in AVAI_OPTIMS:
        raise ValueError('Unsupported optim: {}. Must be one of {}'.format(optim, AVAI_OPTIMS))

    if optim == 'adamw':
        optimizer = torch.optim.AdamW([{'params': model.parameters(),
                                    'weight_decay': weight_decay,
                                    'lr': lr}],
                                    lr=lr, eps=adam_eps)
    return optimizer
