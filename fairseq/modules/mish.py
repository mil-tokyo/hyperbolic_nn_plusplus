import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametericMish(nn.Module):
    def __init__(self, act_shape, learnable=True):
        super(ParametericMish, self).__init__()
        if isinstance(act_shape, numbers.Integral):
            act_shape = (act_shape,)
        self.act_shape = tuple(act_shape)
        self.alpha = nn.Parameter(torch.zeros(*self.act_shape), requires_grad=learnable)
    
    def forward(self, x):
        return x * (F.softplus(self.alpha.exp() * x)).tanh()
