import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class TLU(nn.Module):
    def __init__(self, act_shape, learnable=True):
        super().__init__()
        if isinstance(act_shape, numbers.Integral):
            act_shape = (act_shape,)
        self.act_shape = tuple(act_shape)
        self.tau = nn.Parameter(torch.zeros(*self.act_shape), requires_grad=learnable)
    
    def forward(self, x):
        return F.relu(x - self.tau) + self.tau
