import math
from typing import List, Optional

import torch
import torch.nn as nn
from scipy.special import beta
from geoopt import ManifoldParameter, ManifoldTensor

from geoopt_plusplus.manifolds.stereographic.math import (
    _mobius_add,
    _mobius_scalar_mul,
    _project,
    weighted_midpoint
)

from .multinomial_logistic_regression import unidirectional_poincare_mlr


class PoincareLinear(nn.Module):
    def __init__(self, in_dim, out_dim, out_split=1, bias=True, ball=None, gain=1.):
        super(PoincareLinear, self).__init__()
        gain = 1. ###
        self.ball = ball
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_split = out_split
        weight = torch.empty(in_dim, out_dim).normal_( 
            mean=0, std=(2 * self.in_dim * self.out_dim / out_split) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.out_dim / out_split / 2, 1 / 2)
        self.beta_n = beta(self.out_dim / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        x = poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c,
            # out_split=self.out_split)
            out_split=1)
        if self.out_split > 1:
            size = x.size()
            x = self.ball.logmap0(x).contiguous().view(*size[:-1], self.out_split, size[-1] // self.out_split)
            x = self.ball.expmap0(x * self.beta_ni / self.beta_n)
        return x

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, out_split={}, bias={}'.format(
            self.in_dim, self.out_dim, self.out_split, self.bias.requires_grad
        )


class PoincareConcatLinear(nn.Module):
    def __init__(self, in_stacks, in_dim, out_dim, bias=True, ball=None, gain=1.):
        super().__init__()
        gain = 1. ###
        self.ball = ball
        self.in_stacks = in_stacks
        self.in_dim = in_dim
        self.out_dim = out_dim
        weight = torch.empty(in_stacks * in_dim, out_dim).normal_( 
            mean=0, std=1. / (2 * self.in_dim * in_stacks * self.out_dim) ** 0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.in_dim / 2, 1 / 2)
        self.beta_n = beta(self.in_dim * self.in_stacks / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        size = x.size()
        x = self.ball.logmap0(x).contiguous().view(*size[:-2], self.in_stacks * self.in_dim)
        x = self.ball.expmap0(x * self.beta_n / self.beta_ni)
        return poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c)
    
    def extra_repr(self):
        return (f'in_stacks={self.in_stacks},'
        f' in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias.requires_grad}')



#@torch.jit.script
def poincare_linear(x, weight_g, weight_v, bias, c, out_split : int = 1):
    rc = c.sqrt()
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    x = (rc * x).sinh() / rc
    if out_split > 1:
        size = x.size()
        x = x.view(*size[:-1], out_split, size[-1] // out_split)

    return _project(x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt()), -c, dim=-1)
