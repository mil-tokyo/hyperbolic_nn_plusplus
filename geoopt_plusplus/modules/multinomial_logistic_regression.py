import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter, ManifoldTensor
from geoopt_plusplus.manifolds.stereographic.math import arsinh, artanh
from geoopt_plusplus.utils import *


class UnidirectionalPoincareMLR(nn.Module):
    __constants__ = ['feat_dim', 'num_outcome']

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super(UnidirectionalPoincareMLR, self).__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome
        weight = torch.empty(feat_dim, num_outcome).normal_( 
            mean=0, std=(self.feat_dim) ** -0.5 / self.ball.c.data.sqrt())
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return unidirectional_poincare_mlr(
            x, self.weight_g, self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), self.bias, self.ball.c)
    
    def extra_repr(self):
        return 'feat_dim={}, num_outcome={}, bias={}'.format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )

class WeightTiedUnidirectionalPoincareMLR(nn.Module):
    __constants__ = ['feat_dim', 'num_outcome']

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super().__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x, weight):
        weight = weight.t()
        weight_g = weight.norm(dim=0)
        return unidirectional_poincare_mlr(
            x, weight_g, weight / weight_g.clamp_min(1e-15), self.bias, self.ball.c)
    
    def extra_repr(self):
        return 'feat_dim={}, num_outcome={}, bias={}'.format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * arsinh(
        (2. * torch.matmul(rcx, z_unit) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / torch.clamp_min(1. - cx2, 1e-15))

