# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import beta
from .linear import poincare_linear



class PoincareConvTBC(nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, ball=None):
        super().__init__()
        self.ball = ball
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_ = (0, 0, 0, 0, padding, padding)

        weight = torch.empty(
            self.kernel_size * in_channels, out_channels).normal_(
                std=(2 * in_channels * self.kernel_size * out_channels) ** -0.5)
        weight_g = weight.norm(dim=0).clamp_min(1e-15)
        self.weight_g = nn.Parameter(weight_g)
        self.weight_v = nn.Parameter(weight / weight_g)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.beta_ni = beta(self.in_channels / 2, 1 / 2)
        self.beta_n = beta(self.in_channels * self.kernel_size / 2, 1 / 2)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        assert x.dim() == 3

        # beta-concatenation
        x = self.ball.logmap0(x) * self.beta_n / self.beta_ni
        x = F.pad(x, self.padding_)
        olen = x.size(0) - self.kernel_size + 1

        x = torch.cat([x.narrow(0, k, olen) for k in range(self.kernel_size)], dim=-1)
        x = self.ball.expmap0(x) 

        return poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c,
            out_split=1)
