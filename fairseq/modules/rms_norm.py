# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    __constants__ = ['normalized_shape', 'normalized_size', 'normalized_dim', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-15, elementwise_affine=True, **kwargs):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.normalized_size = float(np.prod(self.normalized_shape))

        self.normalized_dim = [-i-1 for i in range(len(self.normalized_shape))]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x):
        # only rescaling
        if self.elementwise_affine:
            return rms_norm_w_weight(x, self.normalized_size, self.normalized_dim, self.eps, self.weight)
        else:
            return rms_norm_wo_weight(x, self.normalized_size, self.normalized_dim, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


@torch.jit.script
def rms_norm_w_weight(x, size: float, dim: List[int], eps: float, weight):
    return x / (x.pow(2) / size).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt() * weight

@torch.jit.script
def rms_norm_wo_weight(x, size: float, dim: List[int], eps: float):
    return x / (x.pow(2) / size).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt()
