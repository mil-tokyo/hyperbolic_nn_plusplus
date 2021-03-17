# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from geoopt import ManifoldParameter, ManifoldTensor


class PoincareLearnedPositionalEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, init_std=1e-2, ball=None):
        super().__init__()
        self.init_std = init_std
        self.ball = ball
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.weight = ManifoldParameter(torch.empty(num_embeddings, embedding_dim), manifold=ball)
        self.reset_parameters()
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)

    def reset_parameters(self):
        with torch.no_grad():
            direction = torch.randn_like(self.weight)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
            distance = torch.empty(self.num_embeddings, 1).normal_(std=self.init_std / self.ball.c.data.sqrt())
            self.weight.data.copy_(self.ball.expmap0(direction * distance))
            # self.weight.data.copy_(direction * distance)
            if self.padding_idx is not None:
                self.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
        )
