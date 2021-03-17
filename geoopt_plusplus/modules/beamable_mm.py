# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn



class PoincareBeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """
    def __init__(self, beam_size=None, ball=None):
        super().__init__()
        self.ball = ball
        self.beam_size = beam_size

    def forward(self, value, weight):
        # self.ball.weighted_midpoint_bmm(encoder_out[1], x)
        if (
            not self.training and           # test mode
            self.beam_size is not None and  # beam size is set
            weight.dim() == 3 and           # only support batched input
            weight.size(1) == 1             # single time step update
        ):
            bsz, beam = weight.size(0), self.beam_size

            # bsz x 1 x nhu --> bsz/beam x beam x nhu
            weight = weight[:, 0, :].unfold(0, beam, beam).transpose(2, 1)

            # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
            value = value.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
            if weight.size(0) == 1:
                output = self.ball.weighted_midpoint_bmm(value[0, :, :], weight[0, :, :])
            else:
                output = self.ball.weighted_midpoint_bmm(value, weight)
            return output.view(bsz, 1, -1)
        else:
            return self.ball.weighted_midpoint_bmm(value, weight)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size
